'''
NEVPT2 solver for QC-dmet.

Runs CASSCF followed by strongly-contracted NEVPT2 (PySCF's mrpt.NEVPT) on the
DMET embedding cluster (impurity + bath). The cluster is built from the
embedding integrals exactly as in the CASSCF solver, so the NEVPT2 perturber
space is the cluster's inactive + virtual orbitals — the DMET bath's
representation of the environment — not the full physical molecule. This is the
embedded multireference approach used by libDMET / Vayesta style cluster
solvers: the bath captures static entanglement with the environment, and the
correlated solver treats the cluster as a closed problem.

For multi-state runs:
    1. SA-CASSCF on the cluster to get optimized orbitals.
    2. Multi-root CASCI with those MOs.
    3. Per-state SC-NEVPT2 via mrpt.NEVPT(mc_casci, root=i).

oneshot-dmet only.
'''

import numpy as np
from pyscf import ao2mo, gto, scf, mcscf, mrpt
from ..utils import silent_stdout, nullcontext

_eV = 27.21138602


def solve(const, oei, fock, tei, norb, nel, nimp, dm_guess_rhf,
          ncas, nelecas,
          nstates=1, sa_weights=None,
          chempot_imp=0.0,
          casscf_kwargs=None, nevpt2_kwargs=None,
          spin=None, printoutput=True):
    '''
    Run CASSCF + NEVPT2 on the DMET embedding cluster.

    Parameters
    ----------
    const         : float  – embedding constant (added by the dmet driver)
    oei           : ndarray (norb, norb) – embedding one-electron integrals
    fock          : ndarray (norb, norb) – embedding Fock (one-shot: == oei + ...)
    tei           : ndarray – embedding two-electron integrals (norb^4)
    norb          : int  – total cluster orbitals (impurity + bath)
    nel           : int  – total cluster electrons
    nimp          : int  – impurity orbitals (first nimp of norb)
    dm_guess_rhf  : ndarray – RHF/ROHF density-matrix guess in the cluster basis
    ncas          : int  – active orbitals
    nelecas       : int  – active electrons
    nstates       : int  – 1 = single-state; >1 = state-averaged
    sa_weights    : list or None – SA weights (uniform if None)
    chempot_imp   : float – impurity chemical potential (subtracted on diagonal)
    casscf_kwargs : dict – extra attributes set on the CASSCF object
    nevpt2_kwargs : dict – extra attributes set on each mrpt.NEVPT object
    spin          : int or None – cluster spin (2S); defaults to nel % 2
    printoutput   : bool

    Returns
    -------
    e_tot      : ndarray  – NEVPT2 total energy per state (Ha)
    e_corr     : ndarray  – NEVPT2 correlation energy per state (Ha)
    mc         : mcscf object – the CASSCF/CASCI object used for NEVPT2
    nevpt_objs : list of mrpt.NEVPT – one per state
    '''
    casscf_kwargs = casscf_kwargs or {}
    nevpt2_kwargs = nevpt2_kwargs or {}

    _spin = spin if spin is not None else (nel % 2)
    _use_rohf = (_spin != 0)

    fock_copy = fock.copy()
    if chempot_imp != 0.0:
        for orb in range(nimp):
            fock_copy[orb, orb] -= chempot_imp

    ctx = silent_stdout() if not printoutput else nullcontext()

    with ctx:
        # Embedded mean field on the (impurity + bath) cluster. Identical
        # construction to the CASSCF solver: a dummy Mole carries only the
        # electron count and spin, while the embedding integrals are injected
        # via get_hcore / get_ovlp / _eri so the SCF lives in the norb-orbital
        # cluster space.
        mol = gto.Mole()
        mol.build(verbose=0)
        mol.atom.append(('C', (0, 0, 0)))
        mol.nelectron = nel
        mol.spin = _spin
        mol.incore_anyway = True

        mf = scf.ROHF(mol) if _use_rohf else scf.RHF(mol)
        mf.get_hcore = lambda *args: fock_copy
        mf.get_ovlp  = lambda *args: np.eye(norb)
        mf._eri      = ao2mo.restore(8, tei, norb)
        mf.scf(dm_guess_rhf)
        if not mf.converged:
            mf = mf.newton()
            mf.scf(mf.make_rdm1())
        if _use_rohf:
            print(f"nevpt2::solve : embedded ROHF (spin={_spin}, nel={nel}, norb={norb})")

        if nstates == 1:
            mc = mcscf.CASSCF(mf, ncas, nelecas)
            mc.verbose = 5 if printoutput else 0
            for key, val in casscf_kwargs.items():
                setattr(mc, key, val)
            mc.kernel()

            print(f"\nnevpt2::solve : embedded CASSCF energy = {mc.e_tot:.10f} Ha")

            nevpt_obj = mrpt.NEVPT(mc, root=0)
            nevpt_obj.verbose = 5 if printoutput else 0
            for key, val in nevpt2_kwargs.items():
                setattr(nevpt_obj, key, val)
            # mrpt.NEVPT.kernel() calls self.canonicalize(..., cas_natorb=True), which
            # invokes mc.cas_natorb() → orth.orth_ao(mc.mol, 'meta_lowdin') → fails on
            # the dummy mol (no real AO basis). Override to skip the natorb step; Fock
            # diagonalization still canonicalizes inactive/external orbitals correctly.
            _mc_ref = mc
            nevpt_obj.canonicalize = lambda mo, ci, eris=None, sort=False, cas_natorb=True, casdm1=None, verbose=None: \
                _mc_ref.canonicalize(mo, ci, eris, sort, False, casdm1, verbose)
            e_c = nevpt_obj.kernel()

            e_tot  = np.array([nevpt_obj.e_tot])
            e_corr = np.array([e_c])
            nevpt_objs = [nevpt_obj]

        else:
            if sa_weights is None:
                sa_weights = [1.0 / nstates] * nstates
            sa_weights = np.array(sa_weights, dtype=float)
            sa_weights /= sa_weights.sum()

            # Step 1: SA-CASSCF for optimized orbitals
            mc_sa = mcscf.CASSCF(mf, ncas, nelecas)
            mc_sa = mcscf.state_average_(mc_sa, weights=sa_weights.tolist())
            mc_sa.verbose = 5 if printoutput else 0
            for key, val in casscf_kwargs.items():
                setattr(mc_sa, key, val)
            mc_sa.kernel()
            sa_mo = mc_sa.mo_coeff

            print(f"\nnevpt2::solve : embedded SA-CASSCF ({nstates} states) done.")
            for i, e in enumerate(mc_sa.e_states):
                print(f"  State {i}: {e:.10f} Ha  (weight={sa_weights[i]:.4f})")

            # Step 2: Multi-root CASCI with SA-CASSCF MOs
            mc = mcscf.CASCI(mf, ncas, nelecas)
            mc.verbose = 4 if printoutput else 0
            mc.fcisolver.nroots = nstates
            mc.kernel(sa_mo)

            print(f"\nnevpt2::solve : Multi-root CASCI energies:")
            for i, e in enumerate(mc.e_tot):
                print(f"  State {i}: {e:.10f} Ha")

            # Step 3: Per-state NEVPT2
            e_tot  = np.zeros(nstates)
            e_corr = np.zeros(nstates)
            nevpt_objs = []
            _mc_ref = mc
            for i in range(nstates):
                nevpt_i = mrpt.NEVPT(mc, root=i)
                nevpt_i.verbose = 5 if printoutput else 0
                for key, val in nevpt2_kwargs.items():
                    setattr(nevpt_i, key, val)
                nevpt_i.canonicalize = lambda mo, ci, eris=None, sort=False, cas_natorb=True, casdm1=None, verbose=None: \
                    _mc_ref.canonicalize(mo, ci, eris, sort, False, casdm1, verbose)
                e_c_i = nevpt_i.kernel()
                e_tot[i]  = nevpt_i.e_tot
                e_corr[i] = e_c_i
                nevpt_objs.append(nevpt_i)

        print(f"\nnevpt2::solve : NEVPT2 results:")
        print(f"  {'State':>5}  {'E_tot (Ha)':>16}  {'E_corr (Ha)':>14}  {'ΔE from GS (eV)':>16}")
        print("  " + "-"*56)
        for i, (et, ec) in enumerate(zip(e_tot, e_corr)):
            de_ev = (et - e_tot[0]) * _eV
            print(f"  {i:>5d}  {et:>16.10f}  {ec:>14.10f}  {de_ev:>+16.4f}")

    return e_tot, e_corr, mc, nevpt_objs


# ---------------------------------------------------------------------------
# SolverDispatcher entry point
# ---------------------------------------------------------------------------

def execute(task):
    """
    SolverDispatcher entry point for the embedded NEVPT2 solver.

    Operates entirely on the embedding-cluster integrals carried in the task
    dict (dmet_oei / dmet_fock / dmet_tei / norb / nel / nimp), so no physical
    molecule is needed.
    """
    e_tot, e_corr, mc, nevpt_objs = solve(
        task['const'],
        task['dmet_oei'],
        task['dmet_fock'],
        task['dmet_tei'],
        task['norb'],
        task['nel'],
        task['nimp'],
        task.get('dm_guess_rhf'),
        ncas=task.get('ncas'),
        nelecas=task.get('nelecas'),
        nstates=task.get('sa_nstates', 1),
        sa_weights=task.get('sa_weights'),
        chempot_imp=task.get('chempot_imp', 0.0),
        casscf_kwargs=task.get('casscf_kwargs', {}),
        nevpt2_kwargs=task.get('nevpt2_kwargs', {}),
        spin=task.get('spin'),
    )

    # 1-RDM (cluster orbital basis) for the dmet driver: prefer the NEVPT2
    # relaxed density if available, else the CASSCF/CASCI density.
    nevpt_gs = nevpt_objs[0]
    if hasattr(nevpt_gs, 'onerdm') and nevpt_gs.onerdm is not None:
        rdm1 = nevpt_gs.onerdm
    else:
        rdm1 = mc.make_rdm1()

    nevpt2_res = {
        'e_tot'      : e_tot,
        'e_corr'     : e_corr,
        'mc'         : mc,
        'nevpt_objs' : nevpt_objs,
    }
    return e_tot[0], rdm1, nevpt2_res
