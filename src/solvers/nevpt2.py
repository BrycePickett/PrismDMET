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
from pyscf import ao2mo, fci as pyscf_fci, gto, scf, mcscf, mrpt
from ..utils import silent_stdout, nullcontext

_eV = 27.21138602


def solve(const, oei, fock, tei, norb, nel, nimp, dm_guess_rhf,
          ncas, nelecas,
          nstates=1, sa_weights=None,
          chempot_imp=0.0,
          casscf_kwargs=None, nevpt2_kwargs=None,
          spin=None, oei_s=None,
          printoutput=True):
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
    oei_s         : ndarray (norb, norb) or None – spin-asymmetry correction
                    0.5*(F_alpha - F_beta) in the embedding basis; injected into
                    all CASSCF/CASCI CI solvers via fix_casscf_for_nonsinglet_env.
                    Applied only to the reference (consistent with standard
                    ROHF-NEVPT2 and with pDMET); NEVPT2 perturber denominators
                    use spin-free canonical orbital energies.
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

            if _use_rohf and oei_s is not None and not np.all(np.abs(oei_s) < 1e-8):
                _uhf_fci = pyscf_fci.direct_uhf.FCI()
                _uhf_fci.verbose = mc.fcisolver.verbose
                mc.fcisolver = _uhf_fci
            if oei_s is not None:
                from .qcsolver_utils import fix_casscf_for_nonsinglet_env
                mc = fix_casscf_for_nonsinglet_env(mc, oei_s)

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
            # Swap base FCI solver BEFORE state_average_ so the SA wrapper inherits it.
            # direct_spin1.FCI cannot accept [h1e_a, h1e_b]; direct_uhf.FCI handles it.
            if _use_rohf and oei_s is not None and not np.all(np.abs(oei_s) < 1e-8):
                _uhf_fci = pyscf_fci.direct_uhf.FCI()
                _uhf_fci.verbose = mc_sa.fcisolver.verbose
                mc_sa.fcisolver = _uhf_fci
            mc_sa = mcscf.state_average_(mc_sa, weights=sa_weights.tolist())
            mc_sa.verbose = 5 if printoutput else 0
            for key, val in casscf_kwargs.items():
                setattr(mc_sa, key, val)
            if oei_s is not None:
                from .qcsolver_utils import fix_casscf_for_nonsinglet_env
                mc_sa = fix_casscf_for_nonsinglet_env(mc_sa, oei_s)

            mc_sa.kernel()
            sa_mo = mc_sa.mo_coeff

            print(f"\nnevpt2::solve : embedded SA-CASSCF ({nstates} states) done.")
            for i, e in enumerate(mc_sa.e_states):
                print(f"  State {i}: {e:.10f} Ha  (weight={sa_weights[i]:.4f})")

            # Step 2: Multi-root CASCI with SA-CASSCF MOs
            mc = mcscf.CASCI(mf, ncas, nelecas)
            mc.verbose = 4 if printoutput else 0
            mc.fcisolver.nroots = nstates

            if oei_s is not None:
                from .qcsolver_utils import fix_casscf_for_nonsinglet_env
                if _use_rohf and not np.all(np.abs(oei_s) < 1e-8):
                    _uhf_fci = pyscf_fci.direct_uhf.FCI()
                    _uhf_fci.verbose = mc.fcisolver.verbose
                    mc.fcisolver = _uhf_fci
                    mc.fcisolver.nroots = nstates
                mc = fix_casscf_for_nonsinglet_env(mc, oei_s)

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
# mf_real reconstruction helper (used by the QD-NEVPT2 solver, which still
# runs on the real physical molecule). Kept here for that solver's import.
# ---------------------------------------------------------------------------

def _reconstruct_mf_from_task(task):
    """
    Reconstruct a minimal PySCF RHF object from serialized task dict arrays.

    Because PySCF Mole and SCF objects cannot be pickled across process
    boundaries, dmet.doexact() serializes the physical MF state as:
        task['mol_dumps']   : str   - from pyscf.gto.Mole.dumps()
        task['mf_mo_coeff'] : ndarray
        task['mf_mo_energy']: ndarray
        task['mf_mo_occ']   : ndarray
        task['mf_e_tot']    : float

    The returned object can be passed directly into a solver as mf_real.
    No SCF iterations are re-run.
    """
    import pyscf.gto
    import pyscf.scf

    mol = pyscf.gto.Mole.loads(task['mol_dumps'])
    mol.build(verbose=0)

    mf = pyscf.scf.ROHF(mol) if mol.spin != 0 else pyscf.scf.RHF(mol)
    # Inject pre-computed MO state — no SCF cycles run.
    mf.mo_coeff  = task['mf_mo_coeff']
    mf.mo_energy = task['mf_mo_energy']
    mf.mo_occ    = task['mf_mo_occ']
    mf.e_tot     = task['mf_e_tot']
    return mf


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
        oei_s=task.get('oei_s'),
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
