'''
NEVPT2 solver for QC-dmet.

Runs CASSCF followed by strongly-contracted NEVPT2 using PySCF's mrpt.NEVPT.
Supports single-state (nstates=1) and multi-state (nstates > 1) calculations.

For multi-state runs, the correct procedure is:
    1. SA-CASSCF to get optimized orbitals.
    2. Multi-root CASCI with those MOs.
    3. Per-state SC-NEVPT2 via mrpt.NEVPT(mc_casci, root=i).

This solver operates on the real physical molecule (mf_real) because PySCF's
NEVPT2 requires actual mol/mf/mc objects. It is oneshot-dmet only.
'''

import numpy as np
from pyscf import gto, scf, mcscf, mrpt
from ..utils import silent_stdout, nullcontext

_eV = 27.21138602


def solve(mf_real, ncas, nelecas,
          nstates=1, sa_weights=None,
          root=0,
          casscf_kwargs=None,
          nevpt2_kwargs=None,
          mo_guess=None,
          mo_spin_ref=None,
          printoutput=True):
    '''
    Run CASSCF + NEVPT2 on the real physical molecule.

    For a single ground-state calculation (nstates=1), the standard
    CASSCF → NEVPT2 pipeline is used.

    For excited states (nstates > 1), the correct multi-step procedure is:
        1. SA-CASSCF with nstates states to get optimized MOs.
        2. Multi-root CASCI with those MOs (nroots = nstates).
        3. Per-state SC-NEVPT2 via mrpt.NEVPT(mc_casci, root=i).

    Parameters
    ----------
    mf_real       : pyscf.scf.hf.RHF  – converged RHF on the real molecule
    ncas          : int  – number of active orbitals
    nelecas       : int  – number of active electrons
    nstates       : int  – number of states (1 = single-state; >1 = multi-state)
    sa_weights    : list of float or None  – SA weights (uniform if None)
    root          : int  – which state to return as impurity_energy (0 = GS)
    casscf_kwargs : dict – extra attributes set on the CASSCF object
    nevpt2_kwargs : dict – extra attributes set on each mrpt.NEVPT object
    printoutput   : bool

    Returns
    -------
    e_tot      : ndarray  – total NEVPT2 energy per state (Ha)
    e_corr     : ndarray  – NEVPT2 correlation energy per state (Ha)
    mc         : mcscf object  – the CASCI/CASSCF object used for NEVPT2
    nevpt_objs : list of mrpt.NEVPT  – one per state
    '''
    casscf_kwargs = casscf_kwargs or {}
    nevpt2_kwargs = nevpt2_kwargs or {}

    # When mo_spin_ref is set ('alpha' or 'beta'), use that UKS spin channel as the
    # CASSCF initial reference. sort_mo explicitly places the spin-HOMO and the
    # ncas-1 orbitals above it into the active window, bypassing the ROHF energy
    # ordering that would otherwise select a lower-lying Cu 3d virtual instead of
    # the vacancy Pz/Pz* pair.
    # mo_spin_ref=None: use whatever mo_guess was passed in (or ROHF default if None).
    if mo_spin_ref is not None and mo_guess is None:
        _spin_idx = {'alpha': 0, 'beta': 1}.get(mo_spin_ref)
        if _spin_idx is None:
            raise ValueError(f"nevpt2::solve: mo_spin_ref must be 'alpha', 'beta', or None; got {mo_spin_ref!r}")
        if np.ndim(getattr(mf_real, 'mo_occ', None)) != 2:
            raise ValueError("nevpt2::solve: mo_spin_ref requires a UKS/UHF reference (spin-resolved mo_occ)")
        _mo_ref  = mf_real.mo_coeff[_spin_idx]
        _occ_ref = mf_real.mo_occ[_spin_idx]
        _homo    = int(np.where(_occ_ref > 0)[0][-1])
        _caslst  = [_homo + i + 1 for i in range(ncas)]   # 1-indexed: HOMO, LUMO, ...
        _mc_ref  = mcscf.CASSCF(mf_real, ncas, nelecas)
        mo_guess = mcscf.addons.sort_mo(_mc_ref, _mo_ref, caslst=_caslst, base=1)
        print(f"nevpt2::solve : mo_spin_ref={mo_spin_ref!r} — "
              f"HOMO={_homo} (0-idx), caslst={_caslst}")

    ctx = silent_stdout() if not printoutput else nullcontext()

    with ctx:
        if nstates == 1:
            mc = mcscf.CASSCF(mf_real, ncas, nelecas)
            mc.verbose = 5 if printoutput else 0
            for key, val in casscf_kwargs.items():
                setattr(mc, key, val)
            mc.kernel(mo_guess)

            print(f"\nnevpt2::solve : CASSCF energy = {mc.e_tot:.10f} Ha")

            nevpt_obj = mrpt.NEVPT(mc, root=0)
            nevpt_obj.verbose = 5 if printoutput else 0
            for key, val in nevpt2_kwargs.items():
                setattr(nevpt_obj, key, val)
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
            mc_sa = mcscf.CASSCF(mf_real, ncas, nelecas)
            mc_sa = mcscf.state_average_(mc_sa, weights=sa_weights.tolist())
            mc_sa.verbose = 5 if printoutput else 0
            for key, val in casscf_kwargs.items():
                setattr(mc_sa, key, val)
            mc_sa.kernel(mo_guess)
            sa_mo = mc_sa.mo_coeff

            print(f"\nnevpt2::solve : SA-CASSCF ({nstates} states) done.")
            for i, e in enumerate(mc_sa.e_states):
                print(f"  State {i}: {e:.10f} Ha  (weight={sa_weights[i]:.4f})")

            # Step 2: Multi-root CASCI with SA-CASSCF MOs
            mc = mcscf.CASCI(mf_real, ncas, nelecas)
            # Verbose 4 avoids meta-Lowdin natural-orbital printing in canonicalize(),
            # which fails on ECP boundary atoms (X-Cu, X-O) lacking ANO data.
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
            for i in range(nstates):
                nevpt_i = mrpt.NEVPT(mc, root=i)
                nevpt_i.verbose = 5 if printoutput else 0
                for key, val in nevpt2_kwargs.items():
                    setattr(nevpt_i, key, val)
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
# mf_real reconstruction helper (parallel-worker path)
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

    The returned object can be passed directly into solve() as mf_real.
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
    SolverDispatcher entry point for the NEVPT2 solver.

    Supports two transport modes for the physical SCF object:

    Serialized path (parallel): task carries mol_dumps, mf_mo_coeff,
        mf_mo_energy, mf_mo_occ, mf_e_tot. A fresh mf is reconstructed.
    Live-object path (sequential): task['mf_real'] is the live PySCF RHF
        object. Used when called from the main process directly.

    The serialized path is chosen when 'mol_dumps' is present in the task dict.
    """
    # ------------------------------------------------------------------
    # Resolve mf_real via either the serialized or live-object protocol.
    # ------------------------------------------------------------------
    if 'mol_dumps' in task:
        # Mode 1: reconstruct from serialized arrays (parallel-safe).
        mf_real = _reconstruct_mf_from_task(task)
    else:
        # Mode 2: live object passed directly (sequential / legacy path).
        mf_real = task['mf_real']

    e_tot, e_corr, mc, nevpt_objs = solve(
        mf_real,
        ncas=task.get('ncas'),
        nelecas=task.get('nelecas'),
        nstates=task.get('sa_nstates', 1),
        sa_weights=task.get('sa_weights'),
        casscf_kwargs=task.get('casscf_kwargs', {}),
        nevpt2_kwargs=task.get('nevpt2_kwargs', {}),
        mo_guess=task.get('mo_guess'),
        mo_spin_ref=task.get('mo_spin_ref'),
    )

    # Extract the 1-RDM for the dmet self-consistency loop.
    # Use the NEVPT2 object's relaxed 1-RDM if available; fall back to CASSCF.
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
