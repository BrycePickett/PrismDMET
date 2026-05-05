'''
    QD-NEVPT2 solver for QC-DMET using Prism (https://github.com/sokolov-group/prism).

    IMPORTANT: Unlike the other solvers (FCI, CCSD, CASSCF), this solver operates
    in the PHYSICAL molecular basis, not in the DMET embedding orbital basis. This
    is a fundamental requirement of Prism's interface, which needs the real PySCF
    mol / mf / mc objects (it calls mol.energy_nuc(), mol.intor_symmetric(), etc.).

    Workflow
    --------
    1. The user builds their DMET system as normal (local_integrals, fragments).
    2. Instead of calling dmet.dmet(..., method='CASSCF'), they use
       dmet.dmet(..., method='QD-NEVPT2') and supply the real mf object.
    3. Inside doexact, a full SA-CASSCF is run on the real molecule using the
       DMET-defined active space (ncas, nelecas). The active orbital space
       corresponds to the selected impurity fragment.
    4. Prism QD-NEVPT2 is run on the resulting mc object.
    5. The SA-weighted CASSCF 1-RDM (in the DMET embedding basis) is used for
       the u-matrix fitting loop; QD-NEVPT2 provides the correlated energies.

    Limitations
    -----------
    - Only compatible with one-shot DMET. Calling selfconsistent() with
      method='QD-NEVPT2' raises a RuntimeError because the Prism interface
      requires the real molecular integrals and cannot be embedded in the
      standard DMET u-matrix loop.
    - The active space (ncas, nelecas) must be specified explicitly.
    - Requires Prism to be installed and importable.
'''

import sys
import numpy as np
from pyscf import ao2mo, gto, scf, mcscf
from pyscf import fci as pyscf_fci
from utils import silent_stdout, nullcontext


def _check_prism():
    '''Raise a clear error if Prism is not importable.'''
    try:
        import prism.interface
        import prism.nevpt
    except ImportError as e:
        raise ImportError(
            "Prism is required for QD-NEVPT2 but could not be imported.\n"
            f"Original error: {e}\n"
            "Install Prism from https://github.com/sokolov-group/prism or add it to your PYTHONPATH."
        ) from e


def solve(mf_real, ncas, nelecas,
          sa_nstates=3, sa_weights=None,
          prism_backend='opt_einsum',
          nfrozen=None,
          compute_singles=False,
          s_thresh_singles=1e-8,
          s_thresh_doubles=1e-8,
          select_reference=None,
          casscf_kwargs=None,
          nevpt_kwargs=None,
          mo_guess=None,
          printoutput=True):
    '''
    Run SA-CASSCF + QD-NEVPT2 via Prism on a real PySCF mf object.

    This function is not called directly by the DMET loop — it is invoked
    through dmet.doexact() when method='QD-NEVPT2'. It can also be used as
    a standalone function for testing.

    Parameters
    ----------
    mf_real      : pyscf.scf.hf.RHF
        A converged RHF mean-field object on the REAL physical molecule.
    ncas         : int
        Number of active orbitals for CASSCF.
    nelecas      : int
        Number of active electrons for CASSCF.
    sa_nstates   : int
        Number of states for state-averaging (minimum 2 for QD-NEVPT2).
    sa_weights   : list of float or None
        Weights for state-averaging. Uniform if None.
    prism_backend : str
        Einsum backend for Prism: 'opt_einsum', 'numpy', or 'pytblis'.
    nfrozen      : int or None
        Number of frozen core orbitals for QD-NEVPT2 (not for CASSCF).
    compute_singles : bool
        Whether to include singles amplitudes in the QD-NEVPT2 energy.
    s_thresh_singles : float
        Threshold for singles space linear dependency screening.
    s_thresh_doubles : float
        Threshold for doubles space linear dependency screening.
    select_reference : list of int or None
        If set, select a subset of SA-CASSCF states for NEVPT2. 1-indexed.
        E.g. [1, 3, 5] selects states 1, 3, and 5.
    casscf_kwargs : dict or None
        Extra keyword arguments set on the mc CASSCF object before kernel,
        e.g. {'conv_tol': 1e-11, 'conv_tol_grad': 1e-6}.
    nevpt_kwargs  : dict or None
        Extra keyword arguments set on the Prism NEVPT object before kernel,
        e.g. {'rdm_order': 2}.
    printoutput   : bool
        If False, suppress most output.

    Returns
    -------
    e_tot   : ndarray  – total QD-NEVPT2 energies for each state (Ha)
    e_corr  : ndarray  – QD-NEVPT2 correlation energies
    osc     : object   – oscillator strengths (as returned by Prism)
    mc      : mcscf.CASSCF – the converged SA-CASSCF object (for inspection)
    nevpt   : prism.nevpt.QDNEVPT – the Prism NEVPT object (for RDMs, etc.)
    '''
    _check_prism()
    import prism.interface
    import prism.nevpt

    if sa_nstates < 2:
        raise ValueError(
            "QD-NEVPT2 requires at least 2 states (sa_nstates >= 2). "
            "For single-state NEVPT2 use method='CASSCF' with PySCF's built-in NEVPT2."
        )

    if sa_weights is None:
        sa_weights = [1.0 / sa_nstates] * sa_nstates
    sa_weights = np.array(sa_weights, dtype=float)
    sa_weights /= sa_weights.sum()

    casscf_kwargs = casscf_kwargs or {}
    nevpt_kwargs  = nevpt_kwargs  or {}

    ctx = silent_stdout() if not printoutput else nullcontext()

    with ctx:
        # ------------------------------------------------------------------
        # SA-CASSCF on the real molecule
        # ------------------------------------------------------------------
        mc = mcscf.CASSCF(mf_real, ncas, nelecas)
        mc = mcscf.state_average_(mc, weights=sa_weights.tolist())
        for key, val in casscf_kwargs.items():
            setattr(mc, key, val)

        mc.verbose = 5 if printoutput else 0
        mc.kernel(mo_guess)

        print(f"\nqdnevpt::solve : SA-CASSCF ({sa_nstates} states, ncas={ncas}, nelecas={nelecas})")
        for i, e in enumerate(mc.e_states):
            print(f"  State {i}: {e:.10f} Ha  (weight={sa_weights[i]:.4f})")
        print(f"  SA-weighted e_tot: {mc.e_tot:.10f} Ha")

        # ------------------------------------------------------------------
        # Prism interface + QD-NEVPT2
        # ------------------------------------------------------------------
        interface = prism.interface.PYSCF(
            mf_real, mc,
            backend=prism_backend,
            select_reference=select_reference,
        )

        nevpt_obj = prism.nevpt.QDNEVPT(interface)
        nevpt_obj.compute_singles_amplitudes = compute_singles
        nevpt_obj.s_thresh_singles = s_thresh_singles
        nevpt_obj.s_thresh_doubles = s_thresh_doubles
        if nfrozen is not None:
            nevpt_obj.nfrozen = nfrozen
        for key, val in nevpt_kwargs.items():
            setattr(nevpt_obj, key, val)

        e_tot, e_corr, osc = nevpt_obj.kernel()

        print("\nqdnevpt::solve : QD-NEVPT2 state energies:")
        eV = 27.21138602
        for i, (et, ec) in enumerate(zip(e_tot, e_corr)):
            print(f"  State {i}: E_tot = {et:.10f} Ha,  E_corr = {ec:.10f} Ha")
        print()
        if osc is not None:
            print("  Oscillator strengths:")
            for line in str(osc).splitlines():
                print(f"    {line}")

    return e_tot, e_corr, osc, mc, nevpt_obj



# ---------------------------------------------------------------------------
# SolverFactory entry point
# ---------------------------------------------------------------------------

def execute(task):
    """
    SolverFactory-compatible wrapper for the QD-NEVPT2 solver.

    Supports two transport modes for the physical SCF object (mirroring the
    pattern in ``solvers/nevpt2.py``):

    1. **Serialized (parallel-worker) path** (preferred for ProcessPoolExecutor):
       The task dict carries picklable numpy arrays + a JSON Mole dump:
         task['mol_dumps'], task['mf_mo_coeff'], task['mf_mo_energy'],
         task['mf_mo_occ'], task['mf_e_tot'].
       A dummy PySCF RHF object is reconstructed from these inside the worker
       process before calling solve().

    2. **Live-object (legacy/sequential) path**:
       task['mf_real'] is the live PySCF RHF object from the main process.
       Used when called directly from ``_run_fragment_sequential()`` without
       the parallel-task serialization step.

    Mode 1 is chosen when 'mol_dumps' is present in the task dict.
    Mode 2 is the fallback if 'mol_dumps' is absent.

    Parameters
    ----------
    task : dict
        Mode 1 keys: mol_dumps, mf_mo_coeff, mf_mo_energy, mf_mo_occ,
                     mf_e_tot, ncas, nelecas.
        Mode 2 keys: mf_real, ncas, nelecas.
        Optional (both modes): sa_nstates, sa_weights, casscf_kwargs,
                               qdnevpt2_kwargs, mo_guess.

    Returns
    -------
    (energy, rdm1, qdnevpt2_res) where qdnevpt2_res is a dict with keys
    'e_tot', 'e_corr', 'osc', 'mc', 'nevpt'.
    """
    # ------------------------------------------------------------------
    # Resolve mf_real via either the serialized or live-object protocol.
    # Reuse the helper from nevpt2 — same serialization scheme.
    # ------------------------------------------------------------------
    if 'mol_dumps' in task:
        # Mode 1: reconstruct from serialized arrays (parallel-safe).
        from solvers.nevpt2 import _reconstruct_mf_from_task
        mf_real = _reconstruct_mf_from_task(task)
    else:
        # Mode 2: live object passed directly (sequential / legacy path).
        mf_real = task['mf_real']

    e_tot, e_corr, osc, mc, nevpt_obj = solve(
        mf_real,
        ncas=task.get('ncas'),
        nelecas=task.get('nelecas'),
        sa_nstates=task.get('sa_nstates', 3),
        sa_weights=task.get('sa_weights'),
        casscf_kwargs=task.get('casscf_kwargs', {}),
        nevpt_kwargs=task.get('qdnevpt2_kwargs', {}),
        mo_guess=task.get('mo_guess'),
    )

    rdm1 = mc.make_rdm1()
    qdnevpt2_res = {
        'e_tot' : e_tot,
        'e_corr': e_corr,
        'osc'   : osc,
        'mc'    : mc,
        'nevpt' : nevpt_obj,
    }
    return e_tot[0], rdm1, qdnevpt2_res
