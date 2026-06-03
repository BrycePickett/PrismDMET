'''
QD-NEVPT2 solver for QC-dmet using Prism (https://github.com/sokolov-group/prism).

Runs SA-CASSCF + QD-NEVPT2 on the DMET embedding cluster (impurity + bath),
mirroring the embedded CASSCF and NEVPT2 solvers. The cluster is built from
the embedding integrals exactly as in the other solvers; Prism's PYSCF
interface receives the embedded mf and mc objects.

Note on total energies: the embedded dummy mol has one dummy atom with
energy_nuc() = 0. Total QD-NEVPT2 energies therefore lack the nuclear
repulsion constant. This constant is state-independent and cancels in all
excitation energies, which are the primary deliverable.

Note on oscillator strengths: dipole integrals require the real AO basis, which
the dummy mol does not have. Oscillator strength calculation is disabled by
default (compute_dipole=False).

Only compatible with one-shot DMET (sc_method='NONE'). sa_nstates >= 2 required.
'''

import numpy as np
from pyscf import ao2mo, gto, scf, mcscf
from pyscf import fci as pyscf_fci
from ..utils import silent_stdout, nullcontext


def _check_prism():
    try:
        import prism.interface
        import prism.nevpt
    except ImportError as e:
        raise ImportError(
            "Prism is required for QD-NEVPT2 but could not be imported.\n"
            f"Original error: {e}\n"
            "Install Prism from https://github.com/sokolov-group/prism or add it to PYTHONPATH."
        ) from e


def solve(const, oei, fock, tei, norb, nel, nimp, dm_guess_rhf,
          ncas, nelecas,
          sa_nstates=3, sa_weights=None,
          chempot_imp=0.0,
          prism_backend='opt_einsum',
          nfrozen=None,
          compute_singles=False,
          s_thresh_singles=1e-8,
          s_thresh_doubles=1e-8,
          select_reference=None,
          casscf_kwargs=None,
          nevpt_kwargs=None,
          spin=None,
          oei_s=None,
          printoutput=True):
    '''
    Run SA-CASSCF + QD-NEVPT2 via Prism on the DMET embedding cluster.

    Parameters
    ----------
    const, oei, fock, tei, norb, nel, nimp, dm_guess_rhf
        Embedding integrals in the cluster (impurity + bath) basis, as
        provided by the DMET driver.
    ncas, nelecas
        Active space size.
    sa_nstates
        States for state-averaging (>= 2 required for QD-NEVPT2).
    sa_weights
        SA weights; uniform if None.
    chempot_imp
        Chemical potential on impurity diagonal.
    prism_backend
        Einsum backend for Prism.
    nfrozen
        Frozen core for Prism (cluster core, not physical frozen core).
    compute_singles
        Include singles amplitudes in QD-NEVPT2.
    s_thresh_singles, s_thresh_doubles
        Linear-dependency thresholds.
    select_reference
        Subset of SA states (1-indexed) for QD-NEVPT2.
    casscf_kwargs
        Extra attributes set on the mc CASSCF object.
    nevpt_kwargs
        Extra attributes set on the Prism NEVPT object.
    spin
        Cluster spin (2S); defaults to nel % 2.
    oei_s
        Spin-asymmetry 1e correction 0.5*(F_alpha - F_beta) in the cluster
        basis. Applied via fix_casscf_for_nonsinglet_env to the CASSCF. See
        the spin plan (docs/prismdmet_spin_plan.md) for full context.
    printoutput
        Suppress most output if False.

    Returns
    -------
    e_tot   : ndarray – QD-NEVPT2 energies per state (Ha, no nuclear repulsion)
    e_corr  : ndarray – QD-NEVPT2 correlation energies
    osc     : object  – oscillator strengths from Prism (None if disabled)
    mc      : mcscf.CASSCF – converged SA-CASSCF object
    nevpt   : Prism NEVPT object
    '''
    _check_prism()
    import prism.interface
    import prism.nevpt

    if sa_nstates < 2:
        raise ValueError(
            "QD-NEVPT2 requires sa_nstates >= 2. "
            "For single-state NEVPT2 use method='NEVPT2'."
        )

    if sa_weights is None:
        sa_weights = [1.0 / sa_nstates] * sa_nstates
    sa_weights = np.array(sa_weights, dtype=float)
    sa_weights /= sa_weights.sum()

    casscf_kwargs = casscf_kwargs or {}
    nevpt_kwargs  = nevpt_kwargs  or {}

    _spin    = spin if spin is not None else (nel % 2)
    _use_rohf = (_spin != 0)

    fock_copy = fock.copy()
    if chempot_imp != 0.0:
        for orb in range(nimp):
            fock_copy[orb, orb] -= chempot_imp

    ctx = silent_stdout() if not printoutput else nullcontext()

    with ctx:
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
            print(f"qdnevpt2::solve : embedded ROHF (spin={_spin}, nel={nel}, norb={norb})")

        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc = mcscf.state_average_(mc, weights=sa_weights.tolist())
        mc.verbose = 5 if printoutput else 0
        for key, val in casscf_kwargs.items():
            setattr(mc, key, val)

        if oei_s is not None:
            from .qcsolver_utils import fix_casscf_for_nonsinglet_env
            if _use_rohf and not np.all(np.abs(oei_s) < 1e-8):
                _uhf_fci = pyscf_fci.direct_uhf.FCI()
                _uhf_fci.verbose = mc.fcisolver.verbose
                mc.fcisolver = _uhf_fci
            mc = fix_casscf_for_nonsinglet_env(mc, oei_s)

        mc.kernel()

        print(f"\nqdnevpt2::solve : embedded SA-CASSCF ({sa_nstates} states, "
              f"ncas={ncas}, nelecas={nelecas})")
        for i, e in enumerate(mc.e_states):
            print(f"  State {i}: {e:.10f} Ha  (weight={sa_weights[i]:.4f})")
        print(f"  SA-weighted e_tot: {mc.e_tot:.10f} Ha")

        interface = prism.interface.PYSCF(
            mf, mc,
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

        print("\nqdnevpt2::solve : QD-NEVPT2 state energies (excitation = state - state 0):")
        for i, (et, ec) in enumerate(zip(e_tot, e_corr)):
            de_ev = (et - e_tot[0]) * 27.21138602
            print(f"  State {i}: E_tot = {et:.10f} Ha  ΔE = {de_ev:+.4f} eV")

    return e_tot, e_corr, osc, mc, nevpt_obj


# ---------------------------------------------------------------------------
# SolverDispatcher entry point
# ---------------------------------------------------------------------------

def execute(task):
    """
    SolverDispatcher entry point for the embedded QD-NEVPT2 solver.

    Operates on the embedding-cluster integrals carried in the task dict
    (dmet_oei / dmet_fock / dmet_tei / norb / nel / nimp), so no physical
    molecule is needed.
    """
    e_tot, e_corr, osc, mc, nevpt_obj = solve(
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
        sa_nstates=task.get('sa_nstates', 3),
        sa_weights=task.get('sa_weights'),
        chempot_imp=task.get('chempot_imp', 0.0),
        casscf_kwargs=task.get('casscf_kwargs', {}),
        nevpt_kwargs=task.get('qdnevpt2_kwargs', {}),
        spin=task.get('spin'),
        oei_s=task.get('oei_s'),
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
