'''SA-CASSCF + QD-NEVPT2 solver for DMET embedding clusters via Prism; one-shot only, sa_nstates >= 2.'''

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
          cas_select='energy', ao_labels=None, ao2eo=None, ao_mol_dumps=None,
          printoutput=True):
    '''Run SA-CASSCF + QD-NEVPT2 via Prism on the DMET embedding cluster. Returns (e_tot, e_corr, None, mc, nevpt_obj).'''
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
            mf.max_cycle = 300
            mf.diis_space = 12
            mf.scf(mf.make_rdm1())
        if _use_rohf:
            print(f"qdnevpt2::solve : embedded ROHF (spin={_spin}, nel={nel}, norb={norb})")

        mc = mcscf.CASSCF(mf, ncas, nelecas)
        # Must swap FCI solver before state_average_; direct_uhf.FCI required for spin-asymmetric [h1e_a, h1e_b].
        if _use_rohf and oei_s is not None and not np.all(np.abs(oei_s) < 1e-8):
            _uhf_fci = pyscf_fci.direct_uhf.FCI()
            _uhf_fci.verbose = mc.fcisolver.verbose
            mc.fcisolver = _uhf_fci
        mc = mcscf.state_average_(mc, weights=sa_weights.tolist())
        mc.verbose = 5 if printoutput else 0
        for key, val in casscf_kwargs.items():
            setattr(mc, key, val)

        if oei_s is not None:
            from .qcsolver_utils import fix_casscf_for_nonsinglet_env
            mc = fix_casscf_for_nonsinglet_env(mc, oei_s)

        if cas_select != 'energy':
            from .qcsolver_utils import select_cas_orbitals
            selected_orbs = select_cas_orbitals(
                mc, cas_select, ncas, nimp, norb,
                ao2eo=ao2eo, ao_mol_dumps=ao_mol_dumps, ao_labels=ao_labels)
            if printoutput:
                print(f"qdnevpt2::solve : CAS selection by {cas_select}, "
                      f"selected {ncas} orbitals: {selected_orbs}")

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
        # Dummy mol has no real AO basis; stub out osc_strengths so print_results does not crash.
        _n = sa_nstates
        def _skip_osc():
            nevpt_obj.properties["osc_strengths"] = np.zeros(_n - 1) if _n > 1 else None
        nevpt_obj.compute_properties = _skip_osc
        if nfrozen is not None:
            nevpt_obj.nfrozen = nfrozen
        for key, val in nevpt_kwargs.items():
            setattr(nevpt_obj, key, val)

        e_tot, e_corr, _ = nevpt_obj.kernel()

        print("\nqdnevpt2::solve : QD-NEVPT2 state energies (excitation = state - state 0):")
        for i, (et, ec) in enumerate(zip(e_tot, e_corr)):
            de_ev = (et - e_tot[0]) * 27.21138602
            print(f"  State {i}: E_tot = {et:.10f} Ha  ΔE = {de_ev:+.4f} eV")

    return e_tot, e_corr, None, mc, nevpt_obj


def execute(task):
    """SolverDispatcher entry point for the embedded QD-NEVPT2 solver."""
    # Extract named solver params from qdnevpt2_kwargs; remainder forwarded to Prism NEVPT object.
    _kw = dict(task.get('qdnevpt2_kwargs', {}))
    e_tot, e_corr, _, mc, nevpt_obj = solve(
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
        prism_backend=_kw.pop('prism_backend', 'opt_einsum'),
        nfrozen=_kw.pop('nfrozen', None),
        compute_singles=_kw.pop('compute_singles', False),
        s_thresh_singles=_kw.pop('s_thresh_singles', 1e-8),
        s_thresh_doubles=_kw.pop('s_thresh_doubles', 1e-8),
        select_reference=_kw.pop('select_reference', None),
        casscf_kwargs=task.get('casscf_kwargs', {}),
        nevpt_kwargs=_kw,
        spin=task.get('spin'),
        oei_s=task.get('oei_s'),
        cas_select=task.get('cas_select', 'energy'),
        ao_labels=task.get('ao_labels'),
        ao2eo=task.get('ao2eo'),
        ao_mol_dumps=task.get('ao_mol_dumps'),
    )

    rdm1 = mc.make_rdm1()
    qdnevpt2_res = {
        'e_tot' : e_tot,
        'e_corr': e_corr,
        'mc'    : mc,
        'nevpt' : nevpt_obj,
    }
    return e_tot[0], rdm1, qdnevpt2_res
