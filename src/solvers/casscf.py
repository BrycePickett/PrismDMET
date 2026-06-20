'''CASSCF solver for DMET embedding clusters using PySCF's mcscf module.'''

import numpy as np
from pyscf import ao2mo, gto, scf, mcscf
from pyscf import fci as pyscf_fci
from ..utils import silent_stdout, nullcontext


def solve(const, oei, fock, tei, norb, nel, nimp, dm_guess_rhf,
          ncas=None, nelecas=None,
          chempot_imp=0.0, printoutput=True,
          sa_nstates=1, sa_weights=None,
          frozen=None,
          mo_guess=None, ci_guess=None,
          oei_s=None, spin=None,
          cas_select='energy', ao_labels=None, ao2eo=None, ao_mol_dumps=None,
          avas_threshold=0.2, embed_level_shift=0.0, cas_multiseed=False,
          **casscf_kwargs):
    '''Solve a DMET impurity problem at the CASSCF level. Returns (impurity_energy, rdm1, cas_results).'''
    if ncas is None:
        ncas = norb
    if nelecas is None:
        nelecas = nel

    _spin = spin if spin is not None else (nel % 2)
    _use_rohf = (_spin != 0)

    if _use_rohf:
        assert ncas <= norb, f"casscf::solve: ncas ({ncas}) cannot exceed norb ({norb})"
        assert nelecas <= nel, f"casscf::solve: nelecas ({nelecas}) cannot exceed nel ({nel})"
    else:
        assert nel % 2 == 0, "casscf::solve: nel must be even (RHF reference required)"
        assert ncas <= norb, f"casscf::solve: ncas ({ncas}) cannot exceed norb ({norb})"
        assert nelecas <= nel, f"casscf::solve: nelecas ({nelecas}) cannot exceed nel ({nel})"
        assert (nel - nelecas) % 2 == 0, \
            "casscf::solve: (nel - nelecas) must be even (frozen core must be closed-shell)"

    if sa_nstates > 1:
        if sa_weights is None:
            sa_weights = [1.0 / sa_nstates] * sa_nstates
        assert len(sa_weights) == sa_nstates, \
            "casscf::solve: len(sa_weights) must equal sa_nstates"
        sa_weights = np.array(sa_weights, dtype=float)
        sa_weights /= sa_weights.sum()   # normalize
    else:
        sa_weights = np.array([1.0])

    ctx = silent_stdout() if not printoutput else nullcontext()

    fock_copy = fock.copy()
    if chempot_imp != 0.0:
        for orb in range(nimp):
            fock_copy[orb, orb] -= chempot_imp

    with ctx:
        mol = gto.Mole()
        mol.build(verbose=0)
        mol.atom.append(('C', (0, 0, 0)))
        mol.nelectron = nel
        mol.spin = _spin
        mol.incore_anyway = True

        if _use_rohf:
            mf = scf.ROHF(mol)
            print(f"casscf::solve : Using ROHF reference (spin={_spin}, nel={nel})")
        else:
            mf = scf.RHF(mol)

        mf.get_hcore = lambda *args: fock_copy
        mf.get_ovlp  = lambda *args: np.eye(norb)
        mf._eri      = ao2mo.restore(8, tei, norb)
        # Level shift opens the near-degenerate trap gap for a deterministic embedded SCF.
        mf.level_shift = embed_level_shift
        mf.scf(dm_guess_rhf)
        dm_loc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
        if not mf.converged:
            mf.max_cycle = 300
            mf.diis_space = 12
            mf.scf(dm_loc)
            dm_loc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
        if embed_level_shift != 0.0:
            # Confirm the shifted fixed point is also a stationary point of the real
            # (unshifted) Hamiltonian: reconverge with the shift removed and use that as
            # the actual reference; if it moves, the shift masked rather than fixed the
            # instability.
            e_shifted = mf.e_tot
            mf.level_shift = 0.0
            mf.scf(dm_loc)
            dm_loc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
            print(f"casscf::solve : level-shift verification: E(shift={embed_level_shift})="
                  f"{e_shifted:.10f}  E(shift removed, reconverged)={mf.e_tot:.10f}  "
                  f"dE={abs(mf.e_tot - e_shifted):.2e} Ha")

        numPairs = nel // 2
        fock_loc = (fock_copy
                   + np.einsum('ijkl,ij->kl', tei, dm_loc)
                   - 0.5 * np.einsum('ijkl,ik->jl', tei, dm_loc))
        eigvals = np.linalg.eigvalsh(fock_loc)
        eigvals.sort()
        print("casscf::solve : RHF homo-lumo gap =", eigvals[numPairs] - eigvals[numPairs - 1])

        # Skip selection when a warm-restart MO guess is supplied: mc.kernel(mo_guess)
        # would overwrite the reordered mo_coeff anyway.
        selected_orbs = None
        mo_natorb = None
        if cas_select != 'energy' and mo_guess is None:
            from .qcsolver_utils import plan_cas_active_space
            selected_orbs, ncas, nelecas, mo_natorb = plan_cas_active_space(
                mf, cas_select, ncas, nelecas, nimp, norb,
                ao2eo=ao2eo, ao_mol_dumps=ao_mol_dumps, ao_labels=ao_labels,
                sa_nstates=sa_nstates, avas_threshold=avas_threshold)

        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.verbose = 5 if printoutput else 0
        if frozen is not None:
            mc.frozen = frozen
        for key, val in casscf_kwargs.items():
            setattr(mc, key, val)

        # For ROHF+oei_s, swap the base FCI solver before state_average_ so the
        # SA wrapper inherits direct_uhf.FCI instead of direct_spin1.FCI.
        # direct_spin1.FCI cannot accept the [h1e_a, h1e_b] list that
        # fix_casscf_for_nonsinglet_env injects; direct_uhf.FCI handles it.
        if _use_rohf and oei_s is not None and not np.all(np.abs(oei_s) < 1e-8):
            _uhf_fci = pyscf_fci.direct_uhf.FCI()
            _uhf_fci.verbose = mc.fcisolver.verbose
            mc.fcisolver = _uhf_fci

        if sa_nstates > 1:
            mc = mcscf.state_average_(mc, weights=sa_weights.tolist())

        if oei_s is not None:
            from .qcsolver_utils import fix_casscf_for_nonsinglet_env
            mc = fix_casscf_for_nonsinglet_env(mc, oei_s)

        if mo_natorb is not None:
            mc.mo_coeff = mo_natorb
            if printoutput:
                print(f"casscf::solve : CAS selection by {cas_select}, CAS({nelecas},{ncas})")
        elif selected_orbs is not None:
            mc.mo_coeff = mc.sort_mo(selected_orbs, base=0)
            if printoutput:
                print(f"casscf::solve : CAS selection by {cas_select}, "
                      f"selected {ncas} orbitals: {selected_orbs}")

        _mo0 = mo_guess if mo_guess is not None else None
        _ci0 = ci_guess if ci_guess is not None else None
        if cas_multiseed and _mo0 is None:
            from .qcsolver_utils import multiseed_casscf
            multiseed_casscf(mc, mc.mo_coeff)
        else:
            mc.kernel(_mo0, _ci0)

        ncore = mc.ncore
        if _use_rohf:
            _nelecas_fci = ((nelecas + _spin) // 2, (nelecas - _spin) // 2)
            _used_uhf_fci = oei_s is not None and not np.all(np.abs(oei_s) < 1e-8)
            ci_solver_base = (pyscf_fci.direct_uhf.FCI() if _used_uhf_fci
                              else pyscf_fci.direct_spin1.FCI())
        else:
            _nelecas_fci = nelecas
            ci_solver_base = pyscf_fci.direct_spin0.FCI()

        if sa_nstates > 1:
            # SA-CASSCF: per-state CAS RDMs, then take weighted average
            ci_vecs = mc.ci   # list of CI vectors, one per state
            rdm1_cas = np.zeros((ncas, ncas))
            rdm2_cas = np.zeros((ncas, ncas, ncas, ncas))
            for w, ci_vec in zip(sa_weights, ci_vecs):
                r1, r2 = ci_solver_base.make_rdm12(ci_vec, ncas, _nelecas_fci)
                rdm1_cas += w * r1
                rdm2_cas += w * r2
            e_states = np.array(mc.e_states)
            e_tot    = mc.e_tot   # weighted average
            print(f"\ncasscf::solve : SA-CASSCF state energies:")
            for i, e in enumerate(e_states):
                print(f"  State {i}: {e:.10f} Ha  (weight={sa_weights[i]:.4f})")
            print(f"  Weighted average: {e_tot:.10f} Ha")
        else:
            rdm1_cas, rdm2_cas = ci_solver_base.make_rdm12(mc.ci, ncas, _nelecas_fci)
            e_states = None
            e_tot    = mc.e_tot
            print(f"\ncasscf::solve : CASSCF energy = {e_tot:.10f} Ha")
            print(f"  ncore={ncore}, ncas={ncas}, nelecas={nelecas}")

        nmo = mf.mo_coeff.shape[1]
        dm1_mo = np.zeros((nmo, nmo))
        dm2_mo = np.zeros((nmo, nmo, nmo, nmo))

        for i in range(ncore):
            dm1_mo[i, i] = 2.0
            for j in range(ncore):
                dm2_mo[i, i, j, j] += 4.0
                dm2_mo[i, j, j, i] -= 2.0
            for p in range(ncas):
                for q in range(ncas):
                    dm2_mo[i, i, ncore+p, ncore+q] += 2.0 * rdm1_cas[p, q]
                    dm2_mo[ncore+p, ncore+q, i, i] += 2.0 * rdm1_cas[p, q]
                    dm2_mo[i, ncore+q, ncore+p, i] -= rdm1_cas[p, q]
                    dm2_mo[ncore+p, i, i, ncore+q] -= rdm1_cas[p, q]

        # Active (CAS) orbitals
        dm1_mo[ncore:ncore+ncas, ncore:ncore+ncas] = rdm1_cas
        dm2_mo[ncore:ncore+ncas, ncore:ncore+ncas,
               ncore:ncore+ncas, ncore:ncore+ncas] = rdm2_cas

        print(f"casscf::solve : Full-space 1-RDM trace = {np.trace(dm1_mo):.6f}")

        # Rotate from MO basis to local dmet orbital basis
        C = mc.mo_coeff
        pyscf_rdm1 = C @ dm1_mo @ C.T
        pyscf_rdm2 = np.einsum('ai,ijkl->ajkl', C, dm2_mo)
        pyscf_rdm2 = np.einsum('bj,ajkl->abkl', C, pyscf_rdm2)
        pyscf_rdm2 = np.einsum('ck,abkl->abcl', C, pyscf_rdm2)
        pyscf_rdm2 = np.einsum('dl,abcl->abcd', C, pyscf_rdm2)

        impurity_energy = (
            const
            + 0.25  * np.einsum('ij,ij->', pyscf_rdm1[:nimp,:],     fock[:nimp,:] + oei[:nimp,:])
            + 0.25  * np.einsum('ij,ij->', pyscf_rdm1[:,:nimp],     fock[:,:nimp] + oei[:,:nimp])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:nimp,:,:,:], tei[:nimp,:,:,:])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:,:nimp,:,:], tei[:,:nimp,:,:])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:,:,:nimp,:], tei[:,:,:nimp,:])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:,:,:,:nimp], tei[:,:,:,:nimp])
        )

    cas_results = {
        'e_tot'    : e_tot,
        'e_states' : e_states,
        'e_imp'    : impurity_energy,
        'ncas'     : ncas,
        'nelecas'  : nelecas,
        'nstates'  : sa_nstates,
        'weights'  : sa_weights,
        'ci'       : mc.ci,
        'mo_coeff' : mc.mo_coeff,
        'cas_select'    : cas_select,
        'selected_orbs' : selected_orbs,
    }

    return impurity_energy, pyscf_rdm1, cas_results


def execute(task):
    """SolverDispatcher entry point for the CASSCF solver."""
    return solve(
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
        chempot_imp=task.get('chempot_imp', 0.0),
        sa_nstates=task.get('sa_nstates', 1),
        sa_weights=task.get('sa_weights'),
        mo_guess=task.get('mo_guess'),
        ci_guess=task.get('ci_guess'),
        oei_s=task.get('oei_s'),
        spin=task.get('spin'),
        cas_select=task.get('cas_select', 'energy'),
        ao_labels=task.get('ao_labels'),
        ao2eo=task.get('ao2eo'),
        ao_mol_dumps=task.get('ao_mol_dumps'),
        avas_threshold=task.get('avas_threshold', 0.2),
        embed_level_shift=task.get('embed_level_shift', 0.0),
        cas_multiseed=task.get('cas_multiseed', False),
        **task.get('casscf_kwargs', {}),
    )
