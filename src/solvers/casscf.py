'''
CASSCF solver for QC-dmet.

Solves the dmet impurity problem at the CASSCF level using PySCF's mcscf module.
Supports single-state and state-averaged CASSCF, warm restart via mo_guess/ci_guess,
and open-shell environments via oei_s spin-potential injection.
'''

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
          **casscf_kwargs):
    '''
    Solve a dmet impurity problem at the CASSCF level.

    Parameters
    ----------
    const        : float
    oei          : ndarray (norb, norb)
    fock         : ndarray (norb, norb)
    tei          : ndarray (norb, norb, norb, norb)
    norb         : int – total number of embedding orbitals (impurity + bath)
    nel          : int – total number of electrons (must be even for RHF ref.)
    nimp         : int – number of impurity orbitals (first nimp of norb)
    dm_guess_rhf   : ndarray – RHF density matrix initial guess
    ncas         : int – number of active orbitals.  Defaults to norb (full space).
    nelecas      : int – number of active electrons.  Defaults to nel (full space).
    chempot_imp  : float – chemical potential on impurity block
    printoutput  : bool
    sa_nstates   : int – number of states for state-averaged CASSCF (default 1 = SS)
    sa_weights   : list of float – weights for SA; uniform if None
    frozen       : list or None – orbital indices to freeze (passed to CASSCF)
    mo_guess     : ndarray or None – MO coefficients from a previous iteration
                   (warm restart).  If provided, these are used instead of the
                   RHF canonical MOs as the initial CASSCF orbital guess.
    ci_guess     : ndarray or None – CI vector(s) from a previous iteration.
    oei_s        : ndarray (norb, norb) or None – spin-dependent one-electron
                   potential for open-shell environments.  When provided, the
                   CASSCF object is wrapped to inject spin-gradient corrections.
    **casscf_kwargs : extra keyword arguments set on the mcscf.CASSCF object
                     (e.g. max_cycle=200, conv_tol=1e-9, fcisolver=...)

    Returns
    -------
    impurity_energy : float
    pyscf_rdm1      : ndarray – 1-RDM in local orbital basis (for dmet loop)
    cas_results    : dict – {
        'e_tot'     : float or ndarray   (SS: total energy; SA: weighted avg)
        'e_states'  : ndarray or None    (SA only: individual state energies)
        'ncas'      : int
        'nelecas'   : int
        'nstates'   : int
        'weights'   : ndarray
        'ci'        : CASSCF CI vector(s)
        'mo_coeff'  : ndarray – converged MO coefficients (for caching)
    }
    '''
    # Default: full active space (equivalent to FCI within the embedding space)
    if ncas is None:
        ncas = norb
    if nelecas is None:
        nelecas = nel

    # Auto-detect spin: if nel is odd (or caller passes spin > 0) use ROHF reference.
    # This guarantees a qualitatively correct orbital guess for open-shell CASSCF.
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
        sa_weights /= sa_weights.sum()   # normalise
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
        mf.scf(dm_guess_rhf)
        dm_loc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
        if not mf.converged:
            mf = mf.newton()
            mf.scf(dm_loc)
            dm_loc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)

        numPairs = nel // 2
        fock_loc = (fock_copy
                   + np.einsum('ijkl,ij->kl', tei, dm_loc)
                   - 0.5 * np.einsum('ijkl,ik->jl', tei, dm_loc))
        eigvals = np.linalg.eigvalsh(fock_loc)
        eigvals.sort()
        print("casscf::solve : RHF homo-lumo gap =", eigvals[numPairs] - eigvals[numPairs - 1])

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

        _mo0 = mo_guess if mo_guess is not None else None
        _ci0 = ci_guess if ci_guess is not None else None
        mc.kernel(_mo0, _ci0)

        ncore = mc.ncore
        if _use_rohf:
            _nelecas_fci = ((nelecas + 1) // 2, nelecas // 2)
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

        # -----------
        # dmet impurity energy (half-projector formula)
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
    }

    return impurity_energy, pyscf_rdm1, cas_results


# ---------------------------------------------------------------------------
# SolverDispatcher entry point
# ---------------------------------------------------------------------------

def execute(task):
    """
    SolverDispatcher-compatible wrapper for the CASSCF solver.

    Unpacks the standardised task dict and calls solve(), which returns
    a 3-tuple: (impurity_energy, pyscf_rdm1, cas_results).

    The warm-restart cache (mo_guess, ci_guess) and the open-shell spin
    potential (oei_s) are passed through transparently via the task dict.

    Parameters
    ----------
    task : dict
        Must contain: const, dmet_oei, dmet_fock, dmet_tei, norb, nel, nimp,
        dm_guess_rhf, chempot_imp.
        Optional: ncas, nelecas, sa_nstates, sa_weights, casscf_kwargs,
        mo_guess, ci_guess, oei_s.

    Returns
    -------
    (impurity_energy, pyscf_rdm1, cas_results) — same as solve().
    """
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
        **task.get('casscf_kwargs', {}),
    )
