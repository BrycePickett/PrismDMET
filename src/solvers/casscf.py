'''
    CASSCF solver for QC-DMET.

    Solves the DMET impurity problem at the CASSCF level using PySCF's mcscf
    module. The impurity active space is defined by (ncas, nelecas); the
    remaining electrons occupy frozen core orbitals.

    Supports:
        - Single-state CASSCF (default)
        - State-averaged CASSCF (sa_nstates > 1) for multi-state problems
          and as the reference for NEVPT2
        - Warm restart via mo_guess / ci_guess (for iteration caching)
        - Open-shell environments via OEI_S spin-potential injection

    Notes
    -----
    Unlike EOM-CCSD, CASSCF is compatible with both oneshot() and the
    self-consistent DMET loop because it produces a well-defined ground-state
    (or state-averaged) 1-RDM that can be used for u-matrix fitting.

    For state-averaged runs the DMET u-matrix fitting uses the SA-weighted
    1-RDM. Individual state energies are stored in the returned eom_results-
    style dict for analysis, but are not used by the embedding loop.
'''

import numpy as np
from pyscf import ao2mo, gto, scf, mcscf
from pyscf import fci as pyscf_fci
from utils import silent_stdout, nullcontext


def solve(CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, DMguessRHF,
          ncas=None, nelecas=None,
          chempot_imp=0.0, printoutput=True,
          sa_nstates=1, sa_weights=None,
          frozen=None,
          mo_guess=None, ci_guess=None,
          OEI_S=None,
          **casscf_kwargs):
    '''
    Solve a DMET impurity problem at the CASSCF level.

    Parameters
    ----------
    CONST        : float
    OEI          : ndarray (Norb, Norb)
    FOCK         : ndarray (Norb, Norb)
    TEI          : ndarray (Norb, Norb, Norb, Norb)
    Norb         : int – total number of embedding orbitals (impurity + bath)
    Nel          : int – total number of electrons (must be even for RHF ref.)
    Nimp         : int – number of impurity orbitals (first Nimp of Norb)
    DMguessRHF   : ndarray – RHF density matrix initial guess
    ncas         : int – number of active orbitals.  Defaults to Norb (full space).
    nelecas      : int – number of active electrons.  Defaults to Nel (full space).
    chempot_imp  : float – chemical potential on impurity block
    printoutput  : bool
    sa_nstates   : int – number of states for state-averaged CASSCF (default 1 = SS)
    sa_weights   : list of float – weights for SA; uniform if None
    frozen       : list or None – orbital indices to freeze (passed to CASSCF)
    mo_guess     : ndarray or None – MO coefficients from a previous iteration
                   (warm restart).  If provided, these are used instead of the
                   RHF canonical MOs as the initial CASSCF orbital guess.
    ci_guess     : ndarray or None – CI vector(s) from a previous iteration.
    OEI_S        : ndarray (Norb, Norb) or None – spin-dependent one-electron
                   potential for open-shell environments.  When provided, the
                   CASSCF object is wrapped to inject spin-gradient corrections.
    **casscf_kwargs : extra keyword arguments set on the mcscf.CASSCF object
                     (e.g. max_cycle=200, conv_tol=1e-9, fcisolver=...)

    Returns
    -------
    ImpurityEnergy : float
    pyscfRDM1      : ndarray – 1-RDM in local orbital basis (for DMET loop)
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
        ncas = Norb
    if nelecas is None:
        nelecas = Nel

    assert Nel % 2 == 0, "casscf::solve: Nel must be even (RHF reference required)"
    assert ncas <= Norb, f"casscf::solve: ncas ({ncas}) cannot exceed Norb ({Norb})"
    assert nelecas <= Nel, f"casscf::solve: nelecas ({nelecas}) cannot exceed Nel ({Nel})"
    assert (Nel - nelecas) % 2 == 0, \
        "casscf::solve: (Nel - nelecas) must be even (frozen core must be closed-shell)"

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

    # ------------------------------------------------------------------
    # Chemical potential shift on the impurity block of FOCK
    # ------------------------------------------------------------------
    FOCKcopy = FOCK.copy()
    if chempot_imp != 0.0:
        for orb in range(Nimp):
            FOCKcopy[orb, orb] -= chempot_imp

    with ctx:
        # --------------------------------------------------------------
        # Build dummy PySCF Mole and run RHF (same pattern as cc.py)
        # --------------------------------------------------------------
        mol = gto.Mole()
        mol.build(verbose=0)
        mol.atom.append(('C', (0, 0, 0)))
        mol.nelectron = Nel
        mol.incore_anyway = True
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: FOCKcopy
        mf.get_ovlp  = lambda *args: np.eye(Norb)
        mf._eri      = ao2mo.restore(8, TEI, Norb)
        mf.scf(DMguessRHF)
        DMloc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
        if not mf.converged:
            mf = mf.newton()
            mf.scf(DMloc)
            DMloc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)

        numPairs = Nel // 2
        FOCKloc = (FOCKcopy
                   + np.einsum('ijkl,ij->kl', TEI, DMloc)
                   - 0.5 * np.einsum('ijkl,ik->jl', TEI, DMloc))
        eigvals = np.linalg.eigvalsh(FOCKloc)
        eigvals.sort()
        print("casscf::solve : RHF homo-lumo gap =", eigvals[numPairs] - eigvals[numPairs - 1])

        # --------------------------------------------------------------
        # Build CASSCF object
        # --------------------------------------------------------------
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.verbose = 5 if printoutput else 0
        if frozen is not None:
            mc.frozen = frozen
        for key, val in casscf_kwargs.items():
            setattr(mc, key, val)

        # --------------------------------------------------------------
        # State-averaged or single-state
        # --------------------------------------------------------------
        if sa_nstates > 1:
            mc = mcscf.state_average_(mc, weights=sa_weights.tolist())

        # --------------------------------------------------------------
        # Open-shell spin-environment wrapper (must come AFTER state_average_)
        # --------------------------------------------------------------
        if OEI_S is not None:
            from solvers.qcsolver_utils import fix_casscf_for_nonsinglet_env
            mc = fix_casscf_for_nonsinglet_env(mc, OEI_S)

        # --------------------------------------------------------------
        # Warm restart: use cached MOs / CI from previous iteration
        # --------------------------------------------------------------
        _mo0 = mo_guess if mo_guess is not None else None
        _ci0 = ci_guess if ci_guess is not None else None
        mc.kernel(_mo0, _ci0)

        # --------------------------------------------------------------
        # Extract 1-RDM and 2-RDM in the CAS space, then expand to
        # full embedding space and rotate to local orbital basis
        # --------------------------------------------------------------
        ncore = mc.ncore
        ci_solver_base = pyscf_fci.direct_spin0.FCI()

        if sa_nstates > 1:
            # SA-CASSCF: per-state CAS RDMs, then take weighted average
            ci_vecs = mc.ci   # list of CI vectors, one per state
            rdm1_cas = np.zeros((ncas, ncas))
            rdm2_cas = np.zeros((ncas, ncas, ncas, ncas))
            for w, ci_vec in zip(sa_weights, ci_vecs):
                r1, r2 = ci_solver_base.make_rdm12(ci_vec, ncas, nelecas)
                rdm1_cas += w * r1
                rdm2_cas += w * r2
            e_states = np.array(mc.e_states)
            e_tot    = mc.e_tot   # weighted average
            print(f"\ncasscf::solve : SA-CASSCF state energies:")
            for i, e in enumerate(e_states):
                print(f"  State {i}: {e:.10f} Ha  (weight={sa_weights[i]:.4f})")
            print(f"  Weighted average: {e_tot:.10f} Ha")
        else:
            rdm1_cas, rdm2_cas = ci_solver_base.make_rdm12(mc.ci, ncas, nelecas)
            e_states = None
            e_tot    = mc.e_tot
            print(f"\ncasscf::solve : CASSCF energy = {e_tot:.10f} Ha")
            print(f"  ncore={ncore}, ncas={ncas}, nelecas={nelecas}")

        # Build full MO-space 1-RDM and 2-RDM
        nmo = mf.mo_coeff.shape[1]
        dm1_mo = np.zeros((nmo, nmo))
        dm2_mo = np.zeros((nmo, nmo, nmo, nmo))

        # Core orbitals: doubly occupied
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

        # Rotate from MO basis to local DMET orbital basis
        C = mc.mo_coeff
        pyscfRDM1 = C @ dm1_mo @ C.T
        pyscfRDM2 = np.einsum('ai,ijkl->ajkl', C, dm2_mo)
        pyscfRDM2 = np.einsum('bj,ajkl->abkl', C, pyscfRDM2)
        pyscfRDM2 = np.einsum('ck,abkl->abcl', C, pyscfRDM2)
        pyscfRDM2 = np.einsum('dl,abcl->abcd', C, pyscfRDM2)

        # --------------------------------------------------------------
        # DMET impurity energy (half-projector formula)
        # --------------------------------------------------------------
        ImpurityEnergy = (
            CONST
            + 0.25  * np.einsum('ij,ij->', pyscfRDM1[:Nimp,:],     FOCK[:Nimp,:] + OEI[:Nimp,:])
            + 0.25  * np.einsum('ij,ij->', pyscfRDM1[:,:Nimp],     FOCK[:,:Nimp] + OEI[:,:Nimp])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:Nimp,:,:,:], TEI[:Nimp,:,:,:])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:Nimp,:,:], TEI[:,:Nimp,:,:])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:,:Nimp,:], TEI[:,:,:Nimp,:])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:,:,:Nimp], TEI[:,:,:,:Nimp])
        )

    cas_results = {
        'e_tot'    : e_tot,
        'e_states' : e_states,
        'e_imp'    : ImpurityEnergy,
        'ncas'     : ncas,
        'nelecas'  : nelecas,
        'nstates'  : sa_nstates,
        'weights'  : sa_weights,
        'ci'       : mc.ci,
        'mo_coeff' : mc.mo_coeff,
    }

    return ImpurityEnergy, pyscfRDM1, cas_results
