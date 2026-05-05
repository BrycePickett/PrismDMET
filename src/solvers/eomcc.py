'''
    EOM-CCSD solver for QC-DMET.

    Supports one-shot DMET only. Using this solver inside the self-consistent
    DMET loop (selfconsistent()) will raise a RuntimeError.

    Supported EOM-CCSD variants (set via eom_type parameter):
        'EE-Singlet'  — Excitation energies, singlet manifold (default)
        'EE-Triplet'  — Excitation energies, triplet manifold
        'EE-SpinFlip' — Spin-flip excitation energies
        'EE'          — General excitation energies (ms-conserving)
        'IP'          — Ionization potentials (N -> N-1)
        'IP*'         — Perturbative IP-EOMCCSDStar correction
        'EA'          — Electron affinities (N -> N+1)
        'EA*'         — Perturbative EA-EOMCCSDStar correction

    The solver returns the ground-state CCSD energy as ImpurityEnergy so that
    the DMET chemical potential optimization (numeleccostfunction) is stable.
    Excitation energies and absolute excited-state energies are accessible via
    the 'results' key returned in the auxiliary dictionary.
'''

import numpy as np
from pyscf import ao2mo, gto, scf
from pyscf.cc import ccsd
from utils import silent_stdout, nullcontext

_VALID_EOM_TYPES = {
    'EE-Singlet', 'EE-Triplet', 'EE-SpinFlip', 'EE',
    'IP', 'IP*', 'EA', 'EA*',
}

_eV = 27.21138602  # Hartree to eV


def solve(CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, DMguessRHF,
          chempot_imp=0.0, printoutput=True,
          eom_type='EE-Singlet', nroots=3, koopmans=False, **eom_kwargs):
    '''
    Solve a DMET impurity problem with CCSD followed by EOM-CCSD.

    The ground-state CCSD energy and Lambda-RDM are used for the DMET
    embedding (chemical-potential optimisation and energy projection).
    EOM-CCSD provides excitation/ionisation/attachment energies on top.

    Parameters
    ----------
    CONST        : float
    OEI          : ndarray (Norb, Norb)
    FOCK         : ndarray (Norb, Norb)
    TEI          : ndarray (Norb, Norb, Norb, Norb)
    Norb         : int
    Nel          : int  (must be even for RHF)
    Nimp         : int
    DMguessRHF   : ndarray
    chempot_imp  : float
    printoutput  : bool
    eom_type     : str   — one of _VALID_EOM_TYPES (default 'EE-Singlet')
    nroots       : int   — number of EOM roots (default 3)
    koopmans     : bool  — use Koopmans-like initial guess (default False)
    **eom_kwargs : extra keyword arguments forwarded to the EOM kernel, e.g.
                   partition='mp'  for IP/EA, guess=<custom>

    Returns
    -------
    ImpurityEnergy : float   — CCSD ground-state impurity energy (for DMET loop)
    pyscfRDM1      : ndarray — CCSD Lambda 1-RDM in local basis (for DMET loop)
    eom_results    : dict    — {
        'eom_type'    : str,
        'E_ccsd'      : float,          ground-state CCSD energy (absolute)
        'delta_E'     : ndarray,        excitation energies in Ha
        'delta_E_eV'  : ndarray,        excitation energies in eV
        'E_states'    : ndarray,        absolute state energies  = E_ccsd + delta_E
        'amplitudes'  : list of arrays, EOM right eigenvectors
    }
    '''
    if eom_type not in _VALID_EOM_TYPES:
        raise ValueError(
            f"eomcc::solve: unknown eom_type='{eom_type}'. "
            f"Valid options: {sorted(_VALID_EOM_TYPES)}"
        )

    ctx = silent_stdout() if not printoutput else nullcontext()

    FOCKcopy = FOCK.copy()
    if chempot_imp != 0.0:
        for orb in range(Nimp):
            FOCKcopy[orb, orb] -= chempot_imp

    with ctx:
        # ------------------------------------------------------------------
        # RHF in the DMET embedding space
        # ------------------------------------------------------------------
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

        assert Nel % 2 == 0
        numPairs = Nel // 2
        FOCKloc = (FOCKcopy
                   + np.einsum('ijkl,ij->kl', TEI, DMloc)
                   - 0.5 * np.einsum('ijkl,ik->jl', TEI, DMloc))
        eigvals, eigvecs = np.linalg.eigh(FOCKloc)
        idx = eigvals.argsort()
        eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
        print("eomcc::solve : RHF homo-lumo gap =", eigvals[numPairs] - eigvals[numPairs-1])

        # ------------------------------------------------------------------
        # Ground-state CCSD + lambda equations
        # ------------------------------------------------------------------
        ccsolver = ccsd.CCSD(mf)
        ccsolver.verbose = 5
        ECORR, t1, t2 = ccsolver.ccsd()
        ERHF  = mf.e_tot
        ECCSD = ERHF + ECORR
        print(f"eomcc::solve : E(RHF) = {ERHF:.10f}  E(CCSD) = {ECCSD:.10f}")

        ccsolver.solve_lambda()
        pyscfRDM1 = ccsolver.make_rdm1()
        pyscfRDM2 = ccsolver.make_rdm2()
        pyscfRDM1 = 0.5 * (pyscfRDM1 + pyscfRDM1.T)

        # Rotate RDMs to local orbital basis
        C = mf.mo_coeff
        pyscfRDM1 = np.dot(C, np.dot(pyscfRDM1, C.T))
        pyscfRDM2 = np.einsum('ai,ijkl->ajkl', C, pyscfRDM2)
        pyscfRDM2 = np.einsum('bj,ajkl->abkl', C, pyscfRDM2)
        pyscfRDM2 = np.einsum('ck,abkl->abcl', C, pyscfRDM2)
        pyscfRDM2 = np.einsum('dl,abcl->abcd', C, pyscfRDM2)

        # Ground-state impurity energy (DMET half-projector)
        ImpurityEnergy = (
            CONST
            + 0.25  * np.einsum('ij,ij->', pyscfRDM1[:Nimp,:],     FOCK[:Nimp,:] + OEI[:Nimp,:])
            + 0.25  * np.einsum('ij,ij->', pyscfRDM1[:,:Nimp],     FOCK[:,:Nimp] + OEI[:,:Nimp])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:Nimp,:,:,:], TEI[:Nimp,:,:,:])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:Nimp,:,:], TEI[:,:Nimp,:,:])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:,:Nimp,:], TEI[:,:,:Nimp,:])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:,:,:Nimp], TEI[:,:,:,:Nimp])
        )

        # ------------------------------------------------------------------
        # EOM-CCSD  –  choose the correct class and call its kernel
        # ------------------------------------------------------------------
        print(f"\neomcc::solve : Running EOM-CCSD [{eom_type}] for {nroots} root(s) ...")

        eom_obj, e_exc, amplitudes = _run_eom(
            ccsolver, eom_type, nroots, koopmans, eom_kwargs
        )

        # Normalise output (single root → list)
        if not hasattr(e_exc, '__len__'):
            e_exc   = np.array([e_exc])
            amplitudes = [amplitudes]
        else:
            e_exc = np.asarray(e_exc)

        E_states = ECCSD + e_exc

        print(f"\neomcc::solve : EOM-CCSD [{eom_type}] energies")
        print(f"  {'State':>6}  {'ΔE (Ha)':>14}  {'ΔE (eV)':>12}  {'E_abs (Ha)':>16}")
        print("  " + "-"*54)
        for i, (de, eabs) in enumerate(zip(e_exc, E_states)):
            print(f"  {i:>6d}  {de:>+14.8f}  {de*_eV:>12.4f}  {eabs:>16.10f}")

    eom_results = {
        'eom_type'   : eom_type,
        'E_ccsd'     : ECCSD,
        'delta_E'    : e_exc,
        'delta_E_eV' : e_exc * _eV,
        'E_states'   : E_states,
        'amplitudes' : amplitudes,
    }

    return ImpurityEnergy, pyscfRDM1, eom_results


# ---------------------------------------------------------------------------
# Internal dispatcher: selects the right EOM class and runs kernel
# ---------------------------------------------------------------------------

def _run_eom(ccsolver, eom_type, nroots, koopmans, eom_kwargs):
    '''
    Instantiate the correct EOM object and call its kernel.

    Returns (eom_obj, e_exc, amplitudes).
    '''
    from pyscf.cc import eom_rccsd

    if eom_type == 'EE-Singlet':
        eom = eom_rccsd.EOMEESinglet(ccsolver)
        e, v = eom.kernel(nroots=nroots, koopmans=koopmans, **eom_kwargs)

    elif eom_type == 'EE-Triplet':
        eom = eom_rccsd.EOMEETriplet(ccsolver)
        e, v = eom.kernel(nroots=nroots, koopmans=koopmans, **eom_kwargs)

    elif eom_type == 'EE-SpinFlip':
        eom = eom_rccsd.EOMEESpinFlip(ccsolver)
        e, v = eom.kernel(nroots=nroots, koopmans=koopmans, **eom_kwargs)

    elif eom_type == 'EE':
        eom = eom_rccsd.EOMEE(ccsolver)
        e, v = eom.kernel(nroots=nroots, koopmans=koopmans, **eom_kwargs)

    elif eom_type == 'IP':
        eom = eom_rccsd.EOMIP(ccsolver)
        e, v = eom.kernel(nroots=nroots, koopmans=koopmans, **eom_kwargs)

    elif eom_type == 'IP*':
        eom = eom_rccsd.EOMIP_Ta(ccsolver)
        e, v = eom.kernel(nroots=nroots, koopmans=koopmans, **eom_kwargs)

    elif eom_type == 'EA':
        eom = eom_rccsd.EOMEA(ccsolver)
        e, v = eom.kernel(nroots=nroots, koopmans=koopmans, **eom_kwargs)

    elif eom_type == 'EA*':
        eom = eom_rccsd.EOMEA_Ta(ccsolver)
        e, v = eom.kernel(nroots=nroots, koopmans=koopmans, **eom_kwargs)

    else:
        raise ValueError(f'_run_eom: unhandled eom_type={eom_type!r}')

    return eom, e, v


# ---------------------------------------------------------------------------
# SolverFactory entry point
# ---------------------------------------------------------------------------

def execute(task):
    """
    SolverFactory-compatible wrapper for the EOM-CCSD solver.

    Unpacks the standardised task dict and calls solve(), which returns
    a 3-tuple: (ImpurityEnergy, pyscfRDM1, eom_results).

    Parameters
    ----------
    task : dict
        Must contain: CONST, dmetOEI, dmetFOCK, dmetTEI, Norb, Nel, Nimp,
        DMguessRHF, chempot_imp.
        Optional: eom_type (default 'EE-Singlet'), eom_nroots (default 3),
        eom_koopmans (default False), eom_kwargs (default {}).

    Returns
    -------
    (ImpurityEnergy, pyscfRDM1, eom_results) — same as solve().
    """
    return solve(
        task['CONST'],
        task['dmetOEI'],
        task['dmetFOCK'],
        task['dmetTEI'],
        task['Norb'],
        task['Nel'],
        task['Nimp'],
        task.get('DMguessRHF'),
        chempot_imp=task.get('chempot_imp', 0.0),
        eom_type=task.get('eom_type', 'EE-Singlet'),
        nroots=task.get('eom_nroots', 3),
        koopmans=task.get('eom_koopmans', False),
        **task.get('eom_kwargs', {}),
    )
