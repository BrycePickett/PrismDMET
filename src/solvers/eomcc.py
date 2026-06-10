'''EOM-CCSD solver for DMET embedding clusters (one-shot only); eom_type sets variant: EE-Singlet/Triplet/SpinFlip, EE, IP/IP*, EA/EA*.'''

import numpy as np
from pyscf import ao2mo, gto, scf
from pyscf.cc import ccsd
from ..utils import silent_stdout, nullcontext

_VALID_EOM_TYPES = {
    'EE-Singlet', 'EE-Triplet', 'EE-SpinFlip', 'EE',
    'IP', 'IP*', 'EA', 'EA*',
}

_eV = 27.21138602  # Hartree to eV


def solve(const, oei, fock, tei, norb, nel, nimp, dm_guess_rhf,
          chempot_imp=0.0, printoutput=True,
          eom_type='EE-Singlet', nroots=3, koopmans=False, **eom_kwargs):
    '''Run CCSD + EOM-CCSD on the DMET embedding cluster. Returns (impurity_energy, rdm1, eom_results).'''
    if eom_type not in _VALID_EOM_TYPES:
        raise ValueError(
            f"eomcc::solve: unknown eom_type='{eom_type}'. "
            f"Valid options: {sorted(_VALID_EOM_TYPES)}"
        )

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
        mol.spin      = nel % 2
        mol.incore_anyway = True
        mf = scf.ROHF(mol) if mol.spin != 0 else scf.RHF(mol)
        mf.get_hcore = lambda *args: fock_copy
        mf.get_ovlp  = lambda *args: np.eye(norb)
        mf._eri      = ao2mo.restore(8, tei, norb)
        mf.scf(dm_guess_rhf)
        dm_loc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
        if not mf.converged:
            # Newton/SOSCF rebuilds _eri from the dummy mol and crashes; retry plain SCF.
            mf.max_cycle = 300
            mf.diis_space = 12
            mf.scf(dm_loc)
            dm_loc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)

        ccsolver = ccsd.CCSD(mf)
        ccsolver.verbose = 5
        e_corr, t1, t2 = ccsolver.ccsd()
        e_ccsd = mf.e_tot + e_corr

        ccsolver.solve_lambda()
        pyscf_rdm1 = ccsolver.make_rdm1()
        pyscf_rdm2 = ccsolver.make_rdm2()
        pyscf_rdm1 = 0.5 * (pyscf_rdm1 + pyscf_rdm1.T)

        # Rotate RDMs to local orbital basis
        C = mf.mo_coeff
        pyscf_rdm1 = np.dot(C, np.dot(pyscf_rdm1, C.T))
        pyscf_rdm2 = np.einsum('ai,ijkl->ajkl', C, pyscf_rdm2)
        pyscf_rdm2 = np.einsum('bj,ajkl->abkl', C, pyscf_rdm2)
        pyscf_rdm2 = np.einsum('ck,abkl->abcl', C, pyscf_rdm2)
        pyscf_rdm2 = np.einsum('dl,abcl->abcd', C, pyscf_rdm2)

        # Ground-state impurity energy (dmet half-projector)
        impurity_energy = (
            const
            + 0.25  * np.einsum('ij,ij->', pyscf_rdm1[:nimp,:],     fock[:nimp,:] + oei[:nimp,:])
            + 0.25  * np.einsum('ij,ij->', pyscf_rdm1[:,:nimp],     fock[:,:nimp] + oei[:,:nimp])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:nimp,:,:,:], tei[:nimp,:,:,:])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:,:nimp,:,:], tei[:,:nimp,:,:])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:,:,:nimp,:], tei[:,:,:nimp,:])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:,:,:,:nimp], tei[:,:,:,:nimp])
        )

        print(f"\neomcc::solve : Running EOM-CCSD [{eom_type}] for {nroots} root(s) ...")

        eom_obj, e_exc, amplitudes = _run_eom(
            ccsolver, eom_type, nroots, koopmans, eom_kwargs
        )

        # Normalize output (single root → list)
        if not hasattr(e_exc, '__len__'):
            e_exc   = np.array([e_exc])
            amplitudes = [amplitudes]
        else:
            e_exc = np.asarray(e_exc)

        E_states = e_ccsd + e_exc

        print(f"\neomcc::solve : EOM-CCSD [{eom_type}] energies")
        print(f"  {'State':>6}  {'ΔE (Ha)':>14}  {'ΔE (eV)':>12}  {'E_abs (Ha)':>16}")
        print("  " + "-"*54)
        for i, (de, eabs) in enumerate(zip(e_exc, E_states)):
            print(f"  {i:>6d}  {de:>+14.8f}  {de*_eV:>12.4f}  {eabs:>16.10f}")

    eom_results = {
        'eom_type'   : eom_type,
        'E_ccsd'     : e_ccsd,
        'delta_E'    : e_exc,
        'delta_E_eV' : e_exc * _eV,
        'E_states'   : E_states,
        'amplitudes' : amplitudes,
    }

    return impurity_energy, pyscf_rdm1, eom_results


def _run_eom(ccsolver, eom_type, nroots, koopmans, eom_kwargs):
    '''Instantiate the correct EOM object and return (eom_obj, e_exc, amplitudes).'''
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


def execute(task):
    """SolverDispatcher entry point for the EOM-CCSD solver."""
    return solve(
        task['const'],
        task['dmet_oei'],
        task['dmet_fock'],
        task['dmet_tei'],
        task['norb'],
        task['nel'],
        task['nimp'],
        task.get('dm_guess_rhf'),
        chempot_imp=task.get('chempot_imp', 0.0),
        eom_type=task.get('eom_type', 'EE-Singlet'),
        nroots=task.get('eom_nroots', 3),
        koopmans=task.get('eom_koopmans', False),
        **task.get('eom_kwargs', {}),
    )
