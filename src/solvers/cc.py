'''
    QC-dmet: a python implementation of density matrix embedding theory for ab initio quantum chemistry
    Copyright (C) 2015 Sebastian Wouters
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

import numpy as np
from pyscf import ao2mo, gto, scf
from pyscf.cc import ccsd
from ..utils import silent_stdout, nullcontext

_VALID_ETYPES = {'LAMBDA', 'LAMBDA_AMP', 'LAMBDA_ZERO', 'CASCI',
                 'CCSD(T)', 'CCSD(T)_RDM', 'EOM-CCSD'}


def solve( const, oei, fock, tei, norb, nel, nimp, dm_guess_rhf,
           energytype='LAMBDA', chempot_imp=0.0, printoutput=True,
           eom_nroots=3,
           use_density_fit=False, df_auxbasis=None ):
    assert energytype in _VALID_ETYPES, \
        f"cc::solve: unrecognised energytype='{energytype}'. Valid: {_VALID_ETYPES}"

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
        if use_density_fit:
            mf = mf.density_fit(auxbasis=df_auxbasis)
        mf.get_hcore = lambda *args: fock_copy
        mf.get_ovlp  = lambda *args: np.eye(norb)
        mf._eri      = ao2mo.restore(8, tei, norb)
        mf.scf(dm_guess_rhf)
        dm_loc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
        if not mf.converged:
            mf = mf.newton()
            mf.scf(dm_loc)
            dm_loc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)

        ccsolver = ccsd.CCSD(mf)
        ccsolver.verbose = 5
        e_corr, t1, t2 = ccsolver.ccsd()
        e_rhf  = mf.e_tot
        e_ccsd = e_rhf + e_corr
        _t1_norm = np.linalg.norm(t1) / np.sqrt(ccsolver.nocc)
        print(f"cc::solve : T1 norm = {_t1_norm:.4f}  (>0.02: moderate MR; >0.05: strong MR)")

        if energytype == 'CASCI':
            ccsolver.solve_lambda()
            pyscf_rdm1 = ccsolver.make_rdm1()
            pyscf_rdm1 = 0.5 * (pyscf_rdm1 + pyscf_rdm1.T)
            pyscf_rdm1 = np.dot(mf.mo_coeff, np.dot(pyscf_rdm1, mf.mo_coeff.T))
            impurity_energy = e_ccsd
            if chempot_imp != 0.0:
                impurity_energy += np.einsum('ij,ij->', fock - fock_copy, pyscf_rdm1)

        elif energytype == 'CCSD(T)':
            from pyscf.cc import ccsd_t as _ccsd_t
            eris = ccsolver.ao2mo()
            ET   = _ccsd_t.kernel(ccsolver, eris, t1, t2)
            ECCSD_T = e_ccsd + ET
            print(f"cc::solve : e_ccsd = {e_ccsd:.10f},  E(T) = {ET:.10f},  e_ccsd(T) = {ECCSD_T:.10f}")

            ccsolver.solve_lambda()
            pyscf_rdm1 = ccsolver.make_rdm1()
            pyscf_rdm2 = ccsolver.make_rdm2()
            pyscf_rdm1 = 0.5 * (pyscf_rdm1 + pyscf_rdm1.T)
            pyscf_rdm1, pyscf_rdm2 = _rotate_rdms_to_local(mf, pyscf_rdm1, pyscf_rdm2)

            impurity_energy = _compute_impurity_energy(
                const, fock_copy, oei, tei, nimp, pyscf_rdm1, pyscf_rdm2,
                fock, chempot_imp, extra_energy=ET
            )

        elif energytype == 'CCSD(T)_RDM':
            from pyscf.cc import ccsd_t as _ccsd_t
            from pyscf.cc import ccsd_t_lambda as _ccsd_t_lambda
            from pyscf.cc import ccsd_t_rdm   as _ccsd_t_rdm
            eris = ccsolver.ao2mo()
            ET   = _ccsd_t.kernel(ccsolver, eris, t1, t2)
            ECCSD_T = e_ccsd + ET
            print(f"cc::solve : e_ccsd = {e_ccsd:.10f},  E(T) = {ET:.10f},  e_ccsd(T) = {ECCSD_T:.10f}")

            print("cc::solve : Solving CCSD(T) lambda equations ...")
            l1, l2 = _ccsd_t_lambda.kernel(ccsolver, eris, t1, t2)[1:]
            pyscf_rdm1 = _ccsd_t_rdm.make_rdm1(ccsolver, t1, t2, l1, l2, eris)
            pyscf_rdm2 = _ccsd_t_rdm.make_rdm2(ccsolver, t1, t2, l1, l2, eris)
            pyscf_rdm1 = 0.5 * (pyscf_rdm1 + pyscf_rdm1.T)
            pyscf_rdm1, pyscf_rdm2 = _rotate_rdms_to_local(mf, pyscf_rdm1, pyscf_rdm2)

            impurity_energy = _compute_impurity_energy(
                const, fock_copy, oei, tei, nimp, pyscf_rdm1, pyscf_rdm2,
                fock, chempot_imp
            )

        elif energytype == 'EOM-CCSD':
            ccsolver.solve_lambda()
            pyscf_rdm1 = ccsolver.make_rdm1()
            pyscf_rdm2 = ccsolver.make_rdm2()
            pyscf_rdm1 = 0.5 * (pyscf_rdm1 + pyscf_rdm1.T)
            pyscf_rdm1, pyscf_rdm2 = _rotate_rdms_to_local(mf, pyscf_rdm1, pyscf_rdm2)
            impurity_energy = _compute_impurity_energy(
                const, fock_copy, oei, tei, nimp, pyscf_rdm1, pyscf_rdm2,
                fock, chempot_imp
            )

            print(f"\ncc::solve : Running EOM-CCSD for {eom_nroots} singlet root(s) ...")
            myeom = ccsolver.EOMEESinglet()
            e_exc, c_exc = myeom.kernel(nroots=eom_nroots)
            if not hasattr(e_exc, '__len__'):   # single root returns a scalar
                e_exc = [e_exc]
            print("cc::solve : EOM-CCSD singlet excitation energies (Ha / eV):")
            eV = 27.21138602  # Hartree to eV
            for idx, e in enumerate(e_exc):
                print(f"  State {idx+1}: {e:+.8f} Ha  ({e * eV:.4f} eV)")

        else:
            if energytype == 'LAMBDA':
                ccsolver.solve_lambda()
                pyscf_rdm1 = ccsolver.make_rdm1()
                pyscf_rdm2 = ccsolver.make_rdm2()
            elif energytype == 'LAMBDA_AMP':
                pyscf_rdm1 = ccsolver.make_rdm1(t1, t2, t1, t2)
                pyscf_rdm2 = ccsolver.make_rdm2(t1, t2, t1, t2)
            elif energytype == 'LAMBDA_ZERO':
                fake_l1 = np.zeros(t1.shape, dtype=float)
                fake_l2 = np.zeros(t2.shape, dtype=float)
                pyscf_rdm1 = ccsolver.make_rdm1(t1, t2, fake_l1, fake_l2)
                pyscf_rdm2 = ccsolver.make_rdm2(t1, t2, fake_l1, fake_l2)

            pyscf_rdm1 = 0.5 * (pyscf_rdm1 + pyscf_rdm1.T)
            pyscf_rdm1, pyscf_rdm2 = _rotate_rdms_to_local(mf, pyscf_rdm1, pyscf_rdm2)

            impurity_energy = _compute_impurity_energy(
                const, fock_copy, oei, tei, nimp, pyscf_rdm1, pyscf_rdm2,
                fock, chempot_imp
            )

    return (impurity_energy, pyscf_rdm1)


def _rotate_rdms_to_local(mf, pyscf_rdm1, pyscf_rdm2):
    C = mf.mo_coeff
    pyscf_rdm1 = np.dot(C, np.dot(pyscf_rdm1, C.T))
    pyscf_rdm2 = np.einsum('ai,ijkl->ajkl', C, pyscf_rdm2)
    pyscf_rdm2 = np.einsum('bj,ajkl->abkl', C, pyscf_rdm2)
    pyscf_rdm2 = np.einsum('ck,abkl->abcl', C, pyscf_rdm2)
    pyscf_rdm2 = np.einsum('dl,abcl->abcd', C, pyscf_rdm2)
    return pyscf_rdm1, pyscf_rdm2


def _compute_impurity_energy(const, fock_copy, oei, tei, nimp,
                              pyscf_rdm1, pyscf_rdm2, fock, chempot_imp,
                              extra_energy=0.0):
    E_imp = const \
          + 0.25  * np.einsum('ij,ij->',     pyscf_rdm1[:nimp,:],     fock[:nimp,:] + oei[:nimp,:]) \
          + 0.25  * np.einsum('ij,ij->',     pyscf_rdm1[:,:nimp],     fock[:,:nimp] + oei[:,:nimp]) \
          + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:nimp,:,:,:], tei[:nimp,:,:,:]) \
          + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:,:nimp,:,:], tei[:,:nimp,:,:]) \
          + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:,:,:nimp,:], tei[:,:,:nimp,:]) \
          + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:,:,:,:nimp], tei[:,:,:,:nimp])

    return E_imp + extra_energy


def execute(task):
    return solve(
        task['const'],
        task['dmet_oei'],
        task['dmet_fock'],
        task['dmet_tei'],
        task['norb'],
        task['nel'],
        task['nimp'],
        task.get('dm_guess_rhf'),
        energytype=task.get('CC_E_TYPE', 'LAMBDA'),
        chempot_imp=task.get('chempot_imp', 0.0),
        eom_nroots=task.get('eom_nroots', 3),
        use_density_fit=task.get('use_density_fit', False),
        df_auxbasis=task.get('df_auxbasis', None),
    )

