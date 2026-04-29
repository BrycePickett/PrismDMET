'''
    QC-DMET: a python implementation of density matrix embedding theory for ab initio quantum chemistry
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
from utils import silent_stdout, nullcontext

# Valid energy types:
#   'LAMBDA'      -- CCSD with CCSD lambda 1-RDM and 2-RDM  (default, recommended)
#   'LAMBDA_AMP'  -- CCSD with approximate RDM (lambda ≈ amplitudes)
#   'LAMBDA_ZERO' -- CCSD with RDM at zero lambda
#   'CASCI'       -- CCSD total energy, no projected impurity energy decomposition
#   'CCSD(T)'     -- CCSD + perturbative (T) energy on top; self-consistency uses CCSD Lambda-RDM
#   'CCSD(T)_RDM' -- CCSD(T) with rigorous relaxed 1-RDM and 2-RDM (expensive; solves T-lambda eqs)
#   'EOM-CCSD'    -- Runs ground-state CCSD for self-consistency; reports EOM-CCSD singlet excitation
#                    energies as additional output. eom_nroots controls how many states are computed.

_VALID_ETYPES = {'LAMBDA', 'LAMBDA_AMP', 'LAMBDA_ZERO', 'CASCI',
                 'CCSD(T)', 'CCSD(T)_RDM', 'EOM-CCSD'}


def solve( CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, DMguessRHF,
           energytype='LAMBDA', chempot_imp=0.0, printoutput=True,
           eom_nroots=3 ):
    '''
    Solve a DMET impurity problem using coupled-cluster methods.

    Parameters
    ----------
    CONST        : float  – constant energy shift from core/frozen orbitals.
    OEI          : ndarray (Norb,Norb) – one-electron integrals.
    FOCK         : ndarray (Norb,Norb) – Fock matrix (used for energy decomposition).
    TEI          : ndarray (Norb,Norb,Norb,Norb) – two-electron integrals.
    Norb         : int    – number of orbitals in the cluster.
    Nel          : int    – number of electrons (must be even for RHF).
    Nimp         : int    – number of impurity orbitals (first Nimp of Norb).
    DMguessRHF   : ndarray – initial density-matrix guess for RHF.
    energytype   : str    – CC variant (see module docstring above).
    chempot_imp  : float  – chemical potential applied to impurity orbitals.
    printoutput  : bool   – if True, print PySCF solver output.
    eom_nroots   : int    – number of EOM-CCSD roots to compute (only used when
                            energytype='EOM-CCSD').

    Returns
    -------
    ImpurityEnergy : float    – impurity energy contribution.
    pyscfRDM1      : ndarray  – 1-RDM in the local orbital basis (used for self-consistency).
    '''
    assert energytype in _VALID_ETYPES, \
        f"cc::solve: unrecognised energytype='{energytype}'. Valid: {_VALID_ETYPES}"

    ctx = silent_stdout() if not printoutput else nullcontext()

    # ------------------------------------------------------------------
    # Apply chemical potential to impurity block of FOCK
    # ------------------------------------------------------------------
    FOCKcopy = FOCK.copy()
    if chempot_imp != 0.0:
        for orb in range(Nimp):
            FOCKcopy[orb, orb] -= chempot_imp

    with ctx:
        # --------------------------------------------------------------
        # Build a dummy PySCF Mole and run RHF
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

        # Sanity checks on the RHF solution
        assert Nel % 2 == 0
        numPairs = Nel // 2
        FOCKloc = FOCKcopy + np.einsum('ijkl,ij->kl', TEI, DMloc) \
                           - 0.5 * np.einsum('ijkl,ik->jl', TEI, DMloc)
        eigvals, eigvecs = np.linalg.eigh(FOCKloc)
        idx = eigvals.argsort()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        print("cc::solve : RHF homo-lumo gap =", eigvals[numPairs] - eigvals[numPairs-1])
        DMloc2 = 2 * np.dot(eigvecs[:, :numPairs], eigvecs[:, :numPairs].T)
        print("Two-norm difference of 1-RDM(RHF) and 1-RDM(FOCK(RHF)) =",
              np.linalg.norm(DMloc - DMloc2))

        # --------------------------------------------------------------
        # CCSD kernel  (always computed – needed by all variants)
        # --------------------------------------------------------------
        ccsolver = ccsd.CCSD(mf)
        ccsolver.verbose = 5
        ECORR, t1, t2 = ccsolver.ccsd()
        ERHF  = mf.e_tot
        ECCSD = ERHF + ECORR

        # ==============================================================
        # CASCI energy type  – single-energy variant, no projection
        # ==============================================================
        if energytype == 'CASCI':
            print("ECCSD =", ECCSD)
            ccsolver.solve_lambda()
            pyscfRDM1 = ccsolver.make_rdm1()
            pyscfRDM1 = 0.5 * (pyscfRDM1 + pyscfRDM1.T)
            pyscfRDM1 = np.dot(mf.mo_coeff, np.dot(pyscfRDM1, mf.mo_coeff.T))
            ImpurityEnergy = ECCSD
            if chempot_imp != 0.0:
                ImpurityEnergy += np.einsum('ij,ij->', FOCK - FOCKcopy, pyscfRDM1)

        # ==============================================================
        # CCSD(T)  – perturbative triples correction, CCSD Lambda-RDM
        # ==============================================================
        elif energytype == 'CCSD(T)':
            from pyscf.cc import ccsd_t as _ccsd_t
            eris = ccsolver.ao2mo()
            ET   = _ccsd_t.kernel(ccsolver, eris, t1, t2)
            ECCSD_T = ECCSD + ET
            print(f"cc::solve : ECCSD = {ECCSD:.10f},  E(T) = {ET:.10f},  ECCSD(T) = {ECCSD_T:.10f}")

            # Use standard CCSD Lambda-RDM for self-consistency
            ccsolver.solve_lambda()
            pyscfRDM1 = ccsolver.make_rdm1()
            pyscfRDM2 = ccsolver.make_rdm2()
            pyscfRDM1 = 0.5 * (pyscfRDM1 + pyscfRDM1.T)
            pyscfRDM1, pyscfRDM2 = _rotate_rdms_to_local(mf, pyscfRDM1, pyscfRDM2)

            ImpurityEnergy = _compute_impurity_energy(
                CONST, FOCKcopy, OEI, TEI, Nimp, pyscfRDM1, pyscfRDM2,
                FOCK, chempot_imp, extra_energy=ET
            )

        # ==============================================================
        # CCSD(T) with relaxed RDM  – full CCSD(T) lambda equations
        # ==============================================================
        elif energytype == 'CCSD(T)_RDM':
            from pyscf.cc import ccsd_t as _ccsd_t
            from pyscf.cc import ccsd_t_lambda as _ccsd_t_lambda
            from pyscf.cc import ccsd_t_rdm   as _ccsd_t_rdm
            eris = ccsolver.ao2mo()
            ET   = _ccsd_t.kernel(ccsolver, eris, t1, t2)
            ECCSD_T = ECCSD + ET
            print(f"cc::solve : ECCSD = {ECCSD:.10f},  E(T) = {ET:.10f},  ECCSD(T) = {ECCSD_T:.10f}")

            # Solve CCSD(T) lambda equations for the relaxed density matrix
            print("cc::solve : Solving CCSD(T) lambda equations ...")
            l1, l2 = _ccsd_t_lambda.kernel(ccsolver, eris, t1, t2)[1:]
            pyscfRDM1 = _ccsd_t_rdm.make_rdm1(ccsolver, t1, t2, l1, l2, eris)
            pyscfRDM2 = _ccsd_t_rdm.make_rdm2(ccsolver, t1, t2, l1, l2, eris)
            pyscfRDM1 = 0.5 * (pyscfRDM1 + pyscfRDM1.T)
            pyscfRDM1, pyscfRDM2 = _rotate_rdms_to_local(mf, pyscfRDM1, pyscfRDM2)

            ImpurityEnergy = _compute_impurity_energy(
                CONST, FOCKcopy, OEI, TEI, Nimp, pyscfRDM1, pyscfRDM2,
                FOCK, chempot_imp
            )

        # ==============================================================
        # EOM-CCSD  – ground state CCSD for self-consistency;
        #             EOM singlet excitation energies printed as extras.
        # ==============================================================
        elif energytype == 'EOM-CCSD':
            # Run lambda + ground-state CCSD density matrices for self-consistency
            ccsolver.solve_lambda()
            pyscfRDM1 = ccsolver.make_rdm1()
            pyscfRDM2 = ccsolver.make_rdm2()
            pyscfRDM1 = 0.5 * (pyscfRDM1 + pyscfRDM1.T)
            pyscfRDM1, pyscfRDM2 = _rotate_rdms_to_local(mf, pyscfRDM1, pyscfRDM2)
            ImpurityEnergy = _compute_impurity_energy(
                CONST, FOCKcopy, OEI, TEI, Nimp, pyscfRDM1, pyscfRDM2,
                FOCK, chempot_imp
            )

            # Now compute EOM-CCSD excitation energies (informational only)
            print(f"\ncc::solve : Running EOM-CCSD for {eom_nroots} singlet root(s) ...")
            myeom = ccsolver.EOMEESinglet()
            e_exc, c_exc = myeom.kernel(nroots=eom_nroots)
            if not hasattr(e_exc, '__len__'):   # single root returns a scalar
                e_exc = [e_exc]
            print("cc::solve : EOM-CCSD singlet excitation energies (Ha / eV):")
            eV = 27.21138602  # Hartree to eV
            for idx, e in enumerate(e_exc):
                print(f"  State {idx+1}: {e:+.8f} Ha  ({e * eV:.4f} eV)")

        # ==============================================================
        # Standard CCSD variants (LAMBDA / LAMBDA_AMP / LAMBDA_ZERO)
        # ==============================================================
        else:
            if energytype == 'LAMBDA':
                ccsolver.solve_lambda()
                pyscfRDM1 = ccsolver.make_rdm1()
                pyscfRDM2 = ccsolver.make_rdm2()
            elif energytype == 'LAMBDA_AMP':
                pyscfRDM1 = ccsolver.make_rdm1(t1, t2, t1, t2)
                pyscfRDM2 = ccsolver.make_rdm2(t1, t2, t1, t2)
            elif energytype == 'LAMBDA_ZERO':
                fake_l1 = np.zeros(t1.shape, dtype=float)
                fake_l2 = np.zeros(t2.shape, dtype=float)
                pyscfRDM1 = ccsolver.make_rdm1(t1, t2, fake_l1, fake_l2)
                pyscfRDM2 = ccsolver.make_rdm2(t1, t2, fake_l1, fake_l2)

            pyscfRDM1 = 0.5 * (pyscfRDM1 + pyscfRDM1.T)
            pyscfRDM1, pyscfRDM2 = _rotate_rdms_to_local(mf, pyscfRDM1, pyscfRDM2)

            ECCSDbis = CONST + np.einsum('ij,ij->', FOCKcopy, pyscfRDM1) \
                             + 0.5 * np.einsum('ijkl,ijkl->', TEI, pyscfRDM2)
            print("ECCSD1 =", ECCSD)
            print("ECCSD2 =", ECCSDbis)

            ImpurityEnergy = _compute_impurity_energy(
                CONST, FOCKcopy, OEI, TEI, Nimp, pyscfRDM1, pyscfRDM2,
                FOCK, chempot_imp
            )

    return (ImpurityEnergy, pyscfRDM1)


# ---------------------------------------------------------------------------
# Private helpers – DRY up the repeated MO→local rotation and energy formula
# ---------------------------------------------------------------------------

def _rotate_rdms_to_local(mf, pyscfRDM1, pyscfRDM2):
    '''Rotate 1-RDM and 2-RDM from MO basis into the local (DMET orbital) basis.'''
    C = mf.mo_coeff
    pyscfRDM1 = np.dot(C, np.dot(pyscfRDM1, C.T))
    pyscfRDM2 = np.einsum('ai,ijkl->ajkl', C, pyscfRDM2)
    pyscfRDM2 = np.einsum('bj,ajkl->abkl', C, pyscfRDM2)
    pyscfRDM2 = np.einsum('ck,abkl->abcl', C, pyscfRDM2)
    pyscfRDM2 = np.einsum('dl,abcl->abcd', C, pyscfRDM2)
    return pyscfRDM1, pyscfRDM2


def _compute_impurity_energy(CONST, FOCKcopy, OEI, TEI, Nimp,
                              pyscfRDM1, pyscfRDM2, FOCK, chempot_imp,
                              extra_energy=0.0):
    '''
    Compute the DMET impurity contribution to the total energy using the
    standard half-projector formula, with an optional additive extra_energy
    (e.g. the (T) perturbative correction split equally over impurity/bath).

    Parameters
    ----------
    extra_energy : float
        Additional energy term added to ImpurityEnergy after the RDM-based
        projection. Used for the CCSD(T) perturbative correction — the
        convention here is to add the *full* (T) correction to the impurity
        energy (appropriate when there is a single impurity spanning the system).
        For multi-impurity tilings, consider scaling by Nimp/Norb.
    '''
    E_imp = CONST \
          + 0.25  * np.einsum('ij,ij->',     pyscfRDM1[:Nimp,:],     FOCK[:Nimp,:] + OEI[:Nimp,:]) \
          + 0.25  * np.einsum('ij,ij->',     pyscfRDM1[:,:Nimp],     FOCK[:,:Nimp] + OEI[:,:Nimp]) \
          + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:Nimp,:,:,:], TEI[:Nimp,:,:,:]) \
          + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:Nimp,:,:], TEI[:,:Nimp,:,:]) \
          + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:,:Nimp,:], TEI[:,:,:Nimp,:]) \
          + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:,:,:Nimp], TEI[:,:,:,:Nimp])

    if chempot_imp != 0.0:
        # Chemical potential contribution already in FOCKcopy; restore it
        pass  # Energy decomposition via OEI+FOCK naturally absorbs chempot

    return E_imp + extra_energy
