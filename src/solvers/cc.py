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


def solve( const, oei, fock, tei, norb, nel, nimp, dm_guess_rhf,
           energytype='LAMBDA', chempot_imp=0.0, printoutput=True,
           eom_nroots=3 ):
    '''
    Solve a dmet impurity problem using coupled-cluster methods.

    Parameters
    ----------
    const        : float  – constant energy shift from core/frozen orbitals.
    oei          : ndarray (norb,norb) – one-electron integrals.
    fock         : ndarray (norb,norb) – Fock matrix (used for energy decomposition).
    tei          : ndarray (norb,norb,norb,norb) – two-electron integrals.
    norb         : int    – number of orbitals in the cluster.
    nel          : int    – number of electrons (must be even for RHF).
    nimp         : int    – number of impurity orbitals (first nimp of norb).
    dm_guess_rhf   : ndarray – initial density-matrix guess for RHF.
    energytype   : str    – CC variant (see module docstring above).
    chempot_imp  : float  – chemical potential applied to impurity orbitals.
    printoutput  : bool   – if True, print PySCF solver output.
    eom_nroots   : int    – number of EOM-CCSD roots to compute (only used when
                            energytype='EOM-CCSD').

    Returns
    -------
    impurity_energy : float    – impurity energy contribution.
    pyscf_rdm1      : ndarray  – 1-RDM in the local orbital basis (used for self-consistency).
    '''
    assert energytype in _VALID_ETYPES, \
        f"cc::solve: unrecognised energytype='{energytype}'. Valid: {_VALID_ETYPES}"

    ctx = silent_stdout() if not printoutput else nullcontext()

    # ------------------------------------------------------------------
    # Apply chemical potential to impurity block of fock
    # ------------------------------------------------------------------
    fock_copy = fock.copy()
    if chempot_imp != 0.0:
        for orb in range(nimp):
            fock_copy[orb, orb] -= chempot_imp

    with ctx:
        # --------------------------------------------------------------
        # Build a dummy PySCF Mole and run RHF
        # --------------------------------------------------------------
        mol = gto.Mole()
        mol.build(verbose=0)
        mol.atom.append(('C', (0, 0, 0)))
        mol.nelectron = nel
        mol.incore_anyway = True
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

        # Sanity checks on the RHF solution
        assert nel % 2 == 0
        numPairs = nel // 2
        fock_loc = fock_copy + np.einsum('ijkl,ij->kl', tei, dm_loc) \
                           - 0.5 * np.einsum('ijkl,ik->jl', tei, dm_loc)
        eigvals, eigvecs = np.linalg.eigh(fock_loc)
        idx = eigvals.argsort()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        print("cc::solve : RHF homo-lumo gap =", eigvals[numPairs] - eigvals[numPairs-1])
        dm_loc2 = 2 * np.dot(eigvecs[:, :numPairs], eigvecs[:, :numPairs].T)
        print("Two-norm difference of 1-RDM(RHF) and 1-RDM(fock(RHF)) =",
              np.linalg.norm(dm_loc - dm_loc2))

        # --------------------------------------------------------------
        # CCSD kernel  (always computed – needed by all variants)
        # --------------------------------------------------------------
        ccsolver = ccsd.CCSD(mf)
        ccsolver.verbose = 5
        e_corr, t1, t2 = ccsolver.ccsd()
        e_rhf  = mf.e_tot
        e_ccsd = e_rhf + e_corr

        # ==============================================================
        # CASCI energy type  – single-energy variant, no projection
        # ==============================================================
        if energytype == 'CASCI':
            print("e_ccsd =", e_ccsd)
            ccsolver.solve_lambda()
            pyscf_rdm1 = ccsolver.make_rdm1()
            pyscf_rdm1 = 0.5 * (pyscf_rdm1 + pyscf_rdm1.T)
            pyscf_rdm1 = np.dot(mf.mo_coeff, np.dot(pyscf_rdm1, mf.mo_coeff.T))
            impurity_energy = e_ccsd
            if chempot_imp != 0.0:
                impurity_energy += np.einsum('ij,ij->', fock - fock_copy, pyscf_rdm1)

        # ==============================================================
        # CCSD(T)  – perturbative triples correction, CCSD Lambda-RDM
        # ==============================================================
        elif energytype == 'CCSD(T)':
            from pyscf.cc import ccsd_t as _ccsd_t
            eris = ccsolver.ao2mo()
            ET   = _ccsd_t.kernel(ccsolver, eris, t1, t2)
            ECCSD_T = e_ccsd + ET
            print(f"cc::solve : e_ccsd = {e_ccsd:.10f},  E(T) = {ET:.10f},  e_ccsd(T) = {ECCSD_T:.10f}")

            # Use standard CCSD Lambda-RDM for self-consistency
            ccsolver.solve_lambda()
            pyscf_rdm1 = ccsolver.make_rdm1()
            pyscf_rdm2 = ccsolver.make_rdm2()
            pyscf_rdm1 = 0.5 * (pyscf_rdm1 + pyscf_rdm1.T)
            pyscf_rdm1, pyscf_rdm2 = _rotate_rdms_to_local(mf, pyscf_rdm1, pyscf_rdm2)

            impurity_energy = _compute_impurity_energy(
                const, fock_copy, oei, tei, nimp, pyscf_rdm1, pyscf_rdm2,
                fock, chempot_imp, extra_energy=ET
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
            ECCSD_T = e_ccsd + ET
            print(f"cc::solve : e_ccsd = {e_ccsd:.10f},  E(T) = {ET:.10f},  e_ccsd(T) = {ECCSD_T:.10f}")

            # Solve CCSD(T) lambda equations for the relaxed density matrix
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

        # ==============================================================
        # EOM-CCSD  – ground state CCSD for self-consistency;
        #             EOM singlet excitation energies printed as extras.
        # ==============================================================
        elif energytype == 'EOM-CCSD':
            # Run lambda + ground-state CCSD density matrices for self-consistency
            ccsolver.solve_lambda()
            pyscf_rdm1 = ccsolver.make_rdm1()
            pyscf_rdm2 = ccsolver.make_rdm2()
            pyscf_rdm1 = 0.5 * (pyscf_rdm1 + pyscf_rdm1.T)
            pyscf_rdm1, pyscf_rdm2 = _rotate_rdms_to_local(mf, pyscf_rdm1, pyscf_rdm2)
            impurity_energy = _compute_impurity_energy(
                const, fock_copy, oei, tei, nimp, pyscf_rdm1, pyscf_rdm2,
                fock, chempot_imp
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


# ---------------------------------------------------------------------------
# Private helpers – DRY up the repeated MO→local rotation and energy formula
# ---------------------------------------------------------------------------

def _rotate_rdms_to_local(mf, pyscf_rdm1, pyscf_rdm2):
    '''Rotate 1-RDM and 2-RDM from MO basis into the local (dmet orbital) basis.'''
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
    '''
    Compute the dmet impurity contribution to the total energy using the
    standard half-projector formula, with an optional additive extra_energy
    (e.g. the (T) perturbative correction split equally over impurity/bath).

    Parameters
    ----------
    extra_energy : float
        Additional energy term added to impurity_energy after the RDM-based
        projection. Used for the CCSD(T) perturbative correction — the
        convention here is to add the *full* (T) correction to the impurity
        energy (appropriate when there is a single impurity spanning the system).
        For multi-impurity tilings, consider scaling by nimp/norb.
    '''
    E_imp = const \
          + 0.25  * np.einsum('ij,ij->',     pyscf_rdm1[:nimp,:],     fock[:nimp,:] + oei[:nimp,:]) \
          + 0.25  * np.einsum('ij,ij->',     pyscf_rdm1[:,:nimp],     fock[:,:nimp] + oei[:,:nimp]) \
          + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:nimp,:,:,:], tei[:nimp,:,:,:]) \
          + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:,:nimp,:,:], tei[:,:nimp,:,:]) \
          + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:,:,:nimp,:], tei[:,:,:nimp,:]) \
          + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:,:,:,:nimp], tei[:,:,:,:nimp])

    # chempot already absorbed by fock_copy; no correction needed
    return E_imp + extra_energy


# ---------------------------------------------------------------------------
# SolverDispatcher entry point
# ---------------------------------------------------------------------------

def execute(task):
    """
    SolverDispatcher-compatible wrapper for the CC solver.

    Unpacks the standardised task dict and calls solve(). Supports all
    CC_E_TYPE variants: 'LAMBDA', 'LAMBDA_AMP', 'LAMBDA_ZERO', 'CASCI',
    'CCSD(T)', 'CCSD(T)_RDM', 'EOM-CCSD'.

    Parameters
    ----------
    task : dict
        Must contain: const, dmet_oei, dmet_fock, dmet_tei, norb, nel, nimp,
        dm_guess_rhf, chempot_imp.
        Optional: CC_E_TYPE (default 'LAMBDA'), eom_nroots (default 3).

    Returns
    -------
    (impurity_energy, pyscf_rdm1) — same as solve().
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
        energytype=task.get('CC_E_TYPE', 'LAMBDA'),
        chempot_imp=task.get('chempot_imp', 0.0),
        eom_nroots=task.get('eom_nroots', 3),
    )
