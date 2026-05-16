"""
Prismdmet - Density Matrix Embedding Theory for ab initio quantum chemistry.
Built on the QC-dmet framework (Wouters et al., 2015) under GPL-v2.
"""

from pyscf import gto, scf, ao2mo, lo
from pyscf.lo import nao, orth
from pyscf.tools import molden
from . import rhf
from . import iao_helper
import numpy as np


class LocalIntegrals:
    """
    Constructs and stores the localized molecular orbital (LMO) basis and
    the corresponding 1e/2e integrals required for dmet embedding.

    Supports meta-Lowdin, Boys, Lowdin, and IAO localization schemes.
    Handles both closed-shell (RHF) and open-shell (ROHF/UHF) references.
    """

    def __init__(self, the_mf, active_orbs, localizationtype,
                 ao_rotation=None, use_full_hessian=True,
                 localization_threshold=1e-6):
        """
        Parameters
        ----------
        the_mf : pyscf SCF object
            Converged mean-field (RHF, ROHF, or UHF).
        active_orbs : list of int
            Indices of active (non-frozen) orbitals.
        localizationtype : str
            Localization scheme: 'meta_lowdin', 'boys', 'lowdin', or 'iao'.
        ao_rotation : ndarray, optional
            Optional rotation applied to the localized basis after construction.
        localization_threshold : float
            Convergence threshold for Boys localization.
        """
        assert localizationtype in ('meta_lowdin', 'boys', 'lowdin', 'iao')

        # Mean-field reference data
        self.mol     = the_mf.mol
        self.the_mf  = the_mf
        self.fullEhf = the_mf.e_tot
        _dm = the_mf.make_rdm1()
        if _dm.ndim == 3:  # UHF: (2, nao, nao)
            self.fullDMao = _dm[0] + _dm[1]
            _v = the_mf.get_veff(self.mol, _dm)        # 3D input → correct per-spin exchange
            self.fullJKao = 0.5 * (_v[0] + _v[1])     # spin-averaged Veff
        else:
            self.fullDMao = _dm
            _v = the_mf.get_veff(self.mol, self.fullDMao)
            self.fullJKao = _v[0] if _v.ndim == 3 else _v
        self.fullFOCKao = the_mf.get_hcore() + self.fullJKao

        # Spin potential computed after ao2loc is built (see below)
        self.activeVSPIN = None

        # Active space bookkeeping
        self._which  = localizationtype
        self.active  = np.zeros([self.mol.nao_nr()], dtype=int)
        self.active[active_orbs] = 1
        self.Norbs   = np.sum(self.active)
        _mo_occ = the_mf.mo_occ
        _frozen_mask = self.active == 0
        if _mo_occ.ndim == 2:  # UHF: sum frozen electrons from both spin channels
            if np.any(_frozen_mask):
                raise NotImplementedError(
                    "Frozen-core UHF references are not supported. "
                    "Pass active_orbs=list(range(mol.nao_nr()))."
                )
            _frozen_elec = 0.0
        else:
            _frozen_elec = np.sum(_mo_occ[_frozen_mask])
        self.Nelec = int(np.rint(self.mol.nelectron - _frozen_elec))

        # Build the AO → LMO transformation (ao2loc)
        if self._which in ('meta_lowdin', 'boys'):
            if self._which == 'meta_lowdin':
                assert self.Norbs == self.mol.nao_nr(), "meta_lowdin requires full active space"
            if self._which == 'boys':
                if the_mf.mo_coeff.ndim == 3:
                    raise NotImplementedError(
                        "Boys localization requires a single set of spatial orbitals. "
                        "Use 'meta_lowdin' or 'iao' for UHF references."
                    )
                self.ao2loc = the_mf.mo_coeff[:, self.active == 1]
            if self.Norbs == self.mol.nao_nr():
                nao.AOSHELL[4] = ['1s0p0d0f', '2s1p0d0f']  # redefine Be valence shell
                self.ao2loc = orth.orth_ao(self.mol, 'meta_lowdin')
                if ao_rotation is not None:
                    self.ao2loc = np.dot(self.ao2loc, ao_rotation.T)
            if self._which == 'boys':
                old_verbose = self.mol.verbose
                self.mol.verbose = 5
                loc = lo.Boys(self.mol, self.ao2loc)
                loc.conv_tol = localization_threshold
                self.mol.verbose = old_verbose
                self.ao2loc = loc.kernel()
            self.TI_OK = False
        if self._which == 'lowdin':
            assert self.Norbs == self.mol.nao_nr(), "lowdin requires full active space"
            ovlp = self.mol.intor('cint1e_ovlp_sph')
            ovlp_eigs, ovlp_vecs = np.linalg.eigh(ovlp)
            self.ao2loc = np.dot(np.dot(ovlp_vecs, np.diag(np.power(ovlp_eigs, -0.5))), ovlp_vecs.T)
            self.TI_OK  = False
        if self._which == 'iao':
            assert self.Norbs == self.mol.nao_nr(), "iao requires full active space"
            self.ao2loc = iao_helper.localize_iao(self.mol, the_mf)
            if ao_rotation is not None:
                self.ao2loc = np.dot(self.ao2loc, ao_rotation.T)
            self.TI_OK = False
        assert self.loc_ortho() < 1e-8, "LMO basis is not orthonormal"

        # Spin potential (F_alpha - F_beta)/2 in LMO basis; None for RHF.
        # Computed here, after ao2loc is built.
        self.activeVSPIN = self._compute_spin_oei(the_mf)

        # Frozen-core effective Hamiltonian (core contribution to oei)
        if _mo_occ.ndim == 2:  # UHF: frozen core already blocked above, so frozenDM = 0
            self.frozenDMao = np.zeros_like(self.fullDMao)
            self.frozenJKao = np.zeros_like(self.fullJKao)
        else:
            self.frozenDMmo  = _mo_occ.copy()
            self.frozenDMmo[self.active == 1] = 0
            self.frozenDMao  = the_mf.mo_coeff @ np.diag(self.frozenDMmo) @ the_mf.mo_coeff.T
            _v_frozen        = the_mf.get_veff(self.mol, self.frozenDMao)
            self.frozenJKao  = _v_frozen[0] if _v_frozen.ndim == 3 else _v_frozen
        self.frozenOEIao = self.fullFOCKao - self.fullJKao + self.frozenJKao

        # Active-space integrals in LMO basis
        self.activeCONST = the_mf.energy_nuc() + np.einsum('ij,ij->', self.frozenOEIao - 0.5 * self.frozenJKao, self.frozenDMao)
        self.activeOEI   = np.dot(np.dot(self.ao2loc.T, self.frozenOEIao), self.ao2loc)
        self.activeFOCK  = np.dot(np.dot(self.ao2loc.T, self.fullFOCKao), self.ao2loc)
        if self.Norbs <= 150:
            self.ERIinMEM  = True
            self.activeERI = ao2mo.outcore.full_iofree(self.mol, self.ao2loc, compact=False).reshape(
                self.Norbs, self.Norbs, self.Norbs, self.Norbs)
        else:
            self.ERIinMEM  = False
            self.activeERI = None

    def _compute_spin_oei(self, the_mf):
        """
        Compute the spin-dependent oei in the LMO basis: oei_s = (F_alpha - F_beta) / 2.
        Returns None for closed-shell (RHF) references.
        """
        from pyscf import scf as pyscf_scf
        is_uhf  = isinstance(the_mf, pyscf_scf.uhf.UHF)
        is_rohf = isinstance(the_mf, pyscf_scf.rohf.ROHF)
        if not (is_uhf or is_rohf):
            return None

        if is_uhf:
            dm_ao = the_mf.make_rdm1()
            dm_a, dm_b = dm_ao[0], dm_ao[1]
        else:  # ROHF: reconstruct alpha/beta DMs from mo_occ
            mo  = the_mf.mo_coeff
            occ = the_mf.mo_occ
            dm_a = np.dot(mo * (occ > 0),  mo.T)
            dm_b = np.dot(mo * (occ == 2), mo.T)

        hcore  = the_mf.get_hcore()
        veff   = the_mf.get_veff(self.mol, np.stack([dm_a, dm_b]))
        fock_a = hcore + veff[0]
        fock_b = hcore + veff[1]

        spin_loc = self.ao2loc.T @ (0.5 * (fock_a - fock_b)) @ self.ao2loc
        print(f"localintegrals: open-shell reference detected; ||oei_s||_F = {np.linalg.norm(spin_loc):.6f}")
        return spin_loc

    def molden(self, filename):
        """Write LMOs to a Molden file."""
        with open(filename, 'w') as thefile:
            molden.header(self.mol, thefile)
            molden.orbital_coeff(self.mol, thefile, self.ao2loc)

    def loc_ortho(self):
        """Return the Frobenius norm of (ao2loc^T S ao2loc - I); should be ~0."""
        S = self.mol.intor('cint1e_ovlp_sph')
        return np.linalg.norm(np.dot(np.dot(self.ao2loc.T, S), self.ao2loc) - np.eye(self.Norbs))

    def debug_matrixelements(self):
        """Verify that the localized Hamiltonian reproduces the RHF energy (debug utility)."""
        eigvals, eigvecs = np.linalg.eigh(self.activeFOCK)
        eigvecs = eigvecs[:, eigvals.argsort()]
        assert self.Nelec % 2 == 0
        numPairs = self.Nelec // 2
        dm_guess  = 2 * np.dot(eigvecs[:, :numPairs], eigvecs[:, :numPairs].T)
        if self.ERIinMEM:
            dm_loc = rhf.solve_ERI(self.activeOEI, self.activeERI, dm_guess, numPairs)
        else:
            dm_loc = rhf.solve_JK(self.activeOEI, self.mol, self.ao2loc, dm_guess, numPairs)
        newFOCKloc = self.loc_fock(dm_loc)
        newRHFener = self.activeCONST + 0.5 * np.einsum('ij,ij->', dm_loc, self.activeOEI + newFOCKloc)
        print("||RDM(fock) - RDM(oei,ERI)||  =", np.linalg.norm(dm_guess - dm_loc))
        print("||fock - fock(RDM(oei,ERI))|| =", np.linalg.norm(self.activeFOCK - newFOCKloc))
        print("RHF energy (MF input)     =", self.fullEhf)
        print("RHF energy (oei+ERI)      =", newRHFener)

    # ── Accessors ──────────────────────────────────────────────────────────

    def const(self):
        """Return the nuclear repulsion + frozen-core energy constant."""
        return self.activeCONST

    def loc_oei(self):
        """Return the active oei in the LMO basis."""
        return self.activeOEI

    def loc_fock(self, dm_loc=None):
        """
        Return the Fock matrix in the LMO basis.
        If dm_loc is None, returns the mean-field Fock; otherwise recomputes
        the Fock from the given 1-RDM (used during self-consistency).
        """
        if dm_loc is None:
            return self.activeFOCK
        if not self.ERIinMEM:
            DM_ao  = np.dot(np.dot(self.ao2loc, dm_loc), self.ao2loc.T)
            _v_ao  = self.the_mf.get_veff(self.mol, DM_ao)
            JK_ao  = _v_ao[0] if _v_ao.ndim == 3 else _v_ao
            JK_loc = np.dot(np.dot(self.ao2loc.T, JK_ao), self.ao2loc)
        else:
            JK_loc = (np.einsum('ijkl,ij->kl', self.activeERI, dm_loc)
                      - 0.5 * np.einsum('ijkl,ik->jl', self.activeERI, dm_loc))
        return self.activeOEI + JK_loc

    def loc_tei(self):
        """Return the active 2e integrals (tei) in the LMO basis (requires ERIinMEM=True)."""
        assert self.ERIinMEM, "local_integrals::loc_tei: ERIs not stored in memory."
        return self.activeERI

    def loc_spin_oei(self):
        """Return the localized spin oei (F_alpha - F_beta)/2, or None for RHF."""
        return self.activeVSPIN

    # ── dmet embedding integral projectors ────────────────────────────────

    def dmet_oei(self, loc_2_dmet, numActive):
        """Project oei into the dmet embedding space of size numActive."""
        return np.dot(np.dot(loc_2_dmet[:, :numActive].T, self.activeOEI), loc_2_dmet[:, :numActive])

    def dmet_oei_s(self, loc_2_dmet, numActive):
        """Project spin oei into the dmet embedding space; returns None for RHF."""
        if self.activeVSPIN is None:
            return None
        return loc_2_dmet[:, :numActive].T @ self.activeVSPIN @ loc_2_dmet[:, :numActive]

    def dmet_fock(self, loc_2_dmet, numActive, coreDMloc):
        """Project the Fock matrix (computed from coreDMloc) into the dmet embedding space."""
        return np.dot(np.dot(loc_2_dmet[:, :numActive].T, self.loc_fock(coreDMloc)), loc_2_dmet[:, :numActive])

    def dmet_init_guess_rhf(self, loc_2_dmet, numActive, numPairs, nimp, chempot_imp):
        """Generate an RHF initial density guess in the dmet embedding space."""
        Fock_emb = np.dot(np.dot(loc_2_dmet[:, :numActive].T, self.activeFOCK), loc_2_dmet[:, :numActive])
        if chempot_imp != 0.0:
            for orb in range(nimp):
                Fock_emb[orb, orb] -= chempot_imp
        eigvals, eigvecs = np.linalg.eigh(Fock_emb)
        eigvecs = eigvecs[:, eigvals.argsort()]
        return 2 * np.dot(eigvecs[:, :numPairs], eigvecs[:, :numPairs].T)

    def dmet_tei(self, loc_2_dmet, numAct):
        """Transform tei into the dmet embedding space of size numAct."""
        if not self.ERIinMEM:
            transfo = np.dot(self.ao2loc, loc_2_dmet[:, :numAct])
            return ao2mo.outcore.full_iofree(self.mol, transfo, compact=False).reshape(
                numAct, numAct, numAct, numAct)
        return ao2mo.incore.full(
            ao2mo.restore(8, self.activeERI, self.Norbs), loc_2_dmet[:, :numAct], compact=False
        ).reshape(numAct, numAct, numAct, numAct)
