"""
Per-fragment embedding Hamiltonian builder for Prismdmet.

Given a fragment index and the current mean-field 1-RDM, fragment_builder.build()
performs the Schmidt decomposition, environment occupation clamping, integral
projection into the dmet basis, and density matrix initial guess construction.

It does not call any solver and does not know about parallelism. The result
dict it returns is plain Python and numpy, suitable for solver_dispatcher or a
parallel worker.
"""

import numpy as np


class fragment_builder:
    """
    Builds the per-fragment embedding Hamiltonian from the mean-field 1-RDM.

    Parameters
    ----------
    ints : local_integrals.local_integrals
        Localized integral object for the full system.
    helper : prismdmet_helper.PrismdmetHelper
        Helper providing constructbath and construct1RDM_loc.
    impClust : list of ndarray
        Impurity cluster masks, one per fragment. Negative values signal
        the flag_rhf sentinel.
    method : str
        Active solver method name (e.g. 'CASSCF', 'CC', 'ED').
    BATH_ORBS : list of int or None
        Per-fragment bath size overrides. If None, bath size equals impurity size.
    NI_hack : bool
        If True, zero out tei bath-bath and bath-impurity blocks.
    umat : ndarray (norb, norb)
        Current correlation potential in the LMO basis.
    bath_tol : float
        Threshold for discarding unentangled bath orbitals.
    """

    _NEEDS_DM_METHODS = frozenset({'CC', 'MP2', 'EOM-CC', 'CASSCF'})

    def __init__(self, ints, helper, impClust, method, BATH_ORBS, NI_hack, umat, bath_tol):
        self._ints      = ints
        self._helper    = helper
        self._impClust  = impClust
        self._method    = method
        self._BATH_ORBS = BATH_ORBS
        self._NI_hack   = NI_hack
        self._umat      = umat
        self._bath_tol  = bath_tol

    def build(self, counter, one_rdm, chempot_imp):
        """
        Construct the full embedding Hamiltonian for fragment counter.

        Parameters
        ----------
        counter : int
            Zero-based fragment index into impClust.
        one_rdm : ndarray (norb, norb)
            Mean-field 1-RDM in the LMO basis.
        chempot_imp : float
            Chemical potential applied to the impurity block.

        Returns
        -------
        dict
            Keys: counter, flag_rhf, impurity_orbs, num_imp_orbs, norb_in_imp,
            nelec_in_imp, loc_2_dmet, core_1rdm_loc, core_1rdm_dmet, dmet_oei,
            dmet_fock, dmet_tei, dm_guess_rhf, method_key.

        Raises
        ------
        RuntimeError
            If environment occupation clamping fails.
        """
        flag_rhf     = np.sum(self._impClust[counter]) < 0
        impurity_orbs = np.abs(self._impClust[counter])
        num_imp_orbs   = int(np.sum(impurity_orbs))
        bath_request = num_imp_orbs if self._BATH_ORBS is None else self._BATH_ORBS[counter]

        numBathOrbs, loc_2_dmet, core_1rdm_dmet = self._helper.constructbath(
            one_rdm, impurity_orbs, bath_request, threshold=self._bath_tol)

        core_cutoff = 0.01 if self._BATH_ORBS is None else 0.5
        for idx in range(len(core_1rdm_dmet)):
            occ = core_1rdm_dmet[idx]
            if occ < core_cutoff:
                core_1rdm_dmet[idx] = 0.0
            elif occ > 2.0 - core_cutoff:
                core_1rdm_dmet[idx] = 2.0
            else:
                raise RuntimeError(
                    f"fragment_builder.build: fragment {counter}, environment orbital {idx} "
                    f"has occupation {occ:.6f}, which is neither near 0 nor 2 "
                    f"(cutoff={core_cutoff}). The bath size may be too small."
                )

        norb_in_imp  = num_imp_orbs + numBathOrbs
        nelec_in_imp = int(round(self._ints.Nelec - np.sum(core_1rdm_dmet)))
        core_1rdm_loc = np.dot(
            np.dot(loc_2_dmet, np.diag(core_1rdm_dmet)), loc_2_dmet.T)

        assert norb_in_imp <= self._ints.Norbs, (
            f"fragment_builder: norb_in_imp={norb_in_imp} exceeds Norbs={self._ints.Norbs}")

        dmet_oei  = self._ints.dmet_oei( loc_2_dmet, norb_in_imp)
        dmet_fock = self._ints.dmet_fock(loc_2_dmet, norb_in_imp, core_1rdm_loc)
        dmet_tei  = self._ints.dmet_tei( loc_2_dmet, norb_in_imp)

        if self._NI_hack:
            dmet_tei[:, :, :, num_imp_orbs:] = 0.0
            dmet_tei[:, :, num_imp_orbs:, :] = 0.0
            dmet_tei[:, num_imp_orbs:, :, :] = 0.0
            dmet_tei[num_imp_orbs:, :, :, :] = 0.0
            umat_rotated = np.dot(np.dot(loc_2_dmet.T, self._umat), loc_2_dmet)
            umat_rotated[:num_imp_orbs, :num_imp_orbs] = 0.0
            dmet_oei  += umat_rotated[:norb_in_imp, :norb_in_imp]
            dmet_fock  = np.array(dmet_oei, copy=True)

        needs_dm   = flag_rhf or self._method in self._NEEDS_DM_METHODS
        dm_guess_rhf = None
        if needs_dm:
            dm_guess_rhf = self._ints.dmet_init_guess_rhf(
                loc_2_dmet, norb_in_imp, nelec_in_imp // 2,
                num_imp_orbs, chempot_imp)

        method_key = 'flag_rhf' if flag_rhf else self._method

        return {
            'counter'       : counter,
            'flag_rhf'      : flag_rhf,
            'impurity_orbs'  : impurity_orbs,
            'num_imp_orbs'    : num_imp_orbs,
            'norb_in_imp'   : norb_in_imp,
            'nelec_in_imp'  : nelec_in_imp,
            'loc_2_dmet'      : loc_2_dmet,
            'core_1rdm_loc'  : core_1rdm_loc,
            'core_1rdm_dmet' : core_1rdm_dmet,
            'dmet_oei'       : dmet_oei,
            'dmet_fock'      : dmet_fock,
            'dmet_tei'       : dmet_tei,
            'dm_guess_rhf'    : dm_guess_rhf,
            'method_key'    : method_key,
        }

    def build_symmetry_bath(self, counter, one_rdm, sym_parent):
        """
        Build bath orbitals and loc_2_dmet for a symmetry-copied fragment.

        Runs constructbath to populate dmetOrbs, but does not project any
        integrals. The result and energy are copied from the parent fragment.

        Parameters
        ----------
        counter : int
            Fragment index of the copy.
        one_rdm : ndarray (norb, norb)
            Current mean-field 1-RDM in the LMO basis.
        sym_parent : int
            Fragment index of the parent whose result will be reused.

        Returns
        -------
        dict with keys: counter, sym_parent, impurity_orbs, norb_in_imp, loc_2_dmet.
        """
        impurity_orbs = np.abs(self._impClust[counter])
        num_imp_orbs   = int(np.sum(impurity_orbs))
        bath_request = num_imp_orbs if self._BATH_ORBS is None else self._BATH_ORBS[counter]
        numBathOrbs, loc_2_dmet, _ = self._helper.constructbath(
            one_rdm, impurity_orbs, bath_request, threshold=self._bath_tol)
        norb_in_imp = num_imp_orbs + numBathOrbs
        return {
            'counter'      : counter,
            'sym_parent'   : sym_parent,
            'impurity_orbs' : impurity_orbs,
            'norb_in_imp'  : norb_in_imp,
            'loc_2_dmet'     : loc_2_dmet,
        }
