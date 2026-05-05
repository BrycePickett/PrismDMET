"""
Per-fragment embedding Hamiltonian builder for PrismDMET.

Given a fragment index and the current mean-field 1-RDM, FragmentBuilder.build()
performs the Schmidt decomposition, environment occupation clamping, integral
projection into the DMET basis, and density matrix initial guess construction.

It does not call any solver and does not know about parallelism. The result
dict it returns is plain Python and numpy, suitable for SolverFactory or a
parallel worker.
"""

import numpy as np


class FragmentBuilder:
    """
    Builds the per-fragment embedding Hamiltonian from the mean-field 1-RDM.

    Parameters
    ----------
    ints : local_integrals.localintegrals
        Localized integral object for the full system.
    helper : prismdmethelper.prismdmethelper
        Helper providing constructbath and construct1RDM_loc.
    impClust : list of ndarray
        Impurity cluster masks, one per fragment. Negative values signal
        the flag_rhf sentinel.
    method : str
        Active solver method name (e.g. 'CASSCF', 'CC', 'ED').
    BATH_ORBS : list of int or None
        Per-fragment bath size overrides. If None, bath size equals impurity size.
    NI_hack : bool
        If True, zero out TEI bath-bath and bath-impurity blocks.
    umat : ndarray (Norb, Norb)
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

    def build(self, counter, OneRDM, chempot_imp):
        """
        Construct the full embedding Hamiltonian for fragment counter.

        Parameters
        ----------
        counter : int
            Zero-based fragment index into impClust.
        OneRDM : ndarray (Norb, Norb)
            Mean-field 1-RDM in the LMO basis.
        chempot_imp : float
            Chemical potential applied to the impurity block.

        Returns
        -------
        dict
            Keys: counter, flag_rhf, impurityOrbs, numImpOrbs, Norb_in_imp,
            Nelec_in_imp, loc2dmet, core1RDM_loc, core1RDM_dmet, dmetOEI,
            dmetFOCK, dmetTEI, DMguessRHF, method_key.

        Raises
        ------
        RuntimeError
            If environment occupation clamping fails.
        """
        flag_rhf     = np.sum(self._impClust[counter]) < 0
        impurityOrbs = np.abs(self._impClust[counter])
        numImpOrbs   = int(np.sum(impurityOrbs))
        bath_request = numImpOrbs if self._BATH_ORBS is None else self._BATH_ORBS[counter]

        numBathOrbs, loc2dmet, core1RDM_dmet = self._helper.constructbath(
            OneRDM, impurityOrbs, bath_request, threshold=self._bath_tol)

        core_cutoff = 0.01 if self._BATH_ORBS is None else 0.5
        for idx in range(len(core1RDM_dmet)):
            occ = core1RDM_dmet[idx]
            if occ < core_cutoff:
                core1RDM_dmet[idx] = 0.0
            elif occ > 2.0 - core_cutoff:
                core1RDM_dmet[idx] = 2.0
            else:
                raise RuntimeError(
                    f"FragmentBuilder.build: fragment {counter}, environment orbital {idx} "
                    f"has occupation {occ:.6f}, which is neither near 0 nor 2 "
                    f"(cutoff={core_cutoff}). The bath size may be too small."
                )

        Norb_in_imp  = numImpOrbs + numBathOrbs
        Nelec_in_imp = int(round(self._ints.Nelec - np.sum(core1RDM_dmet)))
        core1RDM_loc = np.dot(
            np.dot(loc2dmet, np.diag(core1RDM_dmet)), loc2dmet.T)

        assert Norb_in_imp <= self._ints.Norbs, (
            f"FragmentBuilder: Norb_in_imp={Norb_in_imp} exceeds Norbs={self._ints.Norbs}")

        dmetOEI  = self._ints.dmet_oei( loc2dmet, Norb_in_imp)
        dmetFOCK = self._ints.dmet_fock(loc2dmet, Norb_in_imp, core1RDM_loc)
        dmetTEI  = self._ints.dmet_tei( loc2dmet, Norb_in_imp)

        if self._NI_hack:
            dmetTEI[:, :, :, numImpOrbs:] = 0.0
            dmetTEI[:, :, numImpOrbs:, :] = 0.0
            dmetTEI[:, numImpOrbs:, :, :] = 0.0
            dmetTEI[numImpOrbs:, :, :, :] = 0.0
            umat_rotated = np.dot(np.dot(loc2dmet.T, self._umat), loc2dmet)
            umat_rotated[:numImpOrbs, :numImpOrbs] = 0.0
            dmetOEI  += umat_rotated[:Norb_in_imp, :Norb_in_imp]
            dmetFOCK  = np.array(dmetOEI, copy=True)

        needs_dm   = flag_rhf or self._method in self._NEEDS_DM_METHODS
        DMguessRHF = None
        if needs_dm:
            DMguessRHF = self._ints.dmet_init_guess_rhf(
                loc2dmet, Norb_in_imp, Nelec_in_imp // 2,
                numImpOrbs, chempot_imp)

        method_key = 'flag_rhf' if flag_rhf else self._method

        return {
            'counter'       : counter,
            'flag_rhf'      : flag_rhf,
            'impurityOrbs'  : impurityOrbs,
            'numImpOrbs'    : numImpOrbs,
            'Norb_in_imp'   : Norb_in_imp,
            'Nelec_in_imp'  : Nelec_in_imp,
            'loc2dmet'      : loc2dmet,
            'core1RDM_loc'  : core1RDM_loc,
            'core1RDM_dmet' : core1RDM_dmet,
            'dmetOEI'       : dmetOEI,
            'dmetFOCK'      : dmetFOCK,
            'dmetTEI'       : dmetTEI,
            'DMguessRHF'    : DMguessRHF,
            'method_key'    : method_key,
        }

    def build_symmetry_bath(self, counter, OneRDM, sym_parent):
        """
        Build bath orbitals and loc2dmet for a symmetry-copied fragment.

        Runs constructbath to populate dmetOrbs, but does not project any
        integrals. The result and energy are copied from the parent fragment.

        Parameters
        ----------
        counter : int
            Fragment index of the copy.
        OneRDM : ndarray (Norb, Norb)
            Current mean-field 1-RDM in the LMO basis.
        sym_parent : int
            Fragment index of the parent whose result will be reused.

        Returns
        -------
        dict with keys: counter, sym_parent, impurityOrbs, Norb_in_imp, loc2dmet.
        """
        impurityOrbs = np.abs(self._impClust[counter])
        numImpOrbs   = int(np.sum(impurityOrbs))
        bath_request = numImpOrbs if self._BATH_ORBS is None else self._BATH_ORBS[counter]
        numBathOrbs, loc2dmet, _ = self._helper.constructbath(
            OneRDM, impurityOrbs, bath_request, threshold=self._bath_tol)
        Norb_in_imp = numImpOrbs + numBathOrbs
        return {
            'counter'      : counter,
            'sym_parent'   : sym_parent,
            'impurityOrbs' : impurityOrbs,
            'Norb_in_imp'  : Norb_in_imp,
            'loc2dmet'     : loc2dmet,
        }
