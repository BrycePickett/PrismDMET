"""
Prismdmet helper: C-extension interface and core dmet linear-algebra routines.
Wraps libprismdmet.so (C) for the RHF response (1-RDM derivative w.r.t. u-matrix).
"""

import local_integrals
import rhf
import numpy as np
import ctypes
import os

_base_dir    = os.path.dirname(os.path.abspath(__file__))
_so_path     = os.path.join(_base_dir, '..', 'lib', 'libprismdmet.so')
lib_prismdmet = ctypes.CDLL(os.path.abspath(_so_path))


class PrismdmetHelper:
    """
    Provides the bath construction and 1-RDM routines needed by the dmet loop.

    Parameters
    ----------
    locints : local_integrals.local_integrals
        The localized integral object for the full system.
    list_H1 : list of ndarray
        The H1 basis matrices that parametrize the correlation potential (u-matrix).
    use_constrained_opt : bool
        If True, use the constrained cost function (OEI or FOCK_INIT minimization).
    minFunc : str or None
        Which quantity to minimize: 'OEI', 'FOCK_INIT', or None (standard dmet).
    """

    def __init__(self, locints, list_H1, use_constrained_opt, minFunc):
        self.locints  = locints
        assert self.locints.Nelec % 2 == 0, "Nelec must be even (closed-shell reference)"
        self.numPairs = self.locints.Nelec // 2
        self.altcf    = use_constrained_opt
        self.minFunc  = None

        if self.altcf:
            assert minFunc in ('OEI', 'FOCK_INIT')
            self.minFunc = minFunc

        # Sparse representation of the H1 basis for the C-level gradient
        self.list_H1 = list_H1
        self.H1start, self.H1row, self.H1col = self._convert_H1_sparse()
        self.Nterms  = len(self.H1start) - 1

    def _convert_H1_sparse(self):
        """Convert the H1 list to CSR-like arrays for the C gradient routine."""
        H1start, H1row, H1col = [0], [], []
        total = 0
        for mat in self.list_H1:
            rows, cols = np.where(mat == 1)
            total += len(rows)
            H1start.append(total)
            H1row.extend(rows)
            H1col.extend(cols)
        return (np.array(H1start, dtype=ctypes.c_int),
                np.array(H1row,   dtype=ctypes.c_int),
                np.array(H1col,   dtype=ctypes.c_int))

    def construct1RDM_loc(self, doSCF, umat_loc):
        """
        Construct the mean-field 1-RDM in the LMO basis given the u-matrix.

        Parameters
        ----------
        doSCF : bool
            If True, perform a full SCF cycle on top of the initial guess.
        umat_loc : ndarray (Norbs, Norbs)
            The correlation potential in the LMO basis.
        """
        if self.altcf and self.minFunc == 'OEI':
            OEI = self.locints.loc_oei() + umat_loc
        else:
            OEI = self.locints.loc_fock() + umat_loc
        dm_loc = self._build_1rdm(OEI, self.numPairs)
        if doSCF:
            if self.locints.ERIinMEM:
                dm_loc = rhf.solve_ERI(self.locints.loc_oei() + umat_loc, self.locints.loc_tei(), dm_loc, self.numPairs)
            else:
                dm_loc = rhf.solve_JK(self.locints.loc_oei() + umat_loc, self.locints.mol, self.locints.ao2loc, dm_loc, self.numPairs)
        return dm_loc

    def construct1RDM_response(self, doSCF, umat_loc, NOrotation):
        """
        Compute the 1-RDM derivative dγ/du via the C-level RHF response function.
        Used to build the analytical gradient of the cost function.
        """
        OEI = self.locints.loc_fock() + umat_loc
        if doSCF:
            dm_loc = self._build_1rdm(OEI, self.numPairs)
            if self.locints.ERIinMEM:
                dm_loc = rhf.solve_ERI(self.locints.loc_oei() + umat_loc, self.locints.loc_tei(), dm_loc, self.numPairs)
            else:
                dm_loc = rhf.solve_JK(self.locints.loc_oei() + umat_loc, self.locints.mol, self.locints.ao2loc, dm_loc, self.numPairs)
            OEI = self.locints.loc_fock(dm_loc) + umat_loc

        if NOrotation is not None:
            OEI = np.dot(np.dot(NOrotation.T, OEI), NOrotation)
        OEI_flat = np.array(OEI.reshape(self.locints.Norbs ** 2), dtype=ctypes.c_double)

        rdm_deriv = np.ones([self.locints.Norbs * self.locints.Norbs * self.Nterms], dtype=ctypes.c_double)
        lib_prismdmet.rhf_response(
            ctypes.c_int(self.locints.Norbs),
            ctypes.c_int(self.Nterms),
            ctypes.c_int(self.numPairs),
            self.H1start.ctypes.data_as(ctypes.c_void_p),
            self.H1row.ctypes.data_as(ctypes.c_void_p),
            self.H1col.ctypes.data_as(ctypes.c_void_p),
            OEI_flat.ctypes.data_as(ctypes.c_void_p),
            rdm_deriv.ctypes.data_as(ctypes.c_void_p)
        )
        return rdm_deriv.reshape((self.Nterms, self.locints.Norbs, self.locints.Norbs), order='C')

    def _build_1rdm(self, OEI, numPairs):
        """Build the idempotent 1-RDM by occupying the lowest numPairs eigenstates of OEI."""
        eigenvals, eigenvecs = np.linalg.eigh(OEI)
        idx = eigenvals.argsort()
        return 2 * np.dot(eigenvecs[:, idx[:numPairs]], eigenvecs[:, idx[:numPairs]].T)

    def constructbath(self, OneDM, impurity_orbs, numBathOrbs, threshold=1e-13):
        """
        Perform the Schmidt decomposition to find the dmet bath orbitals.

        Returns
        -------
        numBathOrbs : int
            Number of bath orbitals retained (after threshold truncation).
        loc_2_dmet : ndarray (Norbs, Norbs)
            Rotation from LMO to dmet basis. Columns ordered as:
            [impurity | bath | environment].
        coreOccupations : ndarray
            Occupation numbers of the environment (core) orbitals.
        """
        embeddingOrbs  = np.matrix(1 - impurity_orbs)
        if embeddingOrbs.shape[0] > 1:
            embeddingOrbs = embeddingOrbs.T
        isEmbedding    = np.dot(embeddingOrbs.T, embeddingOrbs) == 1
        numEmbedOrbs   = int(np.sum(embeddingOrbs))
        embedding1RDM  = np.reshape(OneDM[isEmbedding], (numEmbedOrbs, numEmbedOrbs))

        num_imp_orbs   = int(np.sum(impurity_orbs))
        numTotalOrbs = len(impurity_orbs)

        eigenvals, eigenvecs = np.linalg.eigh(embedding1RDM)
        # Sort by entanglement: occupations closest to 1 first
        idx    = np.maximum(-eigenvals, eigenvals - 2.0).argsort()
        tokeep = np.sum(-np.maximum(-eigenvals, eigenvals - 2.0)[idx] > threshold)
        if tokeep < numBathOrbs:
            print(f"dmet::constructbath : Throwing out {numBathOrbs - tokeep} orbitals "
                  f"within {threshold} of 0 or 2.")
        numBathOrbs = min(int(tokeep), numBathOrbs)

        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        # Separate environment (core) from bath
        pureEnvVals = -eigenvals[numBathOrbs:]
        pureEnvVecs = eigenvecs[:, numBathOrbs:]
        env_idx = pureEnvVals.argsort()
        eigenvecs[:, numBathOrbs:] = pureEnvVecs[:, env_idx]
        pureEnvVals = -pureEnvVals[env_idx]
        coreOccupations = np.hstack((np.zeros([num_imp_orbs + numBathOrbs]), pureEnvVals))

        # Insert impurity identity columns/rows
        for counter in range(num_imp_orbs):
            eigenvecs = np.insert(eigenvecs, counter, 0.0, axis=1)
        counter = 0
        for counter2 in range(numTotalOrbs):
            if impurity_orbs[counter2]:
                eigenvecs = np.insert(eigenvecs, counter2, 0.0, axis=0)
                eigenvecs[counter2, counter] = 1.0
                counter += 1
        assert counter == num_imp_orbs

        assert np.linalg.norm(np.dot(eigenvecs.T, eigenvecs) - np.identity(numTotalOrbs)) < 1e-12
        return numBathOrbs, eigenvecs, coreOccupations
