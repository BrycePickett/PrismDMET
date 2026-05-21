"""C-extension interface and core DMET linear-algebra routines."""

from . import rhf
import numpy as np
import ctypes
import os

_base_dir    = os.path.dirname(os.path.abspath(__file__))
_so_path     = os.path.join(_base_dir, '..', 'lib', 'libprismdmet.so')
lib_prismdmet = ctypes.CDLL(os.path.abspath(_so_path))


class PrismDMETHelper:

    def __init__(self, locints, list_H1, use_constrained_opt, minFunc):
        self.locints  = locints
        self._is_open_shell = (self.locints.Nelec % 2 != 0)
        self.numPairs = self.locints.Nelec // 2
        
        self.num_alpha = (self.locints.Nelec + self.locints.mol.spin) // 2
        self.num_beta  = (self.locints.Nelec - self.locints.mol.spin) // 2
        self.altcf    = use_constrained_opt
        self.minFunc  = None

        if self.altcf:
            _mf = minFunc.upper() if minFunc else minFunc
            assert _mf in ('OEI', 'FOCK_INIT'), \
                f"PrismDMETHelper: minFunc must be 'OEI' or 'FOCK_INIT', got '{minFunc}'"
            self.minFunc = _mf

        self.list_H1 = list_H1
        self.H1start, self.H1row, self.H1col = self._convert_H1_sparse()
        self.Nterms  = len(self.H1start) - 1

    def _convert_H1_sparse(self):
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
        if self._is_open_shell and doSCF:
            raise NotImplementedError("Open-shell DMET iterative SCF loops are not yet supported.")
        if self.altcf and self.minFunc == 'OEI':
            oei = self.locints.loc_oei() + umat_loc
        else:
            oei = self.locints.loc_fock() + umat_loc
        dm_loc = self._build_1rdm(oei, self.numPairs)
        if doSCF:
            if self.locints.ERIinMEM:
                dm_loc = rhf.solve_ERI(self.locints.loc_oei() + umat_loc, self.locints.loc_tei(), dm_loc, self.numPairs)
            else:
                dm_loc = rhf.solve_JK(self.locints.loc_oei() + umat_loc, self.locints.mol, self.locints.ao2loc, dm_loc, self.numPairs)
        return dm_loc

    def construct1RDM_response(self, doSCF, umat_loc, NOrotation):
        if self._is_open_shell:
            raise NotImplementedError("DMET chemical-potential optimization requires an even electron count.")
        oei = self.locints.loc_fock() + umat_loc
        if doSCF:
            dm_loc = self._build_1rdm(oei, self.numPairs)
            if self.locints.ERIinMEM:
                dm_loc = rhf.solve_ERI(self.locints.loc_oei() + umat_loc, self.locints.loc_tei(), dm_loc, self.numPairs)
            else:
                dm_loc = rhf.solve_JK(self.locints.loc_oei() + umat_loc, self.locints.mol, self.locints.ao2loc, dm_loc, self.numPairs)
            oei = self.locints.loc_fock(dm_loc) + umat_loc

        if NOrotation is not None:
            oei = np.dot(np.dot(NOrotation.T, oei), NOrotation)
        oei_flat = np.array(oei.reshape(self.locints.Norbs ** 2), dtype=ctypes.c_double)

        rdm_deriv = np.ones([self.locints.Norbs * self.locints.Norbs * self.Nterms], dtype=ctypes.c_double)
        lib_prismdmet.rhf_response(
            ctypes.c_int(self.locints.Norbs),
            ctypes.c_int(self.Nterms),
            ctypes.c_int(self.numPairs),
            self.H1start.ctypes.data_as(ctypes.c_void_p),
            self.H1row.ctypes.data_as(ctypes.c_void_p),
            self.H1col.ctypes.data_as(ctypes.c_void_p),
            oei_flat.ctypes.data_as(ctypes.c_void_p),
            rdm_deriv.ctypes.data_as(ctypes.c_void_p)
        )
        return rdm_deriv.reshape((self.Nterms, self.locints.Norbs, self.locints.Norbs), order='C')

    def _build_1rdm(self, oei, numPairs):
        eigenvals, eigenvecs = np.linalg.eigh(oei)
        idx = eigenvals.argsort()
        
        if self._is_open_shell:
            dm_a = np.dot(eigenvecs[:, idx[:self.num_alpha]], eigenvecs[:, idx[:self.num_alpha]].T)
            dm_b = np.dot(eigenvecs[:, idx[:self.num_beta]],  eigenvecs[:, idx[:self.num_beta]].T)
            return dm_a + dm_b
        else:
            return 2 * np.dot(eigenvecs[:, idx[:numPairs]], eigenvecs[:, idx[:numPairs]].T)

    def constructbath(self, OneDM, impurity_orbs, numBathOrbs, threshold=1e-13):
        embeddingOrbs  = np.array(1 - impurity_orbs, dtype=float)
        if embeddingOrbs.ndim == 1:
            embeddingOrbs = embeddingOrbs[:, np.newaxis]  # (Norbs, 1)
        isEmbedding    = np.dot(embeddingOrbs, embeddingOrbs.T) == 1
        numEmbedOrbs   = int(np.sum(embeddingOrbs))
        embedding1RDM  = np.reshape(OneDM[isEmbedding], (numEmbedOrbs, numEmbedOrbs))

        num_imp_orbs   = int(np.sum(impurity_orbs))
        numTotalOrbs = len(impurity_orbs)

        eigenvals, eigenvecs = np.linalg.eigh(embedding1RDM)
        idx    = np.maximum(-eigenvals, eigenvals - 2.0).argsort()
        tokeep = np.sum(-np.maximum(-eigenvals, eigenvals - 2.0)[idx] > threshold)
        if tokeep < numBathOrbs:
            print(f"dmet::constructbath : Throwing out {numBathOrbs - tokeep} orbitals "
                  f"within {threshold} of 0 or 2.")
        numBathOrbs = min(int(tokeep), numBathOrbs)

        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        pureEnvVals = -eigenvals[numBathOrbs:]
        pureEnvVecs = eigenvecs[:, numBathOrbs:]
        env_idx = pureEnvVals.argsort()
        eigenvecs[:, numBathOrbs:] = pureEnvVecs[:, env_idx]
        pureEnvVals = -pureEnvVals[env_idx]
        coreOccupations = np.hstack((np.zeros([num_imp_orbs + numBathOrbs]), pureEnvVals))

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
