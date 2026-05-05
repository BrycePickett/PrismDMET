"""
Geometry-aware utility for defect detection and QM cluster selection
from a pair of XYZ coordinate files (pristine + defect supercell).

This module handles defect detection and cluster index selection only.
QM/MM background charge setup is a planned future extension.

Usage:
    builder = defect_builder.from_xyz('pristine.xyz', 'defect.xyz')
    center, defect_type = builder.detect_center()
    qm_indices = builder.get_qm_region(radius=5.0)
    impurity_clusters = make_fragments(mol, myInts, [qm_indices])

Assumptions:
    - XYZ files use the same atom ordering for all non-defect atoms.
    - Vacancy is detected as a pristine atom with no close match in the
      defect structure (tolerance match_tol, default 0.3 Angstrom).
    - No periodic boundary conditions are applied.
    - Requires numpy only.
"""

import numpy as np


class defect_builder:
    """
    Automated defect detection and QM cluster selection from XYZ files.

    Parameters
    ----------
    pristine_coords : ndarray, shape (N, 3)
        Cartesian coordinates in Angstroms for the pristine supercell.
    pristine_species : list of str
        Element symbols for each atom in the pristine structure.
    defect_coords : ndarray, shape (M, 3)
        Cartesian coordinates in Angstroms for the defect supercell.
    defect_species : list of str
        Element symbols for each atom in the defect structure.
    """

    def __init__(self, pristine_coords, pristine_species,
                 defect_coords, defect_species):
        self._pristine_coords   = np.asarray(pristine_coords, dtype=float)
        self._pristine_species  = list(pristine_species)
        self._defect_coords     = np.asarray(defect_coords, dtype=float)
        self._defect_species    = list(defect_species)
        self._center            = None
        self._defect_type       = None

    @classmethod
    def from_xyz(cls, pristine_path, defect_path):
        """
        Construct a DefectTopologist from two standard XYZ files.

        Parameters
        ----------
        pristine_path : str
            Path to the XYZ file of the pristine reference supercell.
        defect_path : str
            Path to the XYZ file of the defect supercell.

        Returns
        -------
        DefectTopologist

        Raises
        ------
        ValueError
            If either file cannot be parsed as a valid XYZ file.
        """
        pristine_species, pristine_coords = cls._parse_xyz(pristine_path)
        defect_species, defect_coords     = cls._parse_xyz(defect_path)
        return cls(pristine_coords, pristine_species,
                   defect_coords, defect_species)

    def detect_center(self, match_tol=0.3):
        """
        Identify the defect center by comparing the pristine and defect
        coordinate sets.

        For each atom in the pristine structure, the minimum distance to
        any same-species atom in the defect structure is computed. If this
        distance exceeds match_tol, the atom has no counterpart and is the
        defect site.

        Defect types:
            vacancy      - atom in pristine, missing in defect.
            interstitial - atom in defect, absent in pristine.
            substitution - one site changed species between structures.

        Parameters
        ----------
        match_tol : float
            Distance threshold in Angstroms. Default: 0.3.

        Returns
        -------
        center : ndarray (3,)
            Cartesian coordinates of the defect center in Angstroms.
        defect_type : str
            One of 'vacancy', 'substitution', or 'interstitial'.

        Raises
        ------
        RuntimeError
            If no defect can be unambiguously identified.
        """
        unmatched_pristine = self._find_unmatched(
            self._pristine_coords, self._pristine_species,
            self._defect_coords,   self._defect_species,
            match_tol)

        unmatched_defect = self._find_unmatched(
            self._defect_coords,   self._defect_species,
            self._pristine_coords, self._pristine_species,
            match_tol)

        n_pris = len(unmatched_pristine)
        n_def  = len(unmatched_defect)

        if n_pris == 1 and n_def == 0:
            defect_type = 'vacancy'
            center = self._pristine_coords[unmatched_pristine[0]]
        elif n_pris == 0 and n_def == 1:
            defect_type = 'interstitial'
            center = self._defect_coords[unmatched_defect[0]]
        elif n_pris == 1 and n_def == 1:
            defect_type = 'substitution'
            center = 0.5 * (self._pristine_coords[unmatched_pristine[0]] +
                            self._defect_coords[unmatched_defect[0]])
        elif n_pris == 0 and n_def == 0:
            raise RuntimeError(
                "defect_builder.detect_center: no structural difference found. "
                "Are the pristine and defect files identical?"
            )
        else:
            raise RuntimeError(
                f"defect_builder.detect_center: ambiguous defect — "
                f"{n_pris} unmatched pristine atom(s) and "
                f"{n_def} unmatched defect atom(s). "
                f"This module handles single-point defects only. "
                f"Check match_tol={match_tol} or verify your input files."
            )

        self._center      = center
        self._defect_type = defect_type
        print(f"defect_builder: detected {defect_type} at {center} Angstrom")
        return center, defect_type

    def get_qm_region(self, radius):
        """
        Return atom indices from the defect structure within radius Angstroms
        of the detected defect center.

        Parameters
        ----------
        radius : float
            Cutoff radius in Angstroms.

        Returns
        -------
        indices : list of int
            Zero-based indices of atoms within radius of the defect center.
            Pass directly to dmet.make_fragments() as a single group.

        Raises
        ------
        RuntimeError
            If detect_center() has not been called first.
        ValueError
            If radius is not positive.
        """
        if self._center is None:
            raise RuntimeError(
                "defect_builder.get_qm_region: call detect_center() first."
            )
        if radius <= 0:
            raise ValueError(
                f"defect_builder.get_qm_region: radius must be positive, got {radius}."
            )

        distances = np.linalg.norm(self._defect_coords - self._center, axis=1)
        indices   = list(np.where(distances < radius)[0])
        print(f"defect_builder: {len(indices)} atoms within {radius} Angstrom of defect center.")
        return indices

    def neighbor_shells(self, n_shells=2):
        """
        Identify the first n_shells nearest-neighbor shells by finding
        natural gaps in the distance distribution from the defect center.

        Parameters
        ----------
        n_shells : int
            Number of distinct shells to identify. Default: 2.

        Returns
        -------
        shells : list of list of int
            Each inner list contains zero-based defect atom indices for that shell.
        shell_radii : list of float
            Outer radius in Angstroms for each shell.

        Raises
        ------
        RuntimeError
            If detect_center() has not been called first.
        """
        if self._center is None:
            raise RuntimeError(
                "defect_builder.neighbor_shells: call detect_center() first."
            )

        distances = np.linalg.norm(self._defect_coords - self._center, axis=1)
        unique_d  = np.sort(np.unique(np.round(distances, decimals=4)))
        gaps      = np.diff(unique_d)
        breaks    = list(np.where(gaps > 1.5 * np.mean(gaps))[0] + 1)[:n_shells]

        boundaries = [0] + breaks + [len(unique_d)]
        shells, shell_radii = [], []
        for i in range(min(n_shells, len(boundaries) - 1)):
            lo = unique_d[boundaries[i]]
            hi = unique_d[boundaries[i + 1] - 1]
            mask = (distances >= lo - 1e-3) & (distances <= hi + 1e-3)
            shells.append(list(np.where(mask)[0]))
            shell_radii.append(float(hi))
            print(f"defect_builder: shell {i+1}: {len(shells[-1])} atoms within {hi:.3f} Angstrom")

        return shells, shell_radii

    @staticmethod
    def _parse_xyz(path):
        """
        Parse a standard XYZ file. Returns (species list, coords array).
        """
        try:
            with open(path) as f:
                lines = f.readlines()
        except OSError as e:
            raise ValueError(f"defect_builder: cannot open '{path}': {e}")

        if len(lines) < 2:
            raise ValueError(f"defect_builder: file '{path}' is too short for XYZ format.")

        try:
            natoms = int(lines[0].strip())
        except ValueError:
            raise ValueError(
                f"defect_builder: first line of '{path}' must be atom count, "
                f"got '{lines[0].strip()}'."
            )

        species, coords = [], []
        for i, line in enumerate(lines[2:]):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    f"defect_builder: line {i+3} in '{path}' has fewer than 4 columns."
                )
            species.append(parts[0])
            try:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                raise ValueError(
                    f"defect_builder: cannot parse coordinates on line {i+3} of '{path}'."
                )

        if len(species) != natoms:
            raise ValueError(
                f"defect_builder: '{path}' header says {natoms} atoms, "
                f"found {len(species)}."
            )
        return species, np.array(coords, dtype=float)

    @staticmethod
    def _find_unmatched(ref_coords, ref_species, target_coords, target_species, tol):
        """
        Return indices into ref for atoms that have no match in target.
        An atom is matched if an atom of the same species exists within tol Angstroms.
        """
        unmatched = []
        for i, (sp, coord) in enumerate(zip(ref_species, ref_coords)):
            same_species_idx = [j for j, ts in enumerate(target_species) if ts == sp]
            if not same_species_idx:
                unmatched.append(i)
                continue
            dists = np.linalg.norm(target_coords[same_species_idx] - coord, axis=1)
            if np.min(dists) >= tol:
                unmatched.append(i)
        return unmatched
