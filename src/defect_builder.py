"""Defect detection and QM cluster selection from pristine/defect XYZ file pairs."""

import numpy as np


class DefectBuilder:
    """Automated defect detection and QM cluster selection from XYZ files."""

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
        """Construct a DefectBuilder from two standard XYZ files."""
        pristine_species, pristine_coords = cls._parse_xyz(pristine_path)
        defect_species, defect_coords     = cls._parse_xyz(defect_path)
        return cls(pristine_coords, pristine_species,
                   defect_coords, defect_species)

    def detect_center(self, match_tol=0.3):
        """Identify defect center (vacancy/interstitial/substitution) by comparing pristine and defect structures."""
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
        """Return defect-structure atom indices within radius Angstroms of the defect center."""
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
        """Identify the first n_shells nearest-neighbor shells from the defect center using gap detection."""
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
        """Parse a standard XYZ file. Returns (species list, coords array)."""
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
        """Return indices into ref for atoms with no same-species match in target within tol Angstroms."""
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
