"""
MaterialConfig: Dataclass describing the bulk crystal physics for QM/MM embedding.
All length units are Angstroms.

Lattice representation
----------------------
Provide either:
  - `lattice_constant` (float): Assumes a simple cubic lattice.
  - `lattice_vectors` (3x3 array-like): General Bravais lattice support.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MaterialConfig:
    """
    Container for the material-specific parameters used by QMMMBuilder.

    Parameters
    ----------
    unitcell : list of [str, float, float, float]
        Cartesian coordinates (Å) of each basis atom in the primitive cell.
        Format: [['Cu', x, y, z], ['O', x, y, z], ...]
    canonical_charges : dict
        Formal ionic charges, e.g. {'Cu': 1.0, 'O': -2.0}.
    bulk_coordinations : dict
        Ideal bulk coordination number for each species,
        e.g. {'Cu': 2, 'O': 4}.  Used for coordination-scaling.
    bond_cutoff : float
        Nearest-neighbour distance (Å) below which two atoms are
        considered bonded.
    lattice_constant : float, optional
        Cubic lattice parameter (Å).  Auto-builds a diagonal
        ``lattice_vectors`` matrix.  Provide this OR ``lattice_vectors``.
    lattice_vectors : array-like (3, 3), optional
        Full 3x3 lattice matrix in Å.  Rows are the three lattice vectors
        a1, a2, a3.  Provide this OR ``lattice_constant``.
    precision : int
        Number of decimal places kept on coordinates (default 8).
    ecp_exclude : list of str
        Elements excluded from the ECP shell (e.g. ['O']).
    name : str
        Human-readable material label for logging / file headers.
    """

    unitcell:          List[List]
    canonical_charges: Dict[str, float]
    bulk_coordinations: Dict[str, int]
    bond_cutoff:       float
    lattice_constant:  Optional[float]          = None
    lattice_vectors:   Optional[np.ndarray]     = None
    precision:         int                       = 8
    ecp_exclude:       List[str]                = field(default_factory=list)
    name:              str                       = "material"

    def __post_init__(self):
        if self.lattice_constant is None and self.lattice_vectors is None:
            raise ValueError(
                "Provide either 'lattice_constant' (cubic) or "
                "'lattice_vectors' (3x3 array)."
            )
        if self.lattice_vectors is not None:
            self.lattice_vectors = np.asarray(
                self.lattice_vectors, dtype=float
            )
            if self.lattice_vectors.shape != (3, 3):
                raise ValueError("lattice_vectors must be a (3, 3) array.")
            if self.lattice_constant is None:
                # derive a representative scalar from the matrix norm
                self.lattice_constant = float(
                    np.min(np.linalg.norm(self.lattice_vectors, axis=1))
                )
        else:
            # cubic shortcut — build diagonal matrix from lattice_constant
            a = self.lattice_constant
            self.lattice_vectors = np.diag([a, a, a]).astype(float)

    # ------------------------------------------------------------------
    # Derived quantities (always computed from lattice_vectors)
    # ------------------------------------------------------------------

    @property
    def tol(self) -> float:
        """Floating-point tolerance derived from precision."""
        return 10 ** (-self.precision)

    @property
    def characteristic_length(self) -> float:
        """
        The shortest lattice vector norm (Å).

        Used to convert shell thickness (in 'layer' units) to Å, and to
        estimate supercell dimensions.  For cubic systems this equals
        ``lattice_constant`` exactly, preserving backward compatibility.
        """
        return float(np.min(np.linalg.norm(self.lattice_vectors, axis=1)))


# ---------------------------------------------------------------------------
# Cu2O (cuprite, Pn-3m) — HSE06/ONCVPSP optimised unit cell
# Exactly reproduces the parameters from michael_code/config.py (Rev 1)
# ---------------------------------------------------------------------------
cu2o_hse06_opt_config = MaterialConfig(
    name="Cu2O_HSE06",
    lattice_constant=4.29820000,          # Å — HSE06/ONCVPSP (2026-04-26)
    bond_cutoff=1.86117520,               # Cu-O = a * sqrt(3) / 4
    canonical_charges={'Cu': 1.0, 'O': -2.0},
    bulk_coordinations={'Cu': 2, 'O': 4},
    ecp_exclude=[],
    unitcell=[
        ['Cu', 1.07455000, 1.07455000, 3.22365000],
        ['Cu', 1.07455000, 3.22365000, 1.07455000],
        ['Cu', 3.22365000, 1.07455000, 1.07455000],
        ['Cu', 3.22365000, 3.22365000, 3.22365000],
        ['O',  2.14910000, 2.14910000, 2.14910000],
        ['O',  0.00000000, 0.00000000, 4.29820000],
    ],
)

# ---------------------------------------------------------------------------
# Cu2O (cuprite, Pn-3m) — Materials Project
# Derived from examples/qmmm_check/Cu2O.cif
# ---------------------------------------------------------------------------
cu2o_matproj_config = MaterialConfig(
    name="Cu2O_MP",
    lattice_constant=4.24669932,
    bond_cutoff=1.83887475,
    canonical_charges={'Cu': 1.0, 'O': -2.0},
    bulk_coordinations={'Cu': 2, 'O': 4},
    ecp_exclude=[],
    unitcell=[
        ['Cu', 1.06167483, 1.06167483, 3.18502449],
        ['Cu', 3.18502449, 1.06167483, 1.06167483],
        ['Cu', 1.06167483, 3.18502449, 1.06167483],
        ['Cu', 3.18502449, 3.18502449, 3.18502449],
        ['O',  2.12334966, 2.12334966, 2.12334966],
        ['O',  0.00000000, 0.00000000, 0.00000000],
    ],
)
