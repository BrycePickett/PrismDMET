"""
QMMMCluster: structured output object from QMMMBuilder.

Atom labeling strategy (critical for PySCF correctness):
----------------------------------------------------------------
Region 0 (QM atoms)    : stored as 'Cu0', 'O0' (suffix = region index)
Region 1 (ECP atoms)   : stored as 'X-Cu1', 'X-O1' ('X-' prefix allows ECPs)
Region 3 (Ghost atoms) : stored as 'ghost-Cu' (PySCF ghost syntax)
Region 2 (MM charges)  : not in mol.atom; only in mm_coords/mm_charges

For XYZ visualization, we maintain a 'display_element' alongside
each atom so that structure viewers (VESTA, Avogadro, etc.) see
chemically meaningful symbols:
    ghost-Cu  -> 'X'
    X-Cu1     -> 'Th'
    O0 / Cu0  -> 'Cu' / 'O'
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Region constants (module-level so they can be imported anywhere)
# ---------------------------------------------------------------------------
REGION_QM    = 0   # quantum-mechanically treated
REGION_ECP   = 1   # Pauli repulsion boundary (nelec 0 ECP)
REGION_MM    = 2   # classical point charges only
REGION_GHOST = 3   # basis functions, no nuclear charge (vacancy site)

# Visualisation labels for each region
_VIS_LABELS = {
    REGION_QM:    '',          # keep the real element symbol
    REGION_ECP:   'Th',        # heavy/rare-earth placeholder for ECP zone
    REGION_MM:    '',          # keep the real element symbol
    REGION_GHOST: 'X',         # generic unknown-atom marker
}


@dataclass
class AtomRecord:
    """
    Lightweight container for one atom in the cluster.

    Attributes
    ----------
    element : str
        True element symbol (e.g. 'Cu', 'O').
    coords : np.ndarray  shape (3,)
        Cartesian position in Angstrom.
    charge : float
        Point-charge value assigned to this atom (0.0 for QM atoms).
    region : int
        One of REGION_QM, REGION_ECP, REGION_MM, REGION_GHOST.
    pyscf_label : str
        The label that will appear in mol.atom (e.g. 'Cu0', 'X-Cu1',
        'ghost-Cu').  Set by QMMMCluster at build time.
    """
    element:     str
    coords:      np.ndarray
    charge:      float
    region:      int
    pyscf_label: str = ''


class QMMMCluster:
    """
    Structured result from QMMMBuilder.

    The cluster is split into four named groups that map directly onto
    the workflow in PySCF:

      .qm_atoms     → mol.atom  (region 0)
      .ecp_atoms    → mol.atom  (region 1, with ECPs in mol.ecp)
      .ghost_atoms  → mol.atom  (region 3, ghost basis, no nuclear charge)
      .mm_coords    → np.ndarray passed to pyscf.qmmm.mm_charge()
      .mm_charges   → np.ndarray passed to pyscf.qmmm.mm_charge()

    The ECP coords / charges are ALSO added to mm_coords / mm_charges so
    that PySCF's QM/MM driver sees a complete electrostatic environment
    (matching Michael's approach exactly).

    Parameters
    ----------
    atoms : list of AtomRecord
        Full ordered list of all atoms (all regions).
    qm_charge : float
        Net charge of the QM region.
    qm_spin : int
        2 * S for the QM region (0 = closed-shell).
    material_name : str
        Label passed through to file headers.
    """

    def __init__(
        self,
        atoms: List[AtomRecord],
        qm_charge: float,
        qm_spin: int,
        material_name: str = '',
    ):
        self._atoms        = atoms
        self.qm_charge     = qm_charge
        self.qm_spin       = qm_spin
        self.material_name = material_name

    # ------------------------------------------------------------------
    # Filtered atom views
    # ------------------------------------------------------------------

    @property
    def qm_atoms(self) -> List[AtomRecord]:
        return [a for a in self._atoms if a.region == REGION_QM]

    @property
    def ecp_atoms(self) -> List[AtomRecord]:
        return [a for a in self._atoms if a.region == REGION_ECP]

    @property
    def ghost_atoms(self) -> List[AtomRecord]:
        return [a for a in self._atoms if a.region == REGION_GHOST]

    @property
    def mm_atoms(self) -> List[AtomRecord]:
        return [a for a in self._atoms if a.region == REGION_MM]

    @property
    def all_atoms(self) -> List[AtomRecord]:
        return list(self._atoms)

    # ------------------------------------------------------------------
    # Electrostatics arrays (ECP + MM combined, matching Michael's output)
    # ------------------------------------------------------------------

    @property
    def mm_coords(self) -> np.ndarray:
        """
        Cartesian coords (Å) of the entire classical environment:
        ECP atoms first, then MM point charges — exactly matching the
        ``coords`` list in the PySCF scripts that Michael's code generates.
        """
        classical = [a for a in self._atoms
                     if a.region in (REGION_ECP, REGION_MM)]
        if not classical:
            return np.empty((0, 3))
        return np.array([a.coords for a in classical])

    @property
    def mm_charges(self) -> np.ndarray:
        """
        Corresponding charges for mm_coords (same ordering).
        """
        classical = [a for a in self._atoms
                     if a.region in (REGION_ECP, REGION_MM)]
        if not classical:
            return np.empty((0,))
        return np.array([a.charge for a in classical])

    # ------------------------------------------------------------------
    # PySCF mol.atom list
    # ------------------------------------------------------------------

    def to_pyscf_mol_atom(self) -> List[Tuple[str, Tuple[float, float, float]]]:
        """
        Return a list suitable for ``mol.atom`` covering all atoms that
        require a basis set: QM (region 0), ECP boundary (region 1),
        and ghost atoms (region 3).

        Example
        -------
        >>> mol.atom = cluster.to_pyscf_mol_atom()
        """
        mol_atoms = []
        for a in self._atoms:
            if a.region in (REGION_QM, REGION_ECP, REGION_GHOST):
                mol_atoms.append(
                    (a.pyscf_label, tuple(float(c) for c in a.coords))
                )
        return mol_atoms

    def ecp_dict(self, ecp_library: dict) -> dict:
        """
        Return a ``mol.ecp`` dictionary for all ECP boundary atoms.

        Parameters
        ----------
        ecp_library : dict
            Mapping from element symbol to raw ECP string,
            e.g. from ``qmmm.ecp_library.ECP_LIBRARY``.

        Example
        -------
        >>> from qmmm.ecp_library import ECP_LIBRARY
        >>> mol.ecp = cluster.ecp_dict(ECP_LIBRARY)
        """
        import pyscf.gto as gto
        ecp = {}
        for a in self.ecp_atoms:
            lbl = a.pyscf_label           # e.g. 'X-Cu1'
            el  = a.element               # e.g. 'Cu'
            if lbl not in ecp and el in ecp_library:
                ecp[lbl] = gto.basis.parse_ecp(ecp_library[el])
        return ecp

    def basis_dict(self, qm_basis: str = 'def2-svp',
                   ghost_basis: Optional[dict] = None) -> dict:
        """
        Return a ``mol.basis`` dictionary.

        - QM atoms  (e.g. 'Cu0'):    assigned *qm_basis* by name.
        - Ghost atoms (e.g. 'ghost-Cu'): basis loaded explicitly from the
          parent element using ``pyscf.gto.basis.load(qm_basis, element)``
          because PySCF does not always infer it from the 'ghost-X' label.
        - ECP atoms (e.g. 'X-Cu1'): assigned an EMPTY basis ``{}`` so
          PySCF does not warn 'Basis not found'.  These atoms are nelec=0
          repulsion-only — they carry a mol.ecp entry but NO basis functions.

        Parameters
        ----------
        qm_basis : str
            Basis set name for QM and ghost atoms (e.g. 'def2-svp').
        ghost_basis : dict, optional
            Override mapping {'ghost-Cu': <basis>}.  If None, borrows
            *qm_basis* from the parent element automatically.

        Example
        -------
        >>> mol.basis = cluster.basis_dict('def2-svp')
        """
        import pyscf.gto as gto
        basis = {}
        # QM atoms — load by name; PySCF resolves via element suffix
        for a in self.qm_atoms:
            lbl = a.pyscf_label           # e.g. 'Cu0'
            if lbl not in basis:
                basis[lbl] = qm_basis
        # ECP boundary atoms — NOT included in mol.basis.
        # PySCF accepts mol.atom entries with a mol.ecp but no mol.basis;
        # it silently assigns no basis functions to those atoms (correct for
        # nelec=0 repulsion-only ECPs).  Adding {} here causes mol.build()
        # to throw an exception.
        # Ghost atoms — must load basis explicitly; PySCF may not infer it

        for a in self.ghost_atoms:
            lbl = a.pyscf_label           # e.g. 'ghost-Cu'
            if lbl not in basis:
                if ghost_basis and lbl in ghost_basis:
                    basis[lbl] = ghost_basis[lbl]
                else:
                    basis[lbl] = gto.basis.load(qm_basis, a.element)
        return basis

    # ------------------------------------------------------------------
    # XYZ file export
    # ------------------------------------------------------------------

    def to_xyz(
        self,
        filename: str,
        mode: str = 'visual',
        include_charges: bool = False,
        overwrite: bool = True,
    ) -> None:
        """
        Write the cluster to an XYZ file.

        Parameters
        ----------
        filename : str
            Output file path.
        mode : str
            'visual'  — human-readable element labels for structure
                        viewers (ghost → 'X', ECP → 'Th').
            'pyscf'   — PySCF labels (e.g. 'Cu0', 'X-Cu1', 'ghost-Cu').
                        Useful for debugging mol.atom.
            'full'    — all four regions, visual labels.
            'qm_only' — QM + ghost atoms only, visual labels.
        include_charges : bool
            If True, append the point charge as a 5th column (produces
            a .qxyz-style file readable by Molden/Chemcraft).
        overwrite : bool
            Silently overwrite existing files.
        """
        filepath = Path(filename).expanduser()
        if not overwrite and filepath.exists():
            raise FileExistsError(f"File '{filepath}' already exists.")

        mode_map = {
            'visual':  [REGION_QM, REGION_ECP, REGION_MM, REGION_GHOST],
            'pyscf':   [REGION_QM, REGION_ECP, REGION_GHOST],
            'full':    [REGION_QM, REGION_ECP, REGION_MM, REGION_GHOST],
            'qm_only': [REGION_QM, REGION_GHOST],
        }
        if mode not in mode_map:
            raise ValueError(f"mode must be one of {list(mode_map.keys())}")

        selected = [a for a in self._atoms if a.region in mode_map[mode]]

        with open(filepath, 'w') as f:
            f.write(f'{len(selected)}\n')
            f.write(
                f'{self.material_name}  charge={self.qm_charge:.4f}'
                f'  spin={self.qm_spin}\n'
            )
            for a in selected:
                if mode == 'pyscf':
                    label = a.pyscf_label
                else:
                    vis = _VIS_LABELS[a.region]
                    label = vis if vis else a.element
                x, y, z = a.coords
                row = f'{label:<6}  {x:15.8f}  {y:15.8f}  {z:15.8f}'
                if include_charges:
                    row += f'  {a.charge:12.8f}'
                f.write(row + '\n')

    # ------------------------------------------------------------------
    # Quick summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f'QMMMCluster ({self.material_name})',
            f'  QM atoms   : {len(self.qm_atoms)}',
            f'  ECP atoms  : {len(self.ecp_atoms)}',
            f'  Ghost atoms: {len(self.ghost_atoms)}',
            f'  MM charges : {len(self.mm_atoms)}',
            f'  QM charge  : {self.qm_charge:+.4f}',
            f'  QM spin    : {self.qm_spin}',
            f'  MM total Q : {self.mm_charges.sum():+.6f}',
        ]
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (
            f'QMMMCluster(qm={len(self.qm_atoms)}, '
            f'ecp={len(self.ecp_atoms)}, '
            f'mm={len(self.mm_atoms)}, '
            f'ghost={len(self.ghost_atoms)})'
        )
