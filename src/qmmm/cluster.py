"""QMMMCluster: structured output object from QMMMBuilder."""

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

# Visualization labels for each region
_VIS_LABELS = {
    REGION_QM:    '',          # keep the real element symbol
    REGION_ECP:   'Th',        # heavy/rare-earth placeholder for ECP zone
    REGION_MM:    '',          # keep the real element symbol
    REGION_GHOST: 'X',         # generic unknown-atom marker
}


def _parse_pyscf_label(label: str) -> Tuple[str, int]:
    """Recover (element, region) from a PySCF label: 'X-Cu1'→ECP, 'ghost-Cu'→ghost, 'Cu0'→QM."""
    if label.startswith('X-'):
        return label[2:].rstrip('0123456789'), REGION_ECP
    if label.startswith('ghost-'):
        return label[len('ghost-'):].rstrip('0123456789'), REGION_GHOST
    return label.rstrip('0123456789'), REGION_QM


@dataclass
class AtomRecord:
    """Lightweight container for one atom in the cluster (element, coords, charge, region, pyscf_label)."""
    element:     str
    coords:      np.ndarray
    charge:      float
    region:      int
    pyscf_label: str = ''


class QMMMCluster:
    """Structured result from QMMMBuilder with QM/ECP/ghost/MM atom views and PySCF interface helpers."""

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

    @classmethod
    def from_xyz(
        cls,
        filename: str,
        qm_charge: float = 0.0,
        qm_spin: int = 0,
        material_name: str = '',
    ) -> 'QMMMCluster':
        """Build a QMMMCluster from a PySCF-labeled XYZ file (inverse of to_xyz(mode='pyscf'))."""
        path = Path(filename).expanduser()
        atoms: List[AtomRecord] = []
        with open(path) as f:
            lines = f.readlines()
        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            label = parts[0]
            coords = np.array([float(c) for c in parts[1:4]], dtype=float)
            charge = float(parts[4]) if len(parts) >= 5 else 0.0
            element, region = _parse_pyscf_label(label)
            atoms.append(AtomRecord(
                element=element, coords=coords,
                charge=charge, region=region, pyscf_label=label,
            ))
        return cls(atoms, qm_charge=qm_charge, qm_spin=qm_spin,
                   material_name=material_name)

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

    @property
    def mm_coords(self) -> np.ndarray:
        """Cartesian coords (Å) of the classical environment (ECP first, then MM)."""
        classical = [a for a in self._atoms
                     if a.region in (REGION_ECP, REGION_MM)]
        if not classical:
            return np.empty((0, 3))
        return np.array([a.coords for a in classical])

    @property
    def mm_charges(self) -> np.ndarray:
        """Charges corresponding to mm_coords (same ordering)."""
        classical = [a for a in self._atoms
                     if a.region in (REGION_ECP, REGION_MM)]
        if not classical:
            return np.empty((0,))
        return np.array([a.charge for a in classical])

    def to_pyscf_mol_atom(self) -> List[Tuple[str, Tuple[float, float, float]]]:
        """Return a mol.atom list for all basis-carrying atoms (QM, ECP, ghost)."""
        mol_atoms = []
        for a in self._atoms:
            if a.region in (REGION_QM, REGION_ECP, REGION_GHOST):
                mol_atoms.append(
                    (a.pyscf_label, tuple(float(c) for c in a.coords))
                )
        return mol_atoms

    def ecp_dict(self, ecp_library: dict) -> dict:
        """Return a mol.ecp dict for all ECP boundary atoms using the provided element→ECP-string library."""
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
        """Return a mol.basis dict: QM and ghost atoms get qm_basis; ECP atoms get no basis functions."""
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

    def auxbasis_dict(self, library='cu2o_jkfit') -> dict:
        """Return a mol.auxbasis dict for QM and ghost atoms from the named library."""
        from .auxbasis_library import cu2o_jkfit
        _loaders = {'cu2o_jkfit': cu2o_jkfit}
        if library not in _loaders:
            raise ValueError(f"Unknown auxbasis library '{library}'. "
                             f"Available: {list(_loaders)}")
        loader = _loaders[library]
        auxbasis = {}
        for a in self.qm_atoms + self.ghost_atoms:
            lbl = a.pyscf_label
            if lbl not in auxbasis:
                auxbasis[lbl] = loader(a.element)
        return auxbasis

    def to_xyz(
        self,
        filename: str,
        region: str = 'full',
        mode: str = 'visual',
        include_charges: bool = False,
        overwrite: bool = True,
    ) -> None:
        """Write the cluster to an XYZ file; region selects subset, mode controls label style."""
        filepath = Path(filename).expanduser()
        if not overwrite and filepath.exists():
            raise FileExistsError(f"File '{filepath}' already exists.")

        region_map = {
            'full':     [REGION_QM, REGION_ECP, REGION_MM, REGION_GHOST],
            'qm':       [REGION_QM],
            'ecp':      [REGION_ECP],
            'mm':       [REGION_MM],
            'qm_ghost': [REGION_QM, REGION_GHOST],
        }
        if region not in region_map:
            raise ValueError(f"region must be one of {list(region_map.keys())}")
            
        if mode not in ('visual', 'pyscf', 'highlight'):
            raise ValueError("mode must be 'visual', 'pyscf', or 'highlight'")

        selected = [a for a in self._atoms if a.region in region_map[region]]

        with open(filepath, 'w') as f:
            f.write(f'{len(selected)}\n')
            f.write(
                f'{self.material_name}  charge={self.qm_charge:.4f}'
                f'  spin={self.qm_spin}\n'
            )
            for a in selected:
                if mode == 'pyscf':
                    label = a.pyscf_label
                elif mode == 'highlight':
                    vis   = _VIS_LABELS[a.region]
                    label = vis if vis else a.element
                else:
                    label = a.element
                x, y, z = a.coords
                row = f'{label:<6}  {x:15.8f}  {y:15.8f}  {z:15.8f}'
                if include_charges:
                    row += f'  {a.charge:12.8f}'
                f.write(row + '\n')

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

    def find_indices(
        self,
        mol_atom: list,
        region: str = 'qm',
        use_slices: bool = False,
        precision: int = 8,
    ) -> list:
        """Return indices of region atoms within a mol.atom list, matched by coordinate."""
        _region_map = {
            'qm': REGION_QM, 'ecp': REGION_ECP,
            'mm': REGION_MM, 'ghost': REGION_GHOST,
        }
        if region == 'all':
            selected = self._atoms
        elif region in _region_map:
            selected = [a for a in self._atoms if a.region == _region_map[region]]
        else:
            raise ValueError(
                f"region must be one of {list(_region_map) + ['all']}"
            )

        ref_map = {
            tuple(round(float(c), precision) for c in coords): i
            for i, (_label, coords) in enumerate(mol_atom)
        }
        indices = []
        for a in selected:
            key = tuple(round(float(c), precision) for c in a.coords)
            if key in ref_map:
                indices.append(ref_map[key])
        indices.sort()

        if not use_slices or not indices:
            return indices

        result = []
        start = prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                result.append(slice(start, prev + 1) if start != prev else start)
                start = prev = idx
        result.append(slice(start, prev + 1) if start != prev else start)
        return result

    def __repr__(self) -> str:
        return (
            f'QMMMCluster(qm={len(self.qm_atoms)}, '
            f'ecp={len(self.ecp_atoms)}, '
            f'mm={len(self.mm_atoms)}, '
            f'ghost={len(self.ghost_atoms)})'
        )
