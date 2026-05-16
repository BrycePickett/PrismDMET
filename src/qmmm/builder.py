"""
QMMMBuilder: Generalized SKZCAM cluster building framework.

Supported qm_method values:
- 'SKZCAM': Sphere-expansion snapping to magic-number plateaus.

Lattice tiling:
The supercell builder uses the full 3x3 lattice_vectors matrix from
MaterialConfig to support any Bravais lattice.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .material_config import MaterialConfig
from .cluster import (
    AtomRecord, QMMMCluster,
    REGION_QM, REGION_ECP, REGION_MM, REGION_GHOST,
)


# Internal list format:
#   [label: str, x, y, z]              length 4
#   [label, x, y, z, charge]           length 5
#   [label, x, y, z, charge, region]   length 6


class QMMMBuilder:
    """
    Build a QM/MM embedding cluster for a periodic crystal.

    Quick-start — Cu2O, 19-Cu SKZCAM (identical to Michael's output)
    -----------------------------------------------------------------
    >>> from prismdmet.qmmm import QMMMBuilder
    >>> from prismdmet.qmmm import cu2o_matproj_config
    >>> builder = QMMMBuilder(
    ...     config         = cu2o_matproj_config,
    ...     qm_method      = 'SKZCAM',
    ...     target_element = 'Cu',
    ...     center_element = 'Cu',
    ...     qm_target_size = 19,
    ...     ecp_layers     = 1,
    ...     total_layers   = 12.5,
    ...     mm_termination = 'Cu',
    ...     defect_type    = 'pristine',
    ... )
    >>> cluster = builder.build()
    >>> print(cluster.summary())

    Parameters
    ----------
    config : MaterialConfig
        Crystal physics. May use lattice_constant (cubic) or
        lattice_vectors (general 3x3).
    qm_method : str
        Cluster-building method.  Only 'SKZCAM' is currently implemented.
    target_element : str
        Element whose count defines the SKZCAM magic number (e.g. 'Cu').
    center_element : str
        Element to place at the cluster origin (e.g. 'Cu').
    qm_target_size : int, optional
        Number of target_element atoms in the QM region.
        Exactly one of qm_target_size / qm_radius must be supplied.
    qm_radius : float, optional
        Explicit QM sphere radius in Å (bypasses SKZCAM scan).
    ecp_layers : float
        ECP shell thickness in units of config.characteristic_length.
    pc_layers : float, optional
        MM shell thickness beyond the ECP shell, in units of
        config.characteristic_length.  Mutually exclusive with total_layers.
    total_layers : float, optional
        Total cluster radius in units of config.characteristic_length,
        measured from the central atom (origin-based).  Reproduces
        Michael's ``get_sphere(central, sphere_layers=N)`` behavior so the
        cluster size is constant across all QM sizes.  Mutually exclusive
        with pc_layers.
    mm_termination : str
        Element used to fully-coordinate the outer MM boundary
        Element that forms the boundary of the outer passivation shell.
        Only this element is added during boundary coordination.
        Default 'Cu' matches Michael's production model (cluster terminates
        in Cu, adding Cu atoms to satisfy dangling-bond coordination).
        Use 'O' to terminate in oxygen instead.
    defect_type : str
        'pristine'       — no defect.
        'vacancy'        — remove central defect_element; replace with
                           a ghost-basis atom (spin + 1 automatically).
        'substitutional' — swap central defect_element for
                           substitutional_element.
    defect_element : str, optional
        Element to remove/replace (defaults to center_element).
    substitutional_element : str, optional
        Element to insert for 'substitutional' defect_type.
    substitutional_spin : int, optional
        2*S for the substitutional system.  Defaults to 0.
    alt_charges : dict, optional
        Override canonical_charges for MM charge assignment.
    coordination_scaling : bool
        Apply coordination-scaling to MM charges (default True).
    neutralize : bool
        Run make_neutral() after charge assignment (default True).
    """

    def __init__(
        self,
        config:                 MaterialConfig,
        qm_method:              str             = 'SKZCAM',
        target_element:         str             = 'Cu',
        center_element:         str             = 'Cu',
        qm_target_size:         Optional[int]   = None,
        qm_radius:              Optional[float] = None,
        ecp_layers:             float           = 1.0,
        pc_layers:              Optional[float] = None,
        total_layers:           Optional[float] = None,
        mm_termination:         str             = 'Cu',
        defect_type:            str             = 'pristine',
        defect_element:         Optional[str]   = None,
        substitutional_element: Optional[str]   = None,
        substitutional_spin:    int             = 0,
        alt_charges:            Optional[Dict[str, float]] = None,
        coordination_scaling:   bool            = True,
        neutralize:             bool            = True,
    ):
        _valid_methods = ('SKZCAM',)
        _valid_defects = ('pristine', 'vacancy', 'substitutional')

        if qm_method not in _valid_methods:
            raise ValueError(
                f"Unknown qm_method '{qm_method}'. Supported: {_valid_methods}."
            )
        if (qm_target_size is None) == (qm_radius is None):
            raise ValueError(
                "Provide exactly one of qm_target_size or qm_radius."
            )
        if pc_layers is not None and total_layers is not None:
            raise ValueError(
                "Provide exactly one of pc_layers or total_layers, not both."
            )
        if pc_layers is None and total_layers is None:
            raise ValueError(
                "Provide either pc_layers (conformal shell) or "
                "total_layers (origin-based, matches Michael's get_sphere)."
            )
        if defect_type not in _valid_defects:
            raise ValueError(
                f"Unknown defect_type '{defect_type}'. "
                f"Supported: {_valid_defects}."
            )
        if defect_type == 'substitutional' and substitutional_element is None:
            raise ValueError(
                "defect_type='substitutional' requires substitutional_element."
            )

        self.config                 = config
        self.qm_method              = qm_method
        self.target_element         = target_element
        self.center_element         = center_element
        self.qm_target_size         = qm_target_size
        self.qm_radius              = qm_radius
        self.ecp_layers             = ecp_layers
        self.pc_layers              = pc_layers
        self.total_layers           = total_layers
        self.mm_termination         = mm_termination
        self.defect_type            = defect_type
        self.defect_element         = defect_element or center_element
        self.substitutional_element = substitutional_element
        self.substitutional_spin    = substitutional_spin
        self.alt_charges            = alt_charges
        self.coordination_scaling   = coordination_scaling
        self.neutralize             = neutralize

        self._qm_charge = 0.0
        self._qm_spin   = 0

    # ==========================================================================
    # Public API
    # ==========================================================================

    def find_skzcam_sizes(self, upper_bound: int = 200) -> Dict[int, float]:
        """
        Return all valid SKZCAM magic numbers up to *upper_bound*.

        Returns
        -------
        dict  {n_target_atoms: minimum_radius_angstrom}
        """
        return self._find_qm_skzcam(upper_bound=upper_bound)

    def build(self) -> QMMMCluster:
        """
        Construct the full QM/MM cluster and return a QMMMCluster.

        Steps
        -----
        1. Determine QM radius (SKZCAM scan or direct).
        2. Cut QM sphere; fully-coordinate target_element at boundary.
        3. Assign formal charges to QM region (no coordination-scaling).
        4. Build ECP shell (formal charges, no scaling).
        5. Build full environment with coord-scaling and neutralization.
           Two modes:
           - total_layers: origin-based absolute sphere (Michael's method).
             Total cluster size is constant across all QM sizes.
           - pc_layers: conformal shell grown outward from the QM surface.
        6. Partition into regions; apply defect.
        7. Package into QMMMCluster.
        """
        cfg = self.config

        # --- 1. QM radius ---------------------------------------------------
        if self.qm_method == 'SKZCAM':
            if self.qm_target_size is not None:
                radii = self._find_qm_skzcam(upper_bound=self.qm_target_size)
                if self.qm_target_size not in radii:
                    avail = sorted(radii.keys())
                    raise ValueError(
                        f"qm_target_size={self.qm_target_size} is not a "
                        f"valid SKZCAM plateau. Available: {avail}"
                    )
                qm_rad = radii[self.qm_target_size]
            else:
                qm_rad = self.qm_radius

        # --- 2. QM region ---------------------------------------------------
        qm_raw, qm_center = self._make_cluster(
            cfg.unitcell, self.center_element, rad=qm_rad
        )
        qm_raw = self._fully_coordinate(
            qm_raw, atom_type=self._complement_element(self.target_element)
        )

        # --- 3. Formal charges on QM (no coord-scaling) --------------------
        charges     = self.alt_charges or cfg.canonical_charges
        qm_charged  = self._assign_charges(qm_raw, charges,
                                           coordination_scaling=False)
        self._qm_charge = round(sum(a[4] for a in qm_charged), 6)

        # --- 4. ECP shell — formal charges, no scaling ---------------------
        ecp_raw     = self._get_shell(qm_raw, shell_layers=self.ecp_layers)
        ecp_charged = self._assign_charges(ecp_raw, charges,
                                           coordination_scaling=False)

        # --- 5. Full environment for neutralization ------------------------
        if self.total_layers is not None:
            # Origin-based: one fixed sphere from the central atom.
            # Matches: cluster = central + get_sphere(central, total_layers)
            total_rad = self.total_layers * cfg.characteristic_length
            full_raw  = self._get_sphere_absolute(qm_center, total_rad)
        else:
            # Conformal: shell grown outward from the QM surface.
            pc_shell_raw = self._get_shell(
                qm_raw,
                shell_layers=self.ecp_layers + self.pc_layers,
            )
            full_raw = [a[:4] for a in qm_raw] + pc_shell_raw

        full_raw     = self._fully_coordinate(full_raw,
                                              atom_type=self.mm_termination)
        full_charged = self._assign_charges(
            full_raw, charges,
            coordination_scaling=self.coordination_scaling,
        )
        if self.neutralize:
            full_charged = self._make_neutral(full_charged)

        # --- 6. Partition into regions -------------------------------------
        partitioned, qm_charge = self._partition(
            qm_charged, ecp_charged, full_charged
        )
        self._qm_charge = qm_charge

        # --- 7. Apply defect -----------------------------------------------
        if self.defect_type == 'vacancy':
            partitioned     = self._apply_vacancy(partitioned)
            self._qm_spin   = 1
        elif self.defect_type == 'substitutional':
            partitioned     = self._apply_substitutional(partitioned)
            self._qm_spin   = self.substitutional_spin

        # --- 8. Package -------------------------------------------------------
        return self._package(partitioned)

    # ==========================================================================
    # SKZCAM scan  (port of find_qm_SKZCAM)
    # ==========================================================================

    def _find_qm_skzcam(self, upper_bound: int) -> Dict[int, float]:
        """Return {n_target: min_radius} dict for all SKZCAM plateaus."""
        cfg = self.config
        p   = cfg.precision

        num_target_uc = sum(1 for at in cfg.unitcell
                            if at[0] == self.target_element)
        dim = int(math.ceil(
            (1 / num_target_uc * upper_bound * (6 / math.pi)) ** (1/3) + 1
        ))
        # ensure odd so center atom is at origin
        if dim % 2 == 0:
            dim += 1

        lattice, central_atom = self._make_supercell(
            cfg.unitcell, [dim]*3, center=self.center_element
        )
        lattice_sorted = sorted(
            lattice,
            key=lambda at: self._dist2(at[1:4], central_atom[1:4])
        )

        tot_target   = 0
        current_rad2 = None
        result: Dict[int, float] = {}

        for at in lattice_sorted:
            r2 = round(self._dist2(central_atom[1:4], at[1:4]), p)
            if at[0] == self.target_element:
                if current_rad2 is not None and current_rad2 != r2:
                    result[tot_target] = round(current_rad2 ** 0.5, p)
                tot_target  += 1
                current_rad2 = r2

        if current_rad2 is not None:
            result[tot_target] = round(current_rad2 ** 0.5, p)

        return {k: v for k, v in result.items() if k <= upper_bound}

    # ==========================================================================
    # Geometry helpers
    # ==========================================================================

    @staticmethod
    def _dist2(c1, c2) -> float:
        return sum((x - y) ** 2 for x, y in zip(c1, c2))

    def _make_supercell(
        self,
        unitcell: list,
        dimensions: list,
        center=False,
    ) -> Tuple[list, list]:
        """
        Tile *unitcell* into a supercell of size *dimensions*.

        Uses the full 3x3 lattice_vectors matrix: new coordinates are
            r_new = r_atom + l*a1 + w*a2 + h*a3
        where a1, a2, a3 are the rows of config.lattice_vectors.
        This handles all Bravais lattice types correctly.
        """
        cfg = self.config
        lv  = cfg.lattice_vectors   # shape (3, 3); rows = lattice vectors
        p   = cfg.precision

        coords_set = set()
        result = []

        # Center the tiling for odd dimensions (places origin at center)
        if all(d % 2 == 1 for d in dimensions):
            ranges = [range(-(d // 2), d // 2 + 1) for d in dimensions]
        else:
            ranges = [range(d) for d in dimensions]

        for l in ranges[0]:
            for w in ranges[1]:
                for h in ranges[2]:
                    shift = l * lv[0] + w * lv[1] + h * lv[2]
                    for at in unitcell:
                        nc = tuple(
                            round(at[i+1] + shift[i], p) for i in range(3)
                        )
                        if nc in coords_set:
                            continue
                        result.append([at[0], nc[0], nc[1], nc[2]])
                        coords_set.add(nc)

        if center:
            ct      = center if isinstance(center, str) else None
            central = self._find_central_atom(result, atom_type=ct)
            result.sort(key=lambda at: self._dist2(at[1:], central[1:]))
            return result, central

        return result, result[0]

    def _find_central_atom(self, cluster: list, atom_type=None) -> list:
        """Port of find_central_atom() — atom closest to geometric center."""
        cfg = self.config
        p   = cfg.precision

        coords     = [at[1:4] for at in cluster]
        mins       = [min(c[i] for c in coords) for i in range(3)]
        maxs       = [max(c[i] for c in coords) for i in range(3)]
        geo_center = [round((mn + mx) / 2, p) for mn, mx in zip(mins, maxs)]

        best_d2 = float('inf')
        best    = None
        for at in cluster:
            if atom_type and at[0] != atom_type:
                continue
            d2 = round(self._dist2(at[1:4], geo_center), p)
            if d2 < best_d2:
                best_d2 = d2
                best    = at
        if best is None:
            raise RuntimeError(
                f"No atom of type '{atom_type}' found in cluster."
            )
        return best

    def _make_cluster(
        self, unitcell: list, center_type: str, rad: float
    ) -> Tuple[list, list]:
        """Port of make_cluster() — cut a sphere of radius *rad* from lattice."""
        cfg = self.config
        cl  = cfg.characteristic_length
        tol = cfg.tol

        # use characteristic_length for dimension estimate (safe for all lattices)
        dim = int(2 * ((rad + tol) // cl) + 3)
        if dim % 2 == 0:
            dim += 1   # keep odd for symmetric centering

        lattice, central = self._make_supercell(
            unitcell, [dim]*3, center=center_type
        )
        result = [
            at for at in lattice
            if self._dist2(central[1:], at[1:]) < (rad + tol) ** 2
        ]
        return result, central

    def _get_coordinations(self, cluster: list) -> List[float]:
        """Port of get_coordinations() — fractional coordination per atom."""
        cfg    = self.config
        cutoff = cfg.bond_cutoff + cfg.tol
        coords  = [at[1:4] for at in cluster]
        species = [at[0]   for at in cluster]
        tree    = cKDTree(coords)
        fracs   = []
        for i, (sp, c) in enumerate(zip(species, coords)):
            nbrs = [j for j in tree.query_ball_point(c, cutoff) if j != i]
            bulk = cfg.bulk_coordinations.get(sp, len(nbrs) or 1)
            fracs.append(len(nbrs) / bulk)
        return fracs

    def _assign_charges(
        self,
        cluster: list,
        charges: dict,
        coordination_scaling: bool = True,
    ) -> list:
        """Port of assign_charges()."""
        if coordination_scaling:
            fracs = self._get_coordinations(cluster)
            return [
                at[:4] + [charges[at[0]] * f]
                for at, f in zip(cluster, fracs)
            ]
        return [at[:4] + [charges[at[0]]] for at in cluster]

    def _adjust_charges(self, cluster: list, charge_diff: float) -> list:
        """Port of adjust_charges() — distribute residual over under-coord atoms."""
        cfg   = self.config
        p     = cfg.precision
        fracs = self._get_coordinations(cluster)
        under = [i for i, f in enumerate(fracs) if f < 1.0 - cfg.tol]
        if not under:
            raise RuntimeError(
                "No under-coordinated atoms found for charge neutralization. "
                "Try increasing pc_thickness."
            )
        shift  = charge_diff / len(under)
        result = []
        for i, at in enumerate(cluster):
            if i in under:
                new_q = round(at[4] + shift, p)
                result.append(at[:4] + [new_q] + at[5:])
            else:
                result.append(at)
        return result

    def _make_neutral(self, cluster: list) -> list:
        """Port of make_neutral() — neutralize total charge."""
        tot = sum(at[4] for at in cluster)
        if abs(tot) > 1e-3:
            return self._adjust_charges(cluster, -tot)
        return cluster

    def _get_shell(self, cluster: list, shell_layers: float) -> list:
        """
        Return atoms within shell_layers * characteristic_length of *cluster*.

        Surface-based (conformal): shell thickness is measured outward from
        the surface of *cluster*, not from the central atom.
        """
        cfg       = self.config
        cl        = cfg.characteristic_length
        tol       = cfg.tol
        p         = cfg.precision
        shell_rad = shell_layers * cl

        # build a lattice large enough to contain the shell
        central   = self._find_central_atom(cluster,
                                            atom_type=self.center_element)
        cluster_r = max(
            self._dist2(central[1:4], at[1:4]) for at in cluster
        ) ** 0.5
        rad = cluster_r + shell_rad
        dim = int(2 * math.ceil(rad / cl) + 1)
        if dim % 2 == 0:
            dim += 1

        temp_lat, lat_center = self._make_supercell(
            cfg.unitcell, [dim]*3, center=self.center_element
        )

        # realign lattice center onto the cluster center
        dx = [central[i+1] - lat_center[i+1] for i in range(3)]
        lattice = [
            [at[0]] + [round(at[i+1] + dx[i], p) for i in range(3)]
            for at in temp_lat
        ]

        cluster_set = {
            (round(at[1], p), round(at[2], p), round(at[3], p))
            for at in cluster
        }
        lat_filtered = [
            (at[0], [round(at[i+1], p) for i in range(3)])
            for at in lattice
            if (round(at[1], p), round(at[2], p), round(at[3], p))
            not in cluster_set
        ]
        if not lat_filtered:
            return []

        lab_f, crd_f = zip(*lat_filtered)
        tree     = cKDTree(list(crd_f))
        cl_crds  = [[round(at[i+1], p) for i in range(3)] for at in cluster]
        nbr_idxs = tree.query_ball_point(cl_crds, shell_rad + tol)

        shell_atoms  = []
        shell_coords = set()
        for idx_list in nbr_idxs:
            for idx in idx_list:
                ct = tuple(crd_f[idx])
                if ct not in shell_coords:
                    shell_coords.add(ct)
                    shell_atoms.append([lab_f[idx]] + list(ct))
        return shell_atoms

    def _get_sphere_absolute(self, central_atom: list, rad: float) -> list:
        """
        Return all atoms within *rad* Å of *central_atom* (origin-based).

        Reproduces Michael's ``get_sphere(central, sphere_layers=N)``.
        The sphere boundary is fixed to the central atom regardless of the
        QM size, so total cluster size is constant across scaling studies.
        """
        cfg = self.config
        cl  = cfg.characteristic_length
        tol = cfg.tol
        p   = cfg.precision

        dim = int(2 * math.ceil(rad / cl) + 1)
        if dim % 2 == 0:
            dim += 1

        temp_lat, lat_center = self._make_supercell(
            cfg.unitcell, [dim]*3, center=self.center_element
        )

        # align the supercell center onto the exact central_atom position
        dx = [central_atom[i+1] - lat_center[i+1] for i in range(3)]
        lattice = [
            [at[0]] + [round(at[i+1] + dx[i], p) for i in range(3)]
            for at in temp_lat
        ]

        cx, cy, cz = central_atom[1], central_atom[2], central_atom[3]
        return [
            at for at in lattice
            if self._dist2([cx, cy, cz], at[1:4]) < (rad + tol) ** 2
        ]

    def _complement_element(self, element: str) -> Optional[str]:
        """Return the first element in the unitcell that is not *element*."""
        for at in self.config.unitcell:
            if at[0] != element:
                return at[0]
        return None

    def _fully_coordinate(
        self, cluster: list, atom_type: Optional[str] = None
    ) -> list:
        """
        Add missing bond-level neighbors to fully coordinate *cluster*.

        If *atom_type* is given, only atoms of that element are added
        (e.g. 'Cu' adds only Cu to satisfy dangling-bond coordination
        at the cluster boundary).  If None, all bonding neighbors are added.
        """
        cfg       = self.config
        bond_frac = cfg.bond_cutoff / cfg.characteristic_length
        new_atoms = self._get_shell(cluster, shell_layers=bond_frac + cfg.tol)
        if atom_type:
            new_atoms = [a for a in new_atoms if a[0] == atom_type]
        return cluster + new_atoms

    def _partition(
        self,
        qm_charged:   list,
        ecp_charged:  list,
        full_charged: list,
    ) -> Tuple[list, float]:
        """Port of partition() — assign region labels 0/1/2."""
        cfg = self.config
        p   = cfg.precision

        qm_ecp_coords: set = set()
        result: list = []
        qm_charge    = 0.0

        for at in qm_charged:
            result.append(at + [0])
            qm_charge += at[4]
            qm_ecp_coords.add(
                tuple(round(at[i+1], p) for i in range(3))
            )

        for at in ecp_charged:
            ct = tuple(round(at[i+1], p) for i in range(3))
            if ct in qm_ecp_coords:
                continue
            if at[0] in cfg.ecp_exclude:
                continue
            result.append(at + [1])
            qm_ecp_coords.add(ct)

        for at in full_charged:
            ct = tuple(round(at[i+1], p) for i in range(3))
            if ct not in qm_ecp_coords:
                result.append(at + [2])

        center = self._find_central_atom(qm_charged)
        result.sort(key=lambda a: (a[5], self._dist2(a[1:4], center[1:4])))
        return result, round(qm_charge, 6)

    # ==========================================================================
    # Defect handlers
    # ==========================================================================

    def _apply_vacancy(self, partitioned: list) -> list:
        """Convert the first central defect_element QM atom to a ghost (region 3)."""
        replaced = False
        result   = []
        for at in partitioned:
            if not replaced and at[0] == self.defect_element and at[5] == 0:
                result.append([at[0], at[1], at[2], at[3], 0.0, REGION_GHOST])
                replaced = True
            else:
                result.append(at)
        if not replaced:
            warnings.warn(
                f"No central '{self.defect_element}' QM atom found to "
                "convert to vacancy.",
                UserWarning,
            )
        return result

    def _apply_substitutional(self, partitioned: list) -> list:
        """
        Swap the first central defect_element QM atom for substitutional_element.

        The substituted atom keeps its position.  Its charge is set from
        canonical_charges if available; otherwise it inherits the original
        element's charge.

        Note: canonical_charges must include substitutional_element if
        charge accuracy is required.
        """
        sub_el  = self.substitutional_element
        charges = self.alt_charges or self.config.canonical_charges

        replaced = False
        result   = []
        for at in partitioned:
            if not replaced and at[0] == self.defect_element and at[5] == 0:
                sub_charge = charges.get(sub_el, at[4])
                result.append([sub_el, at[1], at[2], at[3], sub_charge, 0])
                replaced = True
            else:
                result.append(at)
        if not replaced:
            warnings.warn(
                f"No central '{self.defect_element}' QM atom found for "
                "substitutional replacement.",
                UserWarning,
            )
        return result

    # ==========================================================================
    # Packaging → QMMMCluster
    # ==========================================================================

    def _pyscf_label(self, element: str, region: int) -> str:
        """
        Map (element, region) → PySCF mol.atom label.

        Follows Michael's file_utils.py convention:
          QM (0)    → 'Cu0', 'O0'          (suffix = region index)
          ECP (1)   → 'X-Cu1', 'X-O1'      (X- prefix; PySCF ECP-eligible)
          Ghost (3) → 'ghost-Cu'            (PySCF native ghost-basis syntax)
          MM (2)    → element only          (not in mol.atom)
        """
        if region == REGION_QM:
            return f'{element}0'
        if region == REGION_ECP:
            return f'X-{element}1'
        if region == REGION_GHOST:
            return f'ghost-{element}'
        return element   # MM — not in mol.atom

    def _package(self, partitioned: list) -> QMMMCluster:
        """Convert raw partitioned list → QMMMCluster."""
        region_map = {0: REGION_QM, 1: REGION_ECP,
                      2: REGION_MM, 3: REGION_GHOST}
        atoms: List[AtomRecord] = []
        for raw in partitioned:
            el     = raw[0]
            coords = np.array([raw[1], raw[2], raw[3]], dtype=float)
            charge = raw[4]
            region = region_map[raw[5]]
            label  = self._pyscf_label(el, region)
            atoms.append(AtomRecord(
                element=el, coords=coords,
                charge=charge, region=region,
                pyscf_label=label,
            ))

        return QMMMCluster(
            atoms=atoms,
            qm_charge=self._qm_charge,
            qm_spin=self._qm_spin,
            material_name=self.config.name,
        )
