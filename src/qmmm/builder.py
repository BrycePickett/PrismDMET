"""SKZCAM QM/MM cluster builder for periodic crystals."""

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
    """Build a QM/MM embedding cluster for a periodic crystal using the SKZCAM method."""

    def __init__(
        self,
        config:                 MaterialConfig,
        qm_method:              str             = 'SKZCAM',
        target_element:         str             = 'Cu',
        center_element:         str             = 'Cu',
        qm_target_size:         Optional[int]   = None,
        qm_radius:              Optional[float] = None,
        remove_unbonded:        bool            = False,
        ecp_layers:             float           = 1.0,
        ecp_shape:              str             = 'shell',
        pc_layers:              Optional[float] = None,
        total_layers:           Optional[float] = None,
        mm_termination:         str             = 'Cu',
        defect_type:            str             = 'pristine',
        defect_element:         Optional[str]   = None,
        defect_site:            Optional[list]  = None,
        substitutional_element: Optional[str]   = None,
        substitutional_spin:    int             = 0,
        interstitial_element:   Optional[str]   = None,
        interstitial_position:  Optional[list]  = None,
        defect_charge:          int             = 0,
        defect_spin:            Optional[int]   = None,
        defects:                Optional[list]  = None,
        alt_charges:            Optional[Dict[str, float]] = None,
        coordination_scaling:   bool            = True,
        neutralize:             bool            = True,
        neutralize_target:      str             = 'qm_charge',
    ):
        _valid_methods    = ('SKZCAM',)
        _valid_defects    = ('pristine', 'vacancy', 'substitutional',
                             'interstitial')
        _valid_ecp_shapes = ('shell', 'bonds', 'sphere')
        _valid_neut       = ('qm_charge', 'environment')

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
        if ecp_shape not in _valid_ecp_shapes:
            raise ValueError(
                f"Unknown ecp_shape '{ecp_shape}'. "
                f"Supported: {_valid_ecp_shapes}."
            )
        if neutralize_target not in _valid_neut:
            raise ValueError(
                f"Unknown neutralize_target '{neutralize_target}'. "
                f"Supported: {_valid_neut}."
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
        if defect_type == 'interstitial' and (
                interstitial_element is None or interstitial_position is None):
            raise ValueError(
                "defect_type='interstitial' requires interstitial_element "
                "and interstitial_position."
            )
        if defects is not None and defect_spin is None:
            raise ValueError(
                "defects requires an explicit defect_spin (total 2*S), since "
                "the spin of a multi-defect system is ambiguous."
            )

        uc_elements = {at[0] for at in config.unitcell}
        for name, el in (('center_element', center_element),
                         ('target_element', target_element),
                         ('mm_termination', mm_termination)):
            if el not in uc_elements:
                raise ValueError(
                    f"{name}='{el}' is not in config.unitcell "
                    f"{sorted(uc_elements)}."
                )
        resolved_defect = defect_element or center_element
        if (defect_type in ('vacancy', 'substitutional')
                and resolved_defect not in uc_elements):
            raise ValueError(
                f"defect_element='{resolved_defect}' is not in config.unitcell "
                f"{sorted(uc_elements)}."
            )

        self.config                 = config
        self.qm_method              = qm_method
        self.target_element         = target_element
        self.center_element         = center_element
        self.qm_target_size         = qm_target_size
        self.qm_radius              = qm_radius
        self.remove_unbonded        = remove_unbonded
        self.ecp_layers             = ecp_layers
        self.ecp_shape              = ecp_shape
        self.pc_layers              = pc_layers
        self.total_layers           = total_layers
        self.mm_termination         = mm_termination
        self.defect_type            = defect_type
        self.defect_element         = defect_element or center_element
        self.defect_site            = defect_site
        self.substitutional_element = substitutional_element
        self.substitutional_spin    = substitutional_spin
        self.interstitial_element   = interstitial_element
        self.interstitial_position  = interstitial_position
        self.defects                = defects
        self.defect_charge          = defect_charge
        self.defect_spin            = defect_spin
        self.alt_charges            = alt_charges
        self.coordination_scaling   = coordination_scaling
        self.neutralize             = neutralize
        self.neutralize_target      = neutralize_target

        self._qm_charge = 0.0
        self._qm_spin   = 0
        self._lattice_cache: dict = {}

    def find_skzcam_sizes(self, upper_bound: int = 200) -> Dict[int, float]:
        """Return {n_target_atoms: min_radius_angstrom} for all valid SKZCAM magic numbers up to upper_bound."""
        return self._find_qm_skzcam(upper_bound=upper_bound)

    def build(self) -> QMMMCluster:
        """Construct the full QM/MM cluster and return a QMMMCluster."""
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
            cfg.unitcell, self.center_element, rad=qm_rad,
            remove_unbonded=self.remove_unbonded,
        )
        qm_raw = self._fully_coordinate(
            qm_raw, atom_type=self._complement_element(self.target_element)
        )

        # --- 3. QM formal charge (canonical; sets mol.charge) --------------
        # The quantum region's charge is fixed by its nuclei, so it always
        # uses canonical charges, independent of any alt_charges applied to
        # the classical embedding field below.
        qm_charged = self._assign_charges(qm_raw, cfg.canonical_charges,
                                          coordination_scaling=False)
        self._qm_charge = round(sum(a[4] for a in qm_charged), 6)

        # Embedding point charges (ECP + MM) may be scaled via alt_charges.
        charges = self.alt_charges or cfg.canonical_charges

        # --- 4. ECP shell — formal charges, no scaling ---------------------
        ecp_raw     = self._build_ecp_shell(qm_raw, qm_center)
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
        if self.neutralize and self.neutralize_target == 'environment':
            full_charged = self._make_neutral(full_charged)

        # --- 6. Partition into regions -------------------------------------
        partitioned, qm_charge = self._partition(
            qm_charged, ecp_charged, full_charged
        )
        self._qm_charge = qm_charge

        # Balance the boundary so ECP+MM sum to -qm_charge (neutral system).
        if self.neutralize and self.neutralize_target == 'qm_charge':
            partitioned = self._rebalance_mm(partitioned, target=-qm_charge)

        # --- 7. Apply defect -----------------------------------------------
        if self.defects is not None:
            for spec in self.defects:
                partitioned = self._apply_defect_spec(partitioned, spec)
        elif self.defect_type == 'vacancy':
            partitioned     = self._apply_vacancy(partitioned, self.defect_site)
            self._qm_spin   = 1
        elif self.defect_type == 'substitutional':
            partitioned     = self._apply_substitutional(partitioned,
                                                         self.defect_site)
            self._qm_spin   = self.substitutional_spin
        elif self.defect_type == 'interstitial':
            partitioned     = self._apply_interstitial(
                partitioned, self.interstitial_element,
                self.interstitial_position,
            )

        # Charged defect: shift the QM charge (MM stays at the neutral bulk
        # value, so the full system carries net charge = defect_charge).
        self._qm_charge += self.defect_charge
        if self.defect_spin is not None:
            self._qm_spin = self.defect_spin

        # --- 8. Package -------------------------------------------------------
        return self._package(partitioned)

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

    @staticmethod
    def _dist2(c1, c2) -> float:
        return sum((x - y) ** 2 for x, y in zip(c1, c2))

    def _make_supercell(
        self,
        unitcell: list,
        dimensions: list,
        center=False,
    ) -> Tuple[list, list]:
        """Tile unitcell into a supercell using the full 3x3 lattice_vectors matrix; results are memoized."""
        cfg = self.config
        lv  = cfg.lattice_vectors   # shape (3, 3); rows = lattice vectors
        p   = cfg.precision

        cache_key = (tuple(dimensions), center)
        if cache_key in self._lattice_cache:
            return self._lattice_cache[cache_key]

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
            self._lattice_cache[cache_key] = (result, central)
            return result, central

        self._lattice_cache[cache_key] = (result, result[0])
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
        self, unitcell: list, center_type: str, rad: float,
        remove_unbonded: bool = False,
    ) -> Tuple[list, list]:
        """Cut a sphere of radius rad from the lattice centered on center_type."""
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
        if remove_unbonded:
            coords = [at[1:4] for at in result]
            tree   = cKDTree(coords)
            cutoff = cfg.bond_cutoff + cfg.tol
            result = [
                at for i, at in enumerate(result)
                if len(tree.query_ball_point(at[1:4], cutoff)) > 1
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

    def _rebalance_mm(self, partitioned: list, target: float) -> list:
        """Distribute charge over under-coordinated MM atoms so ECP+MM sums to target."""
        cfg   = self.config
        p     = cfg.precision
        fracs = self._get_coordinations(partitioned)
        current = sum(at[4] for at in partitioned
                      if at[5] in (REGION_ECP, REGION_MM))
        under = {i for i, (at, f) in enumerate(zip(partitioned, fracs))
                 if at[5] == REGION_MM and f < 1.0 - cfg.tol}
        if not under:
            raise RuntimeError(
                "No under-coordinated MM atoms found for charge balancing. "
                "Try increasing total_layers."
            )
        shift  = (target - current) / len(under)
        result = []
        for i, at in enumerate(partitioned):
            if i in under:
                result.append(at[:4] + [round(at[4] + shift, p)] + at[5:])
            else:
                result.append(at)
        return result

    def _get_shell(self, cluster: list, shell_layers: float) -> list:
        """Return atoms within shell_layers * characteristic_length outward from the surface of cluster."""
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
        """Return all atoms within rad Å of central_atom (origin-based, constant across QM size scaling)."""
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

    def _get_layers(self, cluster: list, n_layers: int) -> Dict[int, list]:
        """Return {layer_index: [atoms]} from bond-layer traversal outward from cluster (layer 0)."""
        cfg    = self.config
        p      = cfg.precision
        cutoff = cfg.bond_cutoff + cfg.tol

        central   = self._find_central_atom(cluster)
        cluster_r = max(self._dist2(central[1:4], at[1:4]) for at in cluster) ** 0.5
        req_rad   = cluster_r + n_layers * cfg.bond_cutoff
        dim       = int(2 * math.ceil(req_rad / cfg.characteristic_length) + 1)
        if dim % 2 == 0:
            dim += 1

        temp_lat, lat_center = self._make_supercell(
            cfg.unitcell, [dim]*3, center=self.center_element
        )
        dx = [central[i+1] - lat_center[i+1] for i in range(3)]
        lattice = [
            [at[0]] + [round(at[i+1] + dx[i], p) for i in range(3)]
            for at in temp_lat
        ]

        lat_coords = [at[1:4] for at in lattice]
        tree       = cKDTree(lat_coords)

        seen: set           = {tuple(round(at[i+1], p) for i in range(3)) for at in cluster}
        result: Dict[int, list] = {0: cluster}

        for i in range(1, n_layers + 1):
            result[i] = []
            prev_coords = [at[1:4] for at in result[i - 1]]
            if not prev_coords:
                break
            for idx_list in tree.query_ball_point(prev_coords, cutoff):
                for idx in idx_list:
                    ct = tuple(round(c, p) for c in lat_coords[idx])
                    if ct not in seen:
                        seen.add(ct)
                        result[i].append(lattice[idx])

        return result

    def _build_ecp_shell(self, qm_raw: list, qm_center: list) -> list:
        """Dispatch ECP shell generation based on self.ecp_shape."""
        cfg = self.config
        if self.ecp_shape == 'shell':
            # Thickness measured from the QM surface, so the ECP stays a
            # constant physical width regardless of QM region size.
            return self._get_shell(qm_raw, shell_layers=self.ecp_layers)

        if self.ecp_shape == 'bonds':
            layers = self._get_layers(qm_raw, n_layers=int(self.ecp_layers))
            return [at for i, layer in layers.items() if i > 0 for at in layer]

        # 'sphere': total sphere radius ecp_layers * characteristic_length from center
        ecp_rad   = self.ecp_layers * cfg.characteristic_length
        sphere    = self._get_sphere_absolute(qm_center, ecp_rad)
        qm_coords = {tuple(round(at[i+1], cfg.precision) for i in range(3)) for at in qm_raw}
        return [at for at in sphere
                if tuple(round(at[i+1], cfg.precision) for i in range(3)) not in qm_coords]

    def _complement_element(self, element: str) -> Optional[str]:
        """Return the first element in the unitcell that is not *element*."""
        for at in self.config.unitcell:
            if at[0] != element:
                return at[0]
        return None

    def _fully_coordinate(
        self, cluster: list, atom_type: Optional[str] = None
    ) -> list:
        """Add bond-level neighbors to fully coordinate cluster boundary; optionally restricted to atom_type."""
        layers    = self._get_layers(cluster, n_layers=1)
        new_atoms = layers.get(1, [])
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

    def _select_defect_index(
        self, partitioned: list, site, element: str
    ) -> Optional[int]:
        """Return index of the QM element atom nearest to site, or the central atom if site is None."""
        candidates = [
            (i, at) for i, at in enumerate(partitioned)
            if at[5] == REGION_QM and at[0] == element
        ]
        if not candidates:
            return None
        if site is None:
            return candidates[0][0]
        target = list(site)
        return min(candidates,
                   key=lambda c: self._dist2(c[1][1:4], target))[0]

    def _apply_vacancy(self, partitioned: list, site=None,
                       element=None) -> list:
        """Convert the selected *element* QM atom to a ghost (region 3)."""
        element = element or self.defect_element
        idx = self._select_defect_index(partitioned, site, element)
        if idx is None:
            warnings.warn(
                f"No '{element}' QM atom found to convert to vacancy.",
                UserWarning,
            )
            return partitioned
        at     = partitioned[idx]
        result = list(partitioned)
        result[idx] = [at[0], at[1], at[2], at[3], 0.0, REGION_GHOST]
        return result

    def _apply_substitutional(self, partitioned: list, site=None,
                              element=None, target=None) -> list:
        """Swap the selected target QM atom for element, inheriting position and updating charge."""
        sub_el  = element or self.substitutional_element
        target  = target or self.defect_element
        charges = self.alt_charges or self.config.canonical_charges

        idx = self._select_defect_index(partitioned, site, target)
        if idx is None:
            warnings.warn(
                f"No '{target}' QM atom found for substitutional replacement.",
                UserWarning,
            )
            return partitioned
        at         = partitioned[idx]
        sub_charge = charges.get(sub_el, at[4])
        result     = list(partitioned)
        result[idx] = [sub_el, at[1], at[2], at[3], sub_charge, REGION_QM]
        return result

    def _apply_interstitial(self, partitioned: list, element, position) -> list:
        """Add element at position as a QM atom with its canonical charge."""
        p       = self.config.precision
        charges = self.alt_charges or self.config.canonical_charges
        pos     = [round(float(c), p) for c in position]

        existing = {tuple(round(at[i+1], p) for i in range(3))
                    for at in partitioned}
        if tuple(pos) in existing:
            raise ValueError(
                f"interstitial_position {pos} coincides with an existing atom."
            )
        q = charges.get(element, 0.0)
        self._qm_charge = round(self._qm_charge + q, 6)
        return partitioned + [[element, pos[0], pos[1], pos[2], q, REGION_QM]]

    def _apply_defect_spec(self, partitioned: list, spec: dict) -> list:
        """Apply one entry from the *defects* list (see the `defects` param)."""
        dtype = spec.get('type')
        if dtype == 'vacancy':
            return self._apply_vacancy(
                partitioned, spec.get('site'), spec.get('element'))
        if dtype == 'substitutional':
            if spec.get('element') is None:
                raise ValueError(
                    "substitutional defect spec requires 'element'."
                )
            return self._apply_substitutional(
                partitioned, spec.get('site'),
                element=spec['element'], target=spec.get('target'))
        if dtype == 'interstitial':
            if spec.get('element') is None or spec.get('position') is None:
                raise ValueError(
                    "interstitial defect spec requires 'element' and "
                    "'position'."
                )
            return self._apply_interstitial(
                partitioned, spec['element'], spec['position'])
        raise ValueError(
            f"Unknown defect spec type '{dtype}'. Supported: "
            "'vacancy', 'substitutional', 'interstitial'."
        )

    def _pyscf_label(self, element: str, region: int) -> str:
        """Map (element, region) to PySCF mol.atom label: QM→'Cu0', ECP→'X-Cu1', ghost→'ghost-Cu', MM→element."""
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
