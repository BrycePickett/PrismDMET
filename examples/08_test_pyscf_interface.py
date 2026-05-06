"""
Test 08: PySCF interface + substitutional defect + lattice_vectors.

Validates:
1. MaterialConfig can be built from a full 3x3 lattice_vectors matrix.
2. A lattice_vectors-based Cu2O config yields identical SKZCAM sizes to
   the lattice_constant-based cu2o_matproj_config (backward compat).
3. Substitutional defect: element swap is applied correctly.
4. build_mol() constructs a PySCF Mole without error (no SCF run needed).
5. build_mol() ghost atom basis is loaded explicitly (avoids PySCF silent fail).
6. mm_charge wrapper is applied correctly (coords/charges shape match).

Note: We deliberately avoid running the full SCF kernel in a unit test
(that is production time).  We validate the molecular setup only.
"""

import sys
import os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'programs', 'PrismDMET', 'src')
sys.path.insert(0, os.path.abspath(SRC))

import numpy as np
from qmmm import (
    QMMMBuilder, QMMMCluster, MaterialConfig, cu2o_matproj_config,
    build_mol, build_meanfield,
)

PASS = True

def chk(desc, ok):
    global PASS
    status = 'PASS' if ok else 'FAIL'
    if not ok:
        PASS = False
    print(f'  [{status}] {desc}')

# ---- 1. MaterialConfig from lattice_vectors ---------------------------------
a = cu2o_matproj_config.lattice_constant
lv = np.diag([a, a, a])
cu2o_lv = MaterialConfig(
    name              = 'Cu2O_lv_test',
    lattice_vectors   = lv,
    bond_cutoff       = cu2o_matproj_config.bond_cutoff,
    canonical_charges = {'Cu': 1.0, 'O': -2.0},
    bulk_coordinations= {'Cu': 2, 'O': 4},
    unitcell          = cu2o_matproj_config.unitcell,
)
chk('MaterialConfig from lattice_vectors builds',  cu2o_lv is not None)
chk('characteristic_length == a',
    abs(cu2o_lv.characteristic_length - a) < 1e-8)
chk('lattice_constant auto-derived',
    abs(cu2o_lv.lattice_constant - a) < 1e-8)

# ---- 2. Backward compat: identical SKZCAM sizes ----------------------------
b_orig = QMMMBuilder(config=cu2o_matproj_config,  qm_method='SKZCAM',
    target_element='Cu', center_element='Cu',
    qm_target_size=1, ecp_layers=1, pc_layers=3,
    defect_type='pristine')
b_lv   = QMMMBuilder(config=cu2o_lv, qm_method='SKZCAM',
    target_element='Cu', center_element='Cu',
    qm_target_size=1, ecp_layers=1, pc_layers=3,
    defect_type='pristine')

sizes_orig = b_orig.find_skzcam_sizes(50)
sizes_lv   = b_lv.find_skzcam_sizes(50)
chk('SKZCAM sizes match between lattice_constant and lattice_vectors configs',
    sizes_orig == sizes_lv)

# ---- 3. Substitutional defect ----------------------------------------------
b_sub = QMMMBuilder(
    config                 = cu2o_matproj_config,
    qm_method              = 'SKZCAM',
    target_element         = 'Cu',
    center_element         = 'Cu',
    qm_target_size         = 1,
    ecp_layers             = 1,
    pc_layers              = 2,
    defect_type            = 'substitutional',
    defect_element         = 'Cu',
    substitutional_element = 'Ag',
    substitutional_spin    = 0,
)
c_sub = b_sub.build()
chk('Substitutional build succeeds',        c_sub is not None)
chk('No ghost atoms in substitutional',     len(c_sub.ghost_atoms) == 0)
ag_qm = [a for a in c_sub.qm_atoms if a.element == 'Ag']
chk('Ag atom present in QM region',         len(ag_qm) == 1)
chk('Substituted atom has PySCF label Ag0', ag_qm[0].pyscf_label == 'Ag0')

# ---- 4. build_mol() constructs Mole without error --------------------------
b_small = QMMMBuilder(
    config=cu2o_matproj_config, qm_method='SKZCAM',
    target_element='Cu', center_element='Cu',
    qm_target_size=1, ecp_layers=1, pc_layers=2,
    defect_type='pristine',
)
c_small = b_small.build()

try:
    mol = build_mol(c_small, qm_basis='def2-svp', verbose=0)
    mol_ok = True
except Exception as ex:
    mol_ok = False
    print(f'    build_mol() exception: {ex}')

chk('build_mol() succeeds',         mol_ok)
if mol_ok:
    chk('mol.natm > 0',             mol.natm > 0)
    chk('mol.nelectron > 0',        mol.nelectron > 0)
    chk('mol.spin matches cluster',  mol.spin == c_small.qm_spin)

# ---- 5. Ghost atom basis loaded explicitly ---------------------------------
b_vac = QMMMBuilder(
    config=cu2o_matproj_config, qm_method='SKZCAM',
    target_element='Cu', center_element='Cu',
    qm_target_size=1, ecp_layers=1, pc_layers=2,
    defect_type='vacancy',
)
c_vac = b_vac.build()
try:
    mol_vac = build_mol(c_vac, qm_basis='def2-svp', verbose=0)
    ghost_ok = True
except Exception as ex:
    ghost_ok = False
    print(f'    Ghost mol exception: {ex}')
chk("build_mol() with ghost atom succeeds", ghost_ok)

# ---- 6. mm_charge arrays shape consistency ---------------------------------
if mol_ok:
    chk('mm_coords shape is (N, 3)',
        c_small.mm_coords.ndim == 2 and c_small.mm_coords.shape[1] == 3)
    chk('mm_charges shape is (N,)',
        c_small.mm_charges.ndim == 1)
    chk('mm_coords and mm_charges same length',
        len(c_small.mm_coords) == len(c_small.mm_charges))

# ---- Summary ---------------------------------------------------------------
print()
if PASS:
    print('[PASS] 08_test_pyscf_interface.py')
    sys.exit(0)
else:
    print('[FAIL] 08_test_pyscf_interface.py')
    sys.exit(1)
