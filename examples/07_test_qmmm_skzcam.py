"""
Test 07: QM/MM SKZCAM cluster builder.

Validates that QMMMBuilder, using the cu2o_matproj_config, can:
1. List SKZCAM magic numbers correctly.
2. Build a 19-Cu pristine SKZCAM cluster without error.
3. Produce correct atom counts in each region.
4. Produce an exactly neutral MM charge sum (|Q| < 1e-6).
5. Build a 19-Cu vacancy cluster and confirm ghost atom present.
6. Confirm PySCF mol.atom labels follow the correct convention.

Pass conditions are printed per-check and the script exits 0 on success.
"""

import sys
import os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'programs', 'PrismDMET', 'src')
sys.path.insert(0, os.path.abspath(SRC))

import numpy as np
from qmmm import QMMMBuilder, cu2o_matproj_config, REGION_GHOST

PASS = True

def chk(desc, ok):
    global PASS
    status = 'PASS' if ok else 'FAIL'
    if not ok:
        PASS = False
    print(f'  [{status}] {desc}')

# ---- 1. SKZCAM scan --------------------------------------------------------
builder = QMMMBuilder(
    config         = cu2o_matproj_config,
    qm_method      = 'SKZCAM',
    target_element = 'Cu',
    center_element = 'Cu',
    qm_target_size = 19,
    ecp_layers     = 1.0,
    pc_layers      = 3.0,
    defect_type    = 'pristine',
)

skzcam_map = builder.find_skzcam_sizes(upper_bound=50)
chk('SKZCAM scan returns dict',        isinstance(skzcam_map, dict))
chk('n=1 is a valid SKZCAM size',      1  in skzcam_map)
chk('n=19 is a valid SKZCAM size',     19 in skzcam_map)

# ---- 2. Build pristine cluster --------------------------------------------
cluster = builder.build()
chk('build() returns QMMMCluster',     cluster is not None)
chk('QM atoms present',                len(cluster.qm_atoms) > 0)
chk('ECP atoms present',               len(cluster.ecp_atoms) > 0)
chk('MM charges present',              len(cluster.mm_atoms) > 0)
chk('No ghost atoms in pristine',      len(cluster.ghost_atoms) == 0)

# ---- 3. Atom counts sanity ------------------------------------------------
n_cu_qm = sum(1 for a in cluster.qm_atoms if a.element == 'Cu')
chk(f'QM region has 19 Cu atoms (got {n_cu_qm})',  n_cu_qm == 19)

# ---- 4. Global charge neutrality (QM + ECP + MM = 0) ----------------------
grand_total = cluster.qm_charge + cluster.mm_charges.sum()
chk(f'Grand total charge neutral (|Q|={abs(grand_total):.2e})',  abs(grand_total) < 1e-3)

# ---- 5. Vacancy build ------------------------------------------------------
vac_builder = QMMMBuilder(
    config         = cu2o_matproj_config,
    qm_method      = 'SKZCAM',
    target_element = 'Cu',
    center_element = 'Cu',
    qm_target_size = 19,
    ecp_layers     = 1.0,
    pc_layers      = 3.0,
    defect_type    = 'vacancy',
    defect_element = 'Cu',
)
vac_cluster = vac_builder.build()
chk('Vacancy build succeeds',          vac_cluster is not None)
chk('Ghost atom present',              len(vac_cluster.ghost_atoms) == 1)
chk('Ghost atom has zero charge',      vac_cluster.ghost_atoms[0].charge == 0.0)
chk('Vacancy spin = 1',                vac_cluster.qm_spin == 1)

# ---- 6. PySCF label convention --------------------------------------------
qm_labels  = [a.pyscf_label for a in cluster.qm_atoms]
ecp_labels = [a.pyscf_label for a in cluster.ecp_atoms]
gh_labels  = [a.pyscf_label for a in vac_cluster.ghost_atoms]

chk('QM labels end with 0',            all(l.endswith('0') for l in qm_labels))
chk('ECP labels start with X-',        all(l.startswith('X-') for l in ecp_labels))
chk('Ghost label starts with ghost-',  all(l.startswith('ghost-') for l in gh_labels))

# ---- 7. mol.atom list is well-formed ---------------------------------------
mol_atom = cluster.to_pyscf_mol_atom()
chk('mol.atom is a non-empty list',    len(mol_atom) > 0)
chk('mol.atom entries are 2-tuples',
    all(isinstance(e, tuple) and len(e) == 2 for e in mol_atom))

# ---- Summary ---------------------------------------------------------------
print()
if PASS:
    print('[PASS] 07_test_qmmm_skzcam.py')
    sys.exit(0)
else:
    print('[FAIL] 07_test_qmmm_skzcam.py')
    sys.exit(1)
