"""
h6_symmetry_test.py -- Fragment symmetry detection on an H6 ring.

Demonstrates the use_symmetry feature. With 6 equivalent H fragments,
symmetry detection solves only 1 unique fragment and copies results to
the other 5. All three runs should yield numerically identical energies.

Usage::

    pip install -e /path/to/PrismDMET
    python h6_symmetry_test.py
"""

import time
import numpy as np
from pyscf import gto, scf
from prismdmet import LocalIntegrals, DMET, make_fragments

# Build H6 ring
nat = 6
bl  = 1.8
r   = 0.5 * bl / np.sin(np.pi / nat)
mol = gto.M(
    atom=[('H', (r * np.cos(i * 2 * np.pi / nat),
                 r * np.sin(i * 2 * np.pi / nat), 0.0)) for i in range(nat)],
    basis='sto-3g', verbose=0,
)

mf = scf.RHF(mol)
mf.verbose = 0
mf.scf()
print(f"RHF energy = {mf.e_tot:.10f}")

my_ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
my_ints.TI_OK = True
atom_groups = [[i] for i in range(nat)]
impurity_clusters = make_fragments(mol, my_ints, atom_groups)

# Run 1: No symmetry
t0 = time.time()
d1 = DMET(my_ints, impurity_clusters, False, method='CC', sc_method='NONE', use_symmetry=False)
e_nosym = d1.selfconsistent()
t_nosym = time.time() - t0
print(f"\n  E (no symmetry)  = {e_nosym:.10f}  ({t_nosym:.2f} s)")

# Run 2: Auto-detect symmetry
t0 = time.time()
d2 = DMET(my_ints, impurity_clusters, False, method='CC', sc_method='NONE', use_symmetry=True)
e_sym = d2.selfconsistent()
t_sym = time.time() - t0
print(f"  E (auto-detect)  = {e_sym:.10f}  ({t_sym:.2f} s)")

# Run 3: User-provided symmetry map
user_map = {i: 0 for i in range(1, nat)}
t0 = time.time()
d3 = DMET(my_ints, impurity_clusters, False, method='CC', sc_method='NONE',
          use_symmetry=True, symmetry_map=user_map)
e_usermap = d3.selfconsistent()
t_usermap = time.time() - t0
print(f"  E (user map)     = {e_usermap:.10f}  ({t_usermap:.2f} s)")

# Verify
diff_auto = abs(e_nosym - e_sym)
diff_user = abs(e_nosym - e_usermap)
passed = diff_auto < 1e-9 and diff_user < 1e-9
print(f"\n  |diff auto| = {diff_auto:.2e}")
print(f"  |diff user| = {diff_user:.2e}")
if t_nosym > 0 and t_sym > 0:
    print(f"  Speedup (auto)  = {t_nosym / t_sym:.1f}x")
print(f"\n  {'PASSED' if passed else 'FAILED'}: Energy agreement within 1e-9 Ha")
assert passed
