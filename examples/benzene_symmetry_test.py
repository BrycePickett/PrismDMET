"""
Prismdmet :: Symmetry Test (H6 ring, sto-3g, CC)
=================================================
Demonstrates the use_symmetry feature on a H6 ring.
With 6 equivalent H fragments, symmetry detection solves only
1 unique fragment and copies results to the other 5.

Comparison: both runs should yield numerically identical energies.

Usage:
    python benzene_symmetry_test.py
"""

import time
import numpy as np
import local_integrals, dmet
from dmet import make_fragments
from pyscf import gto, scf

# ---- Build H6 ring ----
nat = 6
bl  = 1.8  # Angstrom
r   = 0.5 * bl / np.sin(np.pi / nat)
mol = gto.Mole()
mol.atom = [('H', (r * np.cos(i * 2 * np.pi / nat),
                   r * np.sin(i * 2 * np.pi / nat),
                   0.0)) for i in range(nat)]
mol.basis = 'sto-3g'
mol.build(verbose=0)

# ---- Mean-field ----
mf = scf.RHF(mol)
mf.verbose = 0
mf.scf()
print(f"RHF energy = {mf.e_tot:.10f}")

# ---- Localize (meta-lowdin is robust for all basis sets) ----
my_ints = local_integrals.LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
my_ints.TI_OK = True  # H6 ring is translationally invariant

# ---- 1 atom per fragment ----
atom_groups = [[i] for i in range(nat)]
impurity_clusters = make_fragments(mol, my_ints, atom_groups)

# ---- Run WITHOUT symmetry ----
print("\n" + "="*60)
print("  Run 1: NO symmetry (sequential)")
print("="*60)
t0 = time.time()
thedmet_nosym = dmet.DMET(my_ints, impurity_clusters, False,
                           method='CC', sc_method='NONE',
                           use_symmetry=False)
e_nosym = thedmet_nosym.selfconsistent()
t_nosym = time.time() - t0
print(f"\n  Energy (no symmetry) = {e_nosym:.10f}")
print(f"  Wall time            = {t_nosym:.2f} s")

# ---- Run WITH symmetry (auto-detect) ----
print("\n" + "="*60)
print("  Run 2: WITH symmetry auto-detection")
print("="*60)
t0 = time.time()
thedmet_sym = dmet.DMET(my_ints, impurity_clusters, False,
                         method='CC', sc_method='NONE',
                         use_symmetry=True)
e_sym = thedmet_sym.selfconsistent()
t_sym = time.time() - t0
print(f"\n  Energy (symmetry)    = {e_sym:.10f}")
print(f"  Wall time            = {t_sym:.2f} s")

# ---- Run WITH user-provided symmetry_map ----
print("\n" + "="*60)
print("  Run 3: WITH user-provided symmetry_map")
print("="*60)
# Explicitly tell Prismdmet that fragments 1-5 are equivalent to fragment 0
user_map = {i: 0 for i in range(1, nat)}
t0 = time.time()
thedmet_usermap = dmet.DMET(my_ints, impurity_clusters, False,
                              method='CC', sc_method='NONE',
                              use_symmetry=True, symmetry_map=user_map)
e_usermap = thedmet_usermap.selfconsistent()
t_usermap = time.time() - t0
print(f"\n  Energy (user map)    = {e_usermap:.10f}")
print(f"  Wall time            = {t_usermap:.2f} s")

# ---- Summary ----
print("\n" + "="*60)
print("  RESULTS SUMMARY")
print("="*60)
diff_auto = abs(e_nosym - e_sym)
diff_user = abs(e_nosym - e_usermap)
print(f"  E (no symmetry)  = {e_nosym:.10f}")
print(f"  E (auto-detect)  = {e_sym:.10f}   |diff| = {diff_auto:.2e}")
print(f"  E (user map)     = {e_usermap:.10f}   |diff| = {diff_user:.2e}")
if t_nosym > 0 and t_sym > 0:
    print(f"  Speedup (auto)   = {t_nosym / t_sym:.1f}x")
    print(f"  Speedup (user)   = {t_nosym / t_usermap:.1f}x")

passed = diff_auto < 1e-9 and diff_user < 1e-9
print(f"\n  {'✓ PASSED' if passed else '✗ FAILED'}: Energy agreement within 1e-9 Ha")
