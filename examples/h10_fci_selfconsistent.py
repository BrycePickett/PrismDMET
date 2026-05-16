"""
h10_fci_selfconsistent.py -- Self-consistent FCI DMET on an H10 ring.

Demonstrates the full self-consistency loop (u-matrix optimization) in
PrismDMET. The H10 ring is divided into 5 fragments of 2 atoms each.
The LSTSQ (least-squares) fitting of the correlation potential is used
to drive self-consistency.

This is the bread-and-butter use case for DMET: iteratively refining
the embedding potential until the fragment and mean-field density
matrices agree.

Usage::

    pip install -e /path/to/PrismDMET
    python h10_fci_selfconsistent.py
"""

import numpy as np
from pyscf import gto, scf
from prismdmet import LocalIntegrals, DMET, make_fragments

# 1. Build H10 ring
nat = 10
bl  = 1.8
r   = 0.5 * bl / np.sin(np.pi / nat)
mol = gto.M(
    atom=[('H', (r * np.cos(i * 2 * np.pi / nat),
                 r * np.sin(i * 2 * np.pi / nat), 0.0)) for i in range(nat)],
    basis='sto-3g', verbose=0,
)

# 2. Mean-field RHF
mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()
print(f"RHF energy = {mf.e_tot:.10f} Ha")

# 3. Localize and fragment (2 atoms per fragment)
my_ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
my_ints.TI_OK = True

atoms_per_imp = 2
atom_groups = [list(range(i, i + atoms_per_imp)) for i in range(0, nat, atoms_per_imp)]
impurity_clusters = make_fragments(mol, my_ints, atom_groups)

# 4. One-shot DMET (for comparison)
print(f"\n{'='*60}")
print("  One-shot FCI DMET on H10")
print(f"{'='*60}")
dmet_oneshot = DMET(my_ints, impurity_clusters, True, method='FCI', sc_method='NONE')
e_oneshot = dmet_oneshot.selfconsistent()
print(f"  One-shot energy  = {e_oneshot:.10f} Ha")

# 5. Self-consistent DMET with LSTSQ
print(f"\n{'='*60}")
print("  Self-consistent FCI DMET on H10 (LSTSQ)")
print(f"{'='*60}")
dmet_sc = DMET(my_ints, impurity_clusters, True, method='FCI', sc_method='LSTSQ')
e_sc = dmet_sc.selfconsistent()
print(f"  Self-consistent energy = {e_sc:.10f} Ha")

# 6. Summary
print(f"\n{'='*60}")
print("  RESULTS SUMMARY")
print(f"{'='*60}")
print(f"  RHF energy             = {mf.e_tot:.10f} Ha")
print(f"  One-shot FCI DMET      = {e_oneshot:.10f} Ha")
print(f"  Self-consistent FCI DMET = {e_sc:.10f} Ha")
print(f"  SC correction          = {(e_sc - e_oneshot)*1000:.4f} mHa")

assert e_sc < mf.e_tot, "Self-consistent DMET energy should be below RHF"
print("\nPASSED: Self-consistent energy is below RHF.")
