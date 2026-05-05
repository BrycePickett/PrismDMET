"""
CASSCF one-shot dmet on a 6-atom Hydrogen chain (H6).

This example demonstrates how to run the CASSCF solver in Prismdmet.
The molecule is split into 3 fragments of 2 atoms each.

Run from this directory with the prismdmet conda environment active:
    python 02_h6_casscf.py
"""

import sys
import numpy as np

import local_integrals
from dmet import dmet, make_fragments
from pyscf import gto, scf

# ---------------------------------------------------------------------------
# 1. Molecule Setup (H6 chain)
# ---------------------------------------------------------------------------
bond = 1.6
nat  = 6

mol = gto.Mole()
mol.atom = [('H', (i * bond, 0.0, 0.0)) for i in range(nat)]
mol.basis = 'sto-3g'
mol.verbose = 0
mol.build()

# ---------------------------------------------------------------------------
# 2. Mean-field RHF
# ---------------------------------------------------------------------------
mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()
print(f"RHF energy = {mf.e_tot:.10f} Ha")

# ---------------------------------------------------------------------------
# 3. Local integrals and Fragments
# ---------------------------------------------------------------------------
my_ints = local_integrals.local_integrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')

# Group atoms into 3 fragments (0,1), (2,3), (4,5)
atom_groups = [[0, 1], [2, 3], [4, 5]]
impurity_clusters = make_fragments(mol, my_ints, atom_groups)

# ---------------------------------------------------------------------------
# 4. dmet CASSCF Setup
# ---------------------------------------------------------------------------
n_cas = 4     # active orbitals per fragment
n_elecas = 4  # active electrons per fragment

my_dmet = dmet(
    my_ints, 
    impurity_clusters, 
    isTranslationInvariant=False,
    method='CASSCF', 
    SCmethod='NONE',  # one-shot
    ncas=n_cas, 
    nelecas=n_elecas
)

# ---------------------------------------------------------------------------
# 5. Run one-shot dmet
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print(f"  One-shot CASSCF({n_cas},{n_elecas}) dmet on H6")
print("="*60)

e_prismdmet = my_dmet.oneshot(mu_imp=0.0)

print(f"\nPrismdmet CASSCF Total Energy = {e_prismdmet:.10f} Ha")

# ---------------------------------------------------------------------------
# 6. View Fragment Energies
# ---------------------------------------------------------------------------
print("\nFragment Energy Contributions:")
for i, res in enumerate(my_dmet.cas_results):
    print(f"  Fragment {i} Energy = {res['e_imp']:.10f} Ha")
