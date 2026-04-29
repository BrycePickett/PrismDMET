"""
CASSCF one-shot DMET on a 6-atom Hydrogen chain (H6).

This example demonstrates how to run the CASSCF solver in PrismDMET.
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
BOND = 1.6
NAT  = 6

mol = gto.Mole()
mol.atom = [('H', (i * BOND, 0.0, 0.0)) for i in range(NAT)]
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
myInts = local_integrals.localintegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')

# Group atoms into 3 fragments (0,1), (2,3), (4,5)
atom_groups = [[0, 1], [2, 3], [4, 5]]
impurityClusters = make_fragments(mol, myInts, atom_groups)

# ---------------------------------------------------------------------------
# 4. DMET CASSCF Setup
# ---------------------------------------------------------------------------
NCAS = 4     # active orbitals per fragment
NELECAS = 4  # active electrons per fragment

mydmet = dmet(
    myInts, 
    impurityClusters, 
    isTranslationInvariant=False,
    method='CASSCF', 
    SCmethod='NONE',  # one-shot
    ncas=NCAS, 
    nelecas=NELECAS
)

# ---------------------------------------------------------------------------
# 5. Run one-shot DMET
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print(f"  One-shot CASSCF({NCAS},{NELECAS}) DMET on H6")
print("="*60)

e_prismdmet = mydmet.oneshot(mu_imp=0.0)

print(f"\nPrismDMET CASSCF Total Energy = {e_prismdmet:.10f} Ha")

# ---------------------------------------------------------------------------
# 6. View Fragment Energies
# ---------------------------------------------------------------------------
print("\nFragment Energy Contributions:")
for i, res in enumerate(mydmet.cas_results):
    print(f"  Fragment {i} Energy = {res['e_imp']:.10f} Ha")
