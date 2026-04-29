"""
NEVPT2 one-shot DMET on an N2 molecule.

This example demonstrates the built-in PySCF-based NEVPT2 solver
for PrismDMET. The solver automatically handles SA-CASSCF and subsequent
NEVPT2 correlation on the fragments.

Run from this directory with the prismdmet conda environment active:
    python 03_n2_nevpt2.py
"""

import sys
import numpy as np

import local_integrals
from dmet import dmet, make_fragments
from pyscf import gto, scf

# ---------------------------------------------------------------------------
# 1. Molecule Setup (N2)
# ---------------------------------------------------------------------------
mol = gto.Mole()
mol.atom = [['N', (0, 0, 0)], ['N', (0, 0, 1.5)]]
mol.basis = 'sto-3g'
mol.spin = 0
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
myInts.TI_OK = False

# Group atoms into 2 fragments (one for each Nitrogen)
atom_groups = [[0], [1]]
impurityClusters = make_fragments(mol, myInts, atom_groups)

# ---------------------------------------------------------------------------
# 4. DMET NEVPT2 Setup
# ---------------------------------------------------------------------------
NCAS = 4     # active orbitals per fragment
NELECAS = 4  # active electrons per fragment

mydmet = dmet(
    myInts, 
    impurityClusters, 
    isTranslationInvariant=False,
    method='NEVPT2', 
    mf_real=mf,           # Required for NEVPT2 methods
    SCmethod='NONE',      # NEVPT2 only supports one-shot DMET
    ncas=NCAS, 
    nelecas=NELECAS,
    sa_nstates=1          # Single state (default)
)

# ---------------------------------------------------------------------------
# 5. Run one-shot DMET
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print(f"  One-shot NEVPT2({NCAS},{NELECAS}) DMET on N2")
print("="*60)

e_prismdmet = mydmet.oneshot(mu_imp=0.0)

print(f"\nPrismDMET NEVPT2 Total Energy = {e_prismdmet:.10f} Ha")

# ---------------------------------------------------------------------------
# 6. View Fragment Energies
# ---------------------------------------------------------------------------
if mydmet.nevpt2_results:
    print("\nFragment Energy Contributions:")
    for i, res in enumerate(mydmet.nevpt2_results):
        print(f"  Fragment {i}:")
        print(f"    E_tot  = {res['e_tot'][0]:.10f} Ha")
        print(f"    E_corr = {res['e_corr'][0]:.10f} Ha")
