"""
nitrogen_nevpt2.py -- NEVPT2 one-shot DMET on an N2 molecule.

Demonstrates the built-in PySCF-based NEVPT2 solver for PrismDMET.
The solver handles SA-CASSCF and subsequent NEVPT2 correlation on
each fragment.

Usage::

    pip install -e /path/to/PrismDMET
    python nitrogen_nevpt2.py
"""

import numpy as np
from pyscf import gto, scf
from prismdmet import LocalIntegrals, DMET, make_fragments

# 1. Molecule (N2)
mol = gto.M(
    atom=[['N', (0, 0, 0)], ['N', (0, 0, 1.5)]],
    basis='sto-3g', spin=0, verbose=0,
)

# 2. Mean-field RHF
mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()
print(f"RHF energy = {mf.e_tot:.10f} Ha")

# 3. Localize and fragment (one atom per fragment)
my_ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
my_ints.TI_OK = False
atom_groups = [[0], [1]]
impurity_clusters = make_fragments(mol, my_ints, atom_groups)

# 4. DMET NEVPT2 setup
n_cas = 4
n_elecas = 4
my_dmet = DMET(
    my_ints, impurity_clusters,
    is_translation_invariant=False,
    method='NEVPT2',
    sc_method='NONE',
    ncas=n_cas, nelecas=n_elecas,
    sa_nstates=1,
)

# 5. Run one-shot DMET
print(f"\n{'='*60}")
print(f"  One-shot NEVPT2({n_cas},{n_elecas}) DMET on N2")
print(f"{'='*60}")

e_prismdmet = my_dmet.oneshot(mu_imp=0.0)
print(f"\nPrismDMET NEVPT2 Total Energy = {e_prismdmet:.10f} Ha")

# 6. Fragment energies
if my_dmet.nevpt2_results:
    print("\nFragment Energy Contributions:")
    for i, res in enumerate(my_dmet.nevpt2_results):
        print(f"  Fragment {i}:")
        print(f"    E_tot  = {res['e_tot'][0]:.10f} Ha")
        print(f"    E_corr = {res['e_corr'][0]:.10f} Ha")

assert e_prismdmet < mf.e_tot + 0.01, "NEVPT2 energy should be near or below RHF"
print("\nPASSED: NEVPT2 energy is physically reasonable.")
