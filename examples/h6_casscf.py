"""
h6_casscf.py -- CASSCF one-shot DMET on a 6-atom hydrogen chain.

Demonstrates multi-fragment CASSCF in PrismDMET. The H6 chain is split
into 3 fragments of 2 atoms each.

Usage::

    pip install -e /path/to/PrismDMET
    python h6_casscf.py
"""

import numpy as np
from pyscf import gto, scf
from prismdmet import LocalIntegrals, DMET, make_fragments

# 1. Molecule (H6 chain)
bond = 1.6
nat  = 6
mol = gto.M(
    atom=[('H', (i * bond, 0.0, 0.0)) for i in range(nat)],
    basis='sto-3g', verbose=0,
)

# 2. Mean-field RHF
mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()
print(f"RHF energy = {mf.e_tot:.10f} Ha")

# 3. Localize and fragment
my_ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
atom_groups = [[0, 1], [2, 3], [4, 5]]
impurity_clusters = make_fragments(mol, my_ints, atom_groups)

# 4. CASSCF DMET
n_cas = 4
n_elecas = 4
my_dmet = DMET(
    my_ints, impurity_clusters,
    is_translation_invariant=False,
    method='CASSCF', sc_method='NONE',
    ncas=n_cas, nelecas=n_elecas,
)

# 5. Run one-shot DMET
print(f"\n{'='*60}")
print(f"  One-shot CASSCF({n_cas},{n_elecas}) DMET on H6")
print(f"{'='*60}")

e_prismdmet = my_dmet.oneshot(mu_imp=0.0)
print(f"\nPrismDMET CASSCF Total Energy = {e_prismdmet:.10f} Ha")

# 6. Fragment energies
print("\nFragment Energy Contributions:")
for i, res in enumerate(my_dmet.cas_results):
    print(f"  Fragment {i} Energy = {res['e_imp']:.10f} Ha")

# 7. Sanity check: energy should be below RHF
assert e_prismdmet < mf.e_tot + 0.01, "CASSCF energy should be near or below RHF"
print("\nPASSED: CASSCF energy is physically reasonable.")
