"""
hydrogen_fci_oneshot.py -- One-shot FCI DMET on H2.

Demonstrates two equivalent ways to run a one-shot (non-self-consistent) DMET
calculation: (1) via selfconsistent() with sc_method='NONE', and (2) via the
explicit oneshot() method. Both should yield identical results.

Usage::

    pip install -e /path/to/PrismDMET
    python hydrogen_fci_oneshot.py
"""

import numpy as np
from pyscf import gto, scf
from prismdmet import LocalIntegrals, DMET, make_fragments

# 1. Build H2 molecule
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf = scf.RHF(mol).run()

# 2. Localize and define a single fragment covering all atoms
my_ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
fragments = make_fragments(mol, my_ints, [[0, 1]])

# 3. Method A: selfconsistent() with sc_method='NONE' (equivalent to one-shot)
d1 = DMET(my_ints, fragments, is_translation_invariant=False, method='FCI', sc_method='NONE')
e1 = d1.selfconsistent()
mu1 = d1.mu_imp

# 4. Method B: explicit oneshot() call
d2 = DMET(my_ints, fragments, is_translation_invariant=False, method='FCI')
e2 = d2.oneshot(optimize_mu=True)
mu2 = d2.mu_imp

# 5. Verify
print(f"\nResults:")
print(f"  sc_method='NONE' : Energy = {e1:12.8f}, Mu = {mu1:12.8f}")
print(f"  oneshot()        : Energy = {e2:12.8f}, Mu = {mu2:12.8f}")

assert np.isclose(e1, e2), f"Energies differ: {e1} vs {e2}"
assert np.isclose(mu1, mu2), f"Chemical potentials differ: {mu1} vs {mu2}"
print("\nPASSED: oneshot() exactly matches sc_method='NONE' loop.")
