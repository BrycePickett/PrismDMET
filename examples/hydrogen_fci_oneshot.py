import sys
import numpy as np
from pyscf import gto, scf
import local_integrals, dmet
from dmet import make_fragments

# 1. Setup small H2 system
mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74'
mol.basis = 'sto-3g'
mol.build(verbose=0)
mf = scf.RHF(mol).run()

my_ints = local_integrals.LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
fragments = make_fragments(mol, my_ints, [[0, 1]])

print("--- Testing sc_method='NONE' ---")
d1 = dmet.DMET(my_ints, fragments, is_translation_invariant=False, method='FCI', sc_method='NONE')
e1 = d1.selfconsistent()
mu1 = d1.mu_imp

print("\n--- Testing oneshot(optimize_mu=True) ---")
d2 = dmet.DMET(my_ints, fragments, is_translation_invariant=False, method='FCI')
e2 = d2.oneshot(optimize_mu=True)
mu2 = d2.mu_imp

print(f"\nResults:")
print(f"Loop One-Shot  : Energy = {e1:12.8f}, Mu = {mu1:12.8f}")
print(f"Explicit One-Shot: Energy = {e2:12.8f}, Mu = {mu2:12.8f}")

assert np.isclose(e1, e2), "Energies do not match"
assert np.isclose(mu1, mu2), "Chemical potentials do not match"
print("\nVerification SUCCESSFUL: oneshot() exactly matches sc_method='NONE' loop.")
