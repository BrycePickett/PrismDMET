import sys
import numpy as np
from pyscf import gto, scf
sys.path.append('../src')
import local_integrals, dmet, qcdmet_paths
from dmet import make_fragments

# 1. Setup small H2 system
mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74'
mol.basis = 'sto-3g'
mol.build(verbose=0)
mf = scf.RHF(mol).run()

myInts = local_integrals.localintegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
fragments = make_fragments(mol, myInts, [[0, 1]])

print("--- Testing SCmethod='NONE' ---")
d1 = dmet.dmet(myInts, fragments, isTranslationInvariant=False, method='FCI', SCmethod='NONE')
e1 = d1.selfconsistent()
mu1 = d1.mu_imp

print("\n--- Testing oneshot(optimize_mu=True) ---")
d2 = dmet.dmet(myInts, fragments, isTranslationInvariant=False, method='FCI')
e2 = d2.oneshot(optimize_mu=True)
mu2 = d2.mu_imp

print(f"\nResults:")
print(f"Loop One-Shot  : Energy = {e1:12.8f}, Mu = {mu1:12.8f}")
print(f"Explicit One-Shot: Energy = {e2:12.8f}, Mu = {mu2:12.8f}")

assert np.isclose(e1, e2), "Energies do not match"
assert np.isclose(mu1, mu2), "Chemical potentials do not match"
print("\nVerification SUCCESSFUL: oneshot() exactly matches SCmethod='NONE' loop.")
