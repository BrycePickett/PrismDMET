'''
Example: Comparing RKS, ROKS, and UKS DMET solvers on Hydrogen systems.

This script demonstrates how to use the DFT solvers (RKS, ROKS, UKS) in PrismDMET.
We run one-shot DMET on a closed-shell H2 and an open-shell H3 chain, treating 
each full molecule as a single impurity fragment to compare the embedding solvers.
'''

import numpy as np
from pyscf import gto, scf
import local_integrals, dmet
from dmet import make_fragments

print("=" * 60)
print("1. Closed-Shell System (H2, 2 electrons)")
print("=" * 60)

mol_h2 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf_h2 = scf.RHF(mol_h2).run()
ints_h2 = local_integrals.LocalIntegrals(mf_h2, list(range(mol_h2.nao_nr())), 'meta_lowdin')
frag_h2 = make_fragments(mol_h2, ints_h2, [[0, 1]])

print("--- Running RKS (PBE) ---")
d_rks = dmet.DMET(ints_h2, frag_h2, False, method='RKS', xc='pbe')
e_rks = d_rks.oneshot()
print(f"RKS Energy: {e_rks:.8f} Ha\n")

print("--- Running UKS (PBE) [Spin-summed matching] ---")
d_uks = dmet.DMET(ints_h2, frag_h2, False, method='UKS', xc='pbe', spin_polarized=False)
e_uks = d_uks.oneshot()
print(f"UKS Energy: {e_uks:.8f} Ha\n")

print("--- Running UKS (PBE) [Spin-polarized alpha/beta matching] ---")
d_uks_pol = dmet.DMET(ints_h2, frag_h2, False, method='UKS', xc='pbe', spin_polarized=True)
e_uks_pol = d_uks_pol.oneshot(mu_imp=[0.0, 0.0], optimize_mu=True)
print(f"UKS (Pol) Energy: {e_uks_pol:.8f} Ha\n")


print("=" * 60)
print("2. Open-Shell System (H3 chain, 3 electrons, spin=1)")
print("=" * 60)

mol_h3 = gto.M(atom='H 0 0 0; H 0 0 0.74; H 0 0 1.48', basis='sto-3g', spin=1, verbose=0)
mf_h3 = scf.ROHF(mol_h3).run()
ints_h3 = local_integrals.LocalIntegrals(mf_h3, list(range(mol_h3.nao_nr())), 'meta_lowdin')
frag_h3 = make_fragments(mol_h3, ints_h3, [[0, 1, 2]])

print("--- Running ROKS (B3LYP) ---")
d_roks = dmet.DMET(ints_h3, frag_h3, False, method='ROKS', xc='b3lyp')
e_roks = d_roks.oneshot()
print(f"ROKS Energy: {e_roks:.8f} Ha\n")

print("--- Running UKS (B3LYP) ---")
d_uks_h3 = dmet.DMET(ints_h3, frag_h3, False, method='UKS', xc='b3lyp')
e_uks_h3 = d_uks_h3.oneshot()
print(f"UKS Energy: {e_uks_h3:.8f} Ha\n")
