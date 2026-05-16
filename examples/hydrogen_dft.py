"""
hydrogen_dft.py -- DFT impurity solvers in PrismDMET.

Compares RKS, ROKS, and UKS one-shot DMET against standard PySCF energies
for H2 and H3. For a single-fragment DMET where the full system is the
impurity (nimp == norb), the DMET and PySCF energies should agree exactly.

Usage::

    pip install -e /path/to/PrismDMET
    python hydrogen_dft.py
"""

import numpy as np
from pyscf import gto, scf, dft
from prismdmet import LocalIntegrals, DMET, make_fragments

# ============================================================
# H2 (closed-shell, 2 electrons)
# ============================================================
mol2 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf2  = scf.RHF(mol2).run()
ints2  = LocalIntegrals(mf2, list(range(mol2.nao_nr())), 'meta_lowdin')
frags2 = make_fragments(mol2, ints2, [[0, 1]])

print('H2 DFT Comparison: PySCF Full System vs One-shot DMET (PBE)')
print('-' * 60)

# RKS
mf_pyscf = dft.RKS(mol2, xc='pbe').run()
e_dmet   = DMET(ints2, frags2, False, method='RKS', xc='pbe').oneshot()
print(f'RKS:  PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')

# ROKS
mf_pyscf = dft.ROKS(mol2, xc='pbe').run()
e_dmet   = DMET(ints2, frags2, False, method='ROKS', xc='pbe').oneshot()
print(f'ROKS: PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')

# UKS
mf_pyscf = dft.UKS(mol2, xc='pbe').run()
e_dmet   = DMET(ints2, frags2, False, method='UKS', xc='pbe').oneshot()
print(f'UKS:  PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')

# ============================================================
# H3 (open-shell doublet, 3 electrons, spin=1)
# ============================================================
print()
print('H3 DFT Comparison: PySCF Full System vs One-shot DMET (PBE, spin=1)')
print('-' * 60)

mol3 = gto.M(atom='H 0 0 0; H 0 0 0.74; H 0 0 1.48',
             basis='sto-3g', spin=1, verbose=0)
mf3  = scf.ROHF(mol3).run()
ints3  = LocalIntegrals(mf3, list(range(mol3.nao_nr())), 'meta_lowdin')
frags3 = make_fragments(mol3, ints3, [[0, 1, 2]])

# ROKS
mf_pyscf = dft.ROKS(mol3, xc='pbe').run()
e_dmet   = DMET(ints3, frags3, False, method='ROKS', xc='pbe').oneshot()
print(f'ROKS: PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')

# UKS
mf_pyscf = dft.UKS(mol3, xc='pbe').run()
e_dmet   = DMET(ints3, frags3, False, method='UKS', xc='pbe').oneshot()
print(f'UKS:  PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')

print('\nAll H2/H3 DFT checks passed.')
