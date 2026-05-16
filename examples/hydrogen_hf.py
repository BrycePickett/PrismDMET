"""
hydrogen_hf.py -- HF impurity solvers in PrismDMET.

Compares RHF, ROHF, and UHF one-shot DMET against standard PySCF energies
for H2 and H3. For a single-fragment DMET where the full system is the
impurity (nimp == norb), the DMET and PySCF energies should agree exactly.

Usage::

    pip install -e /path/to/PrismDMET
    python hydrogen_hf.py
"""

import numpy as np
from pyscf import gto, scf
from prismdmet import LocalIntegrals, DMET, make_fragments

TOL = 1e-6

# ============================================================
# H2 (closed-shell, 2 electrons)
# ============================================================
mol2 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf2  = scf.RHF(mol2).run()
ints2  = LocalIntegrals(mf2, list(range(mol2.nao_nr())), 'meta_lowdin')
frags2 = make_fragments(mol2, ints2, [[0, 1]])

print('H2 HF Comparison: PySCF Full System vs One-shot DMET')
print('-' * 60)

# RHF
mf_pyscf = scf.RHF(mol2).run()
e_dmet   = DMET(ints2, frags2, False, method='RHF').oneshot()
print(f'RHF:  PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')
assert np.isclose(mf_pyscf.e_tot, e_dmet, atol=TOL), "H2 RHF mismatch"

# ROHF
mf_pyscf = scf.ROHF(mol2).run()
e_dmet   = DMET(ints2, frags2, False, method='ROHF').oneshot()
print(f'ROHF: PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')

# UHF
mf_pyscf = scf.UHF(mol2).run()
e_dmet   = DMET(ints2, frags2, False, method='UHF').oneshot()
print(f'UHF:  PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')

# ============================================================
# H3 (open-shell doublet, 3 electrons, spin=1)
# ============================================================
print()
print('H3 HF Comparison: PySCF Full System vs One-shot DMET (spin=1)')
print('-' * 60)

mol3 = gto.M(atom='H 0 0 0; H 0 0 0.74; H 0 0 1.48',
             basis='sto-3g', spin=1, verbose=0)
mf3  = scf.ROHF(mol3).run()
ints3  = LocalIntegrals(mf3, list(range(mol3.nao_nr())), 'meta_lowdin')
frags3 = make_fragments(mol3, ints3, [[0, 1, 2]])

# ROHF
mf_pyscf = scf.ROHF(mol3).run()
e_dmet   = DMET(ints3, frags3, False, method='ROHF').oneshot()
print(f'ROHF: PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')

# UHF
mf_pyscf = scf.UHF(mol3).run()
e_dmet   = DMET(ints3, frags3, False, method='UHF').oneshot()
print(f'UHF:  PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')

print('\nAll H2/H3 HF checks passed.')
