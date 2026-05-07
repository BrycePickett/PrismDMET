"""
hydrogen_hf.py -- HF impurity solvers in PrismDMET.

Compares HF one-shot DMET against standard PySCF energies for H2 and H3.
For a single-fragment DMET where the full system is the impurity (nimp == norb),
the DMET and PySCF energies should agree exactly.

The UHF energy partitioning bug and the odd-electron system blocker in the 
DMET helper have been resolved.

Usage::

    export PYTHONPATH=/path/to/programs/PrismDMET/src
    python hydrogen_hf.py
"""

import sys
from pyscf import gto, scf
import local_integrals, dmet
from dmet import make_fragments

# ============================================================
# H2 (closed-shell, 2 electrons)
# ============================================================
print('H2 HF Comparison: PySCF Full System vs One-shot DMET')
print('-' * 60)

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf  = scf.RHF(mol).run()
ints  = local_integrals.LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0, 1]])

print(f'RHF (PySCF):    {mf.e_tot:.8f} Ha')
mf_uhf = scf.UHF(mol).run()
print(f'UHF (PySCF):    {mf_uhf.e_tot:.8f} Ha')

e_dmet_uhf = dmet.DMET(ints, frags, False, method='UHF').oneshot()
print(f'UHF DMET:       {e_dmet_uhf:.8f} Ha')

# ============================================================
# H3 (open-shell doublet, 3 electrons, spin=1)
# ============================================================
print()
print('H3 HF Comparison: PySCF Full System vs One-shot DMET (spin=1)')
print('-' * 60)

mol3 = gto.M(atom='H 0 0 0; H 0 0 0.74; H 0 0 1.48',
             basis='sto-3g', spin=1, verbose=0)
mf_rohf = scf.ROHF(mol3).run()
mf_uhf3 = scf.UHF(mol3).run()

print(f'ROHF (PySCF):   {mf_rohf.e_tot:.8f} Ha')
print(f'UHF  (PySCF):   {mf_uhf3.e_tot:.8f} Ha')

# Build local integrals from ROHF reference
ints3 = local_integrals.LocalIntegrals(mf_rohf, list(range(mol3.nao_nr())), 'meta_lowdin')
frags3 = make_fragments(mol3, ints3, [[0, 1, 2]])

# Run UHF DMET one-shot
e_dmet_uhf3 = dmet.DMET(ints3, frags3, False, method='UHF').oneshot()
print(f'UHF DMET:       {e_dmet_uhf3:.8f} Ha')

# Run ROHF DMET one-shot
e_dmet_rohf3 = dmet.DMET(ints3, frags3, False, method='ROHF').oneshot()
print(f'ROHF DMET:      {e_dmet_rohf3:.8f} Ha')
