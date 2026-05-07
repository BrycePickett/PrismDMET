"""
hydrogen_dft.py -- DFT impurity solvers in PrismDMET (H2).

Compares RKS, UKS, and ROKS one-shot DMET on the H2 molecule against the
corresponding standard PySCF DFT energies.  For a single-fragment DMET where
the full system is the impurity (nimp == norb), the two should agree exactly.

Usage::

    export PYTHONPATH=/path/to/programs/PrismDMET/src
    python hydrogen_dft.py
"""

from pyscf import gto, scf, dft
import local_integrals, dmet
from dmet import make_fragments

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf  = scf.RHF(mol).run()

ints  = local_integrals.LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0, 1]])

print('H2 DFT Comparison: PySCF Full System vs One-shot DMET')
print('-' * 60)

# PBE RKS
mf_pyscf = dft.RKS(mol, xc='pbe').run()
e_dmet   = dmet.DMET(ints, frags, False, method='RKS', xc='pbe').oneshot()
print(f'RKS  (PBE):   PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')

# PBE UKS
mf_pyscf = dft.UKS(mol, xc='pbe').run()
e_dmet   = dmet.DMET(ints, frags, False, method='UKS', xc='pbe').oneshot()
print(f'UKS  (PBE):   PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')

# PBE ROKS (same as RKS for closed-shell H2; spin=0)
mf_pyscf = dft.ROKS(mol, xc='pbe').run()
e_dmet   = dmet.DMET(ints, frags, False, method='ROKS', xc='pbe').oneshot()
print(f'ROKS (PBE):   PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')

# B3LYP RKS
mf_pyscf = dft.RKS(mol, xc='b3lyp').run()
e_dmet   = dmet.DMET(ints, frags, False, method='RKS', xc='b3lyp').oneshot()
print(f'RKS  (B3LYP): PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')

# B3LYP UKS
mf_pyscf = dft.UKS(mol, xc='b3lyp').run()
e_dmet   = dmet.DMET(ints, frags, False, method='UKS', xc='b3lyp').oneshot()
print(f'UKS  (B3LYP): PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')

# B3LYP ROKS
mf_pyscf = dft.ROKS(mol, xc='b3lyp').run()
e_dmet   = dmet.DMET(ints, frags, False, method='ROKS', xc='b3lyp').oneshot()
print(f'ROKS (B3LYP): PySCF = {mf_pyscf.e_tot:.8f} Ha | DMET = {e_dmet:.8f} Ha')
