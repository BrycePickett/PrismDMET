"""
hydrogen_dft.py -- DFT impurity solvers in PrismDMET (H2).

Compares RKS and UKS one-shot DMET on the H2 molecule, treating the full
system as a single impurity fragment, and compares against standard PySCF DFT.

Usage::

    export PYTHONPATH=/path/to/programs/PrismDMET/src
    python hydrogen_dft.py
"""

from pyscf import gto, scf, dft
import local_integrals, dmet
from dmet import make_fragments

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf  = scf.RHF(mol).run()

ints = local_integrals.LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0, 1]])

print('H2 DFT Comparison: PySCF Full System vs One-shot DMET')
print('-' * 60)

# PBE RKS
mf_pyscf = dft.RKS(mol)
mf_pyscf.xc = 'pbe'
e_pyscf = mf_pyscf.run().e_tot
d = dmet.DMET(ints, frags, False, method='RKS', xc='pbe')
e_dmet = d.oneshot()
print(f'RKS (PBE):   PySCF = {e_pyscf:.8f} Ha | DMET = {e_dmet:.8f} Ha')

# PBE UKS
mf_pyscf = dft.UKS(mol)
mf_pyscf.xc = 'pbe'
e_pyscf = mf_pyscf.run().e_tot
d = dmet.DMET(ints, frags, False, method='UKS', xc='pbe')
e_dmet = d.oneshot()
print(f'UKS (PBE):   PySCF = {e_pyscf:.8f} Ha | DMET = {e_dmet:.8f} Ha')

# B3LYP RKS
mf_pyscf = dft.RKS(mol)
mf_pyscf.xc = 'b3lyp'
e_pyscf = mf_pyscf.run().e_tot
d = dmet.DMET(ints, frags, False, method='RKS', xc='b3lyp')
e_dmet = d.oneshot()
print(f'RKS (B3LYP): PySCF = {e_pyscf:.8f} Ha | DMET = {e_dmet:.8f} Ha')

# B3LYP UKS
mf_pyscf = dft.UKS(mol)
mf_pyscf.xc = 'b3lyp'
e_pyscf = mf_pyscf.run().e_tot
d = dmet.DMET(ints, frags, False, method='UKS', xc='b3lyp')
e_dmet = d.oneshot()
print(f'UKS (B3LYP): PySCF = {e_pyscf:.8f} Ha | DMET = {e_dmet:.8f} Ha')
