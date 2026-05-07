"""
hydrogen_dft.py -- DFT impurity solvers in PrismDMET (H2).

Compares RKS and UKS one-shot DMET on the H2 molecule, treating the full
system as a single impurity fragment.

Usage::

    export PYTHONPATH=/path/to/programs/PrismDMET/src
    python hydrogen_dft.py
"""

from pyscf import gto, scf
import local_integrals, dmet
from dmet import make_fragments

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf  = scf.RHF(mol).run()

ints = local_integrals.LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0, 1]])

print('H2 one-shot DMET -- RKS and UKS with PBE')
print('-' * 45)

d = dmet.DMET(ints, frags, False, method='RKS', xc='pbe')
print(f'  RKS (PBE):  {d.oneshot():.8f} Ha')

d = dmet.DMET(ints, frags, False, method='UKS', xc='pbe')
print(f'  UKS (PBE):  {d.oneshot():.8f} Ha')

d = dmet.DMET(ints, frags, False, method='RKS', xc='b3lyp')
print(f'  RKS (B3LYP): {d.oneshot():.8f} Ha')

d = dmet.DMET(ints, frags, False, method='UKS', xc='b3lyp')
print(f'  UKS (B3LYP): {d.oneshot():.8f} Ha')
