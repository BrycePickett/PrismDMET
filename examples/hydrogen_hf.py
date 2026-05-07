"""
hydrogen_hf.py -- HF impurity solvers in PrismDMET.

Compares HF one-shot DMET against standard PySCF energies for H2 and H3.
For a single-fragment DMET where the full system is the impurity (nimp == norb),
the DMET and PySCF energies should agree exactly.

Current status
--------------
  - UHF H2: known energy discrepancy under investigation (pre-existing issue
    in the uhf.py energy partitioning formula).
  - H3 (3 electrons): the DMET chemical-potential optimizer in
    prismdmet_helper.py requires an even electron count; open-shell systems
    with an odd number of electrons are not yet supported by the default
    DMET helper.  Use spin_polarized=True with a UHF reference and the
    modified helper when available.

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
print(f'UHF DMET:       {e_dmet_uhf:.8f} Ha  (discrepancy under investigation)')

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
print('DMET: odd-electron open-shell systems are not yet supported by the',
      'default DMET chemical-potential optimizer.')
