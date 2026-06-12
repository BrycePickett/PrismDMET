"""
cas_active_space_selection.py -- CASSCF active-space selection modes in PrismDMET.

The DMET embedding Fock is non-canonical (the bath is built by Schmidt
decomposition, not by energy), so the default energy-ordered active window
[ncore:ncore+ncas] can place the wrong orbitals in the active space. PrismDMET
offers three ways to choose the CAS active orbitals via the cas_select argument:

    'energy'       -- default PySCF energy ordering (no reordering).
    'impurity'     -- rank frontier MOs by impurity localization weight
                      Sum|C[:nimp, i]|^2 and take the ncas largest. Automatic;
                      no AO labels needed.
    'ao_character' -- rank MOs by projection onto named atomic orbitals
                      (ao_labels) and take the ncas largest. Targets a specific
                      chemical character (e.g. a defect/trap state). Falls back
                      to 'impurity' if ao_labels are not provided.

This example embeds the first two atoms of an H4 chain as the impurity (two
impurity orbitals plus a Schmidt bath) and runs CASSCF(2,2) three times,
changing only cas_select. The printed table shows which embedded orbitals each
mode places in the active space and the resulting energy.

Usage::

    pip install -e /path/to/PrismDMET
    python cas_active_space_selection.py
"""

# Cap BLAS threads: the repeated tiny CASSCF macro-iteration calls are dominated
# by threading overhead at this size, so a single thread is fastest here.
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from pyscf import gto, scf
from prismdmet import LocalIntegrals, DMET, make_fragments

# 1. Molecule (H4 chain) and mean-field
bond = 1.4
mol = gto.M(atom=[('H', (i * bond, 0.0, 0.0)) for i in range(4)],
            basis='sto-3g', verbose=0)
mf = scf.RHF(mol).run()
print(f"RHF energy = {mf.e_tot:.10f} Ha")
print(f"AO labels  = {[lbl.strip() for lbl in mol.ao_labels()]}")

# 2. Localize and embed the first two atoms as a single impurity fragment
my_ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
fragments = make_fragments(mol, my_ints, [[0, 1]])

# ao_character targets the 1s orbitals on the two impurity atoms
ao_labels = ['0 H 1s', '1 H 1s']

# 3. Run CASSCF(2,2) under each selection mode
n_cas, n_elecas = 2, 2
print(f"\n{'='*64}")
print(f"  CASSCF({n_cas},{n_elecas}) active-space selection on H4 (impurity = atoms 0,1)")
print(f"{'='*64}")

results = {}
for mode in ('energy', 'impurity', 'ao_character'):
    kwargs = dict(ncas=n_cas, nelecas=n_elecas, cas_select=mode,
                  print_u=False, print_rdm=False)
    if mode == 'ao_character':
        kwargs['ao_labels'] = ao_labels
    d = DMET(my_ints, fragments, is_translation_invariant=False,
             method='CASSCF', sc_method='NONE', **kwargs)
    e = d.oneshot(mu_imp=0.0)
    selected = d.cas_results[0]['selected_orbs']  # None for 'energy' (default window)
    results[mode] = (e, selected)

# 4. Comparison table
print(f"\n{'mode':<14}{'selected orbitals':<22}{'CASSCF energy (Ha)':>20}")
print('-' * 56)
for mode in ('energy', 'impurity', 'ao_character'):
    e, selected = results[mode]
    sel_str = 'default window' if selected is None else str(selected)
    print(f"{mode:<14}{sel_str:<22}{e:>20.10f}")

print("\nThe three modes can select different active orbitals; for a system with")
print("a clear target character, 'ao_character' or 'impurity' avoids the wrong")
print("energy-ordered window that a non-canonical embedding basis can produce.")
