"""
water_eomcc.py -- EOM-CCSD one-shot DMET on a water molecule.

Demonstrates the EOM-CC solver for PrismDMET. A single impurity covers
the whole molecule. The script runs EE-Singlet EOM-CCSD to compute
excited state energies, then runs IP-EOM-CCSD for ionization potentials.

Usage::

    pip install -e /path/to/PrismDMET
    python water_eomcc.py
"""

import numpy as np
from pyscf import gto, scf
from prismdmet import LocalIntegrals, DMET, make_fragments

# 1. Molecule
mol = gto.M(
    atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586',
    basis='aug-cc-pvdz', charge=0, spin=0, verbose=3,
)

# 2. Mean-field RHF
mf = scf.RHF(mol)
mf.verbose = 3
mf.run()
print(f"\nRHF energy = {mf.e_tot:.10f} Ha")

# 3. Localize
my_ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
my_ints.TI_OK = False

# 4. Fragment (whole molecule as one impurity)
atom_groups = [[0, 1, 2]]
fragments = make_fragments(mol, my_ints, atom_groups)

# 5. EE-Singlet EOM-CCSD
eom_solver = DMET(
    my_ints, fragments,
    is_translation_invariant=False,
    method='EOM-CC',
    eom_type='EE-Singlet',
    eom_nroots=5,
    eom_koopmans=False,
    sc_method='NONE',
    print_u=False, print_rdm=False,
)

print(f"\n{'='*60}")
print("  One-shot EOM-CCSD DMET on H2O")
print(f"{'='*60}")

E_gs = eom_solver.oneshot(mu_imp=0.0, optimize_mu=False)

print(f"\n{'='*60}")
print(f"  DMET ground-state energy (CCSD): {E_gs:.10f} Ha")
print(f"{'='*60}")

if eom_solver.eom_results:
    res = eom_solver.eom_results[0]
    print(f"\nEOM-CCSD [{res['eom_type']}] results:")
    print(f"  Ground-state CCSD energy : {res['E_ccsd']:.10f} Ha")
    print(f"\n  {'State':>5}  {'dE (eV)':>10}  {'E_abs (Ha)':>16}")
    print("  " + "-" * 36)
    for i, (de_ev, e_abs) in enumerate(zip(res['delta_E_eV'], res['E_states'])):
        print(f"  {i:>5d}  {de_ev:>10.4f}  {e_abs:>16.10f}")

# 6. IP-EOM-CCSD
print(f"\n{'='*60}")
print("  IP-EOM-CCSD (ionization potentials)")
print(f"{'='*60}")

ip_solver = DMET(
    my_ints, fragments,
    is_translation_invariant=False,
    method='EOM-CC', eom_type='IP', eom_nroots=3,
    sc_method='NONE', print_u=False, print_rdm=False,
)
ip_solver.oneshot(mu_imp=0.0)

if ip_solver.eom_results:
    res = ip_solver.eom_results[0]
    print(f"\nIP-EOM-CCSD ionization potentials:")
    for i, de in enumerate(res['delta_E']):
        print(f"  IP {i}: {-de * 27.2114:.4f} eV")

print("\nDone.")
