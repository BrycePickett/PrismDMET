"""
EOM-CCSD one-shot DMET on a water molecule.

This example demonstrates the EOM-CC solver for PrismDMET. A single
impurity is placed on the oxygen atom (all its AOs) with hydrogen
atoms as the bath.  One-shot DMET is used: no u-matrix optimization
is performed.

The script runs EE-Singlet EOM-CCSD by default and shows how to switch
to other variants (EE-Triplet, IP, EA, Spin-Flip) by changing eom_type.

Run from this directory with the prismdmet conda environment active:
    python 01_water_eomcc.py
"""

import sys
import numpy as np
from pyscf import gto, scf

import local_integrals
import dmet
from dmet import make_fragments

# ---------------------------------------------------------------------------
# 1.  Define the molecule
# ---------------------------------------------------------------------------
mol = gto.Mole()
mol.atom = """
O  0.000000  0.000000  0.000000
H  0.000000  0.757000  0.586000
H  0.000000 -0.757000  0.586000
"""
mol.basis  = 'aug-cc-pvdz'
mol.charge = 0
mol.spin   = 0
mol.build(verbose=3)

# ---------------------------------------------------------------------------
# 2.  Mean-field RHF
# ---------------------------------------------------------------------------
mf = scf.RHF(mol)
mf.verbose = 3
mf.run()
print(f"\nRHF energy = {mf.e_tot:.10f} Ha")

# ---------------------------------------------------------------------------
# 3.  Localise orbitals
# ---------------------------------------------------------------------------
myInts = local_integrals.localintegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
myInts.TI_OK = False

# ---------------------------------------------------------------------------
# 4.  Fragment definition  –  O as impurity, H atoms as bath
#     atom_groups must cover every atom exactly once
#     Here: one fragment = all atoms (whole-molecule impurity), which is the
#     simplest valid choice for a small molecule.
# ---------------------------------------------------------------------------
atom_groups = [[0, 1, 2]]        # whole molecule as one impurity
fragments   = make_fragments(mol, myInts, atom_groups)

# ---------------------------------------------------------------------------
# 5.  EOM-CC DMET solver
# ---------------------------------------------------------------------------
eom_solver = dmet.dmet(
    myInts,
    fragments,
    isTranslationInvariant = False,
    method     = 'EOM-CC',
    eom_type   = 'EE-Singlet',   # switch to 'IP', 'EA', 'EE-Triplet', etc.
    eom_nroots = 5,              # number of excited states per fragment
    eom_koopmans = False,        
    SCmethod   = 'NONE',         # required for EOM-CC (oneshot only)
    print_u    = False,
    print_rdm  = False,
)

# ---------------------------------------------------------------------------
# 6.  Run one-shot DMET
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  One-shot EOM-CCSD DMET on H2O")
print("="*60)

E_gs = eom_solver.oneshot(mu_imp=0.0, optimize_mu=False)

print("\n" + "="*60)
print(f"  DMET ground-state energy (CCSD): {E_gs:.10f} Ha")
print("="*60)

# ---------------------------------------------------------------------------
# 7.  Collect and print EOM results
# ---------------------------------------------------------------------------
if eom_solver.eom_results:
    res = eom_solver.eom_results[0]   # one result per fragment
    print(f"\nEOM-CCSD [{res['eom_type']}] results:")
    print(f"  Ground-state CCSD energy : {res['E_ccsd']:.10f} Ha")
    print()
    print(f"  {'State':>5}  {'ΔE (eV)':>10}  {'E_abs (Ha)':>16}")
    print("  " + "-"*36)
    for i, (de_ev, e_abs) in enumerate(zip(res['delta_E_eV'], res['E_states'])):
        print(f"  {i:>5d}  {de_ev:>10.4f}  {e_abs:>16.10f}")

# ---------------------------------------------------------------------------
# 8.  Demonstrating how to run IP variants
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  IP-EOM-CCSD (ionisation potentials)")
print("="*60)

ip_solver = dmet.dmet(
    myInts, fragments,
    isTranslationInvariant = False,
    method       = 'EOM-CC',
    eom_type     = 'IP',
    eom_nroots   = 3,
    SCmethod     = 'NONE',
    print_u      = False,
    print_rdm    = False,
)
ip_solver.oneshot(mu_imp=0.0)

if ip_solver.eom_results:
    res = ip_solver.eom_results[0]
    print(f"\nIP-EOM-CCSD ionisation potentials:")
    for i, de in enumerate(res['delta_E']):
        print(f"  IP {i}: {-de * 27.2114:.4f} eV")
