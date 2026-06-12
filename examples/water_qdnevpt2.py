"""
water_qdnevpt2.py -- QD-NEVPT2 one-shot DMET on water using the Prism backend.

Usage::

    pip install -e /path/to/PrismDMET
    python water_qdnevpt2.py
"""

import math
import numpy as np
from pyscf import gto, scf
from prismdmet import LocalIntegrals, DMET, make_fragments

r = 0.96
x = r * math.sin(104.5 * math.pi / (2 * 180.0))
y = r * math.cos(104.5 * math.pi / (2 * 180.0))

mol = gto.M(
    atom=[['O', (0.0, 0.0, 0.0)], ['H', (0.0, -x, y)], ['H', (0.0, x, y)]],
    basis='aug-cc-pvdz', charge=0, spin=0, verbose=0,
)

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.verbose = 0
mf.kernel()
print(f"\nRHF energy = {mf.e_tot:.10f} Ha")

my_ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
my_ints.TI_OK = False
fragments = make_fragments(mol, my_ints, [[0, 1, 2]])

n_states = 6
solver = DMET(
    my_ints, fragments,
    is_translation_invariant=False,
    method='QD-NEVPT2',
    ncas=6, nelecas=6,
    sa_nstates=n_states, sa_weights=None,
    casscf_kwargs={'conv_tol': 1e-11, 'conv_tol_grad': 1e-6},
    qdnevpt2_kwargs={'prism_backend': 'opt_einsum', 'nfrozen': 1, 'compute_singles': False},
    sc_method='NONE',
    print_u=False, print_rdm=False,
)

print(f"\n{'='*60}")
print("  One-shot QD-NEVPT2 DMET on H2O")
print(f"{'='*60}")

E_gs = solver.oneshot(mu_imp=0.0, optimize_mu=False)

print(f"\n  DMET ground-state energy (QD-NEVPT2): {E_gs:.10f} Ha")

if solver.qdnevpt2_results:
    res = solver.qdnevpt2_results[0]
    eV  = 27.21138602
    print(f"\nQD-NEVPT2 results ({n_states} states):")
    print(f"  {'State':>5}  {'E_tot (Ha)':>16}  {'E_corr (Ha)':>14}  {'dE (eV)':>10}")
    print("  " + "-" * 50)
    e0 = res['e_tot'][0]
    for i, (et, ec) in enumerate(zip(res['e_tot'], res['e_corr'])):
        print(f"  {i:>5d}  {et:>16.10f}  {ec:>14.10f}  {(et - e0) * eV:>+10.4f}")
