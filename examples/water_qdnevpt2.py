"""
QD-NEVPT2 one-shot DMET on water using the Prism backend.

Pipeline:
  1. Build the real physical molecule and run RHF.
  2. Set up DMET with method='QD-NEVPT2', providing the mf object and
     active space (ncas, nelecas, sa_nstates).
  3. Call oneshot() — this internally runs SA-CASSCF on the real molecule,
     then calls Prism's QD-NEVPT2 solver.

Run from this directory with the prismdmet conda environment active:
    python 04_water_qdnevpt2.py
"""

import sys
import math
import numpy as np
from pyscf import gto, scf, mcscf

import local_integrals
import dmet
from dmet import make_fragments

# ---------------------------------------------------------------------------
# 1.  Define the water molecule
# ---------------------------------------------------------------------------
r = 0.96
x = r * math.sin(104.5 * math.pi / (2 * 180.0))
y = r * math.cos(104.5 * math.pi / (2 * 180.0))

mol = gto.Mole()
mol.atom = [
    ['O', (0.0, 0.0, 0.0)],
    ['H', (0.0,  -x,   y)],
    ['H', (0.0,   x,   y)],
]
mol.basis   = 'aug-cc-pvdz'
mol.charge  = 0
mol.spin    = 0
mol.verbose = 0
mol.build()

# ---------------------------------------------------------------------------
# 2.  RHF mean-field (on the REAL molecule — required for Prism)
# ---------------------------------------------------------------------------
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.verbose = 0
mf.kernel()
print(f"\nRHF energy = {mf.e_tot:.10f} Ha")

# ---------------------------------------------------------------------------
# 3.  Local integrals for DMET embedding
# ---------------------------------------------------------------------------
myInts = local_integrals.localintegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
myInts.TI_OK = False

# ---------------------------------------------------------------------------
# 4.  Fragment definition — whole molecule as one impurity
# ---------------------------------------------------------------------------
atom_groups = [[0, 1, 2]]
fragments   = make_fragments(mol, myInts, atom_groups)

# ---------------------------------------------------------------------------
# 5.  QD-NEVPT2 DMET solver
#
#  Key parameters:
#    mf_real    — the converged RHF object on the REAL molecule (required)
#    ncas       — number of active orbitals for SA-CASSCF
#    nelecas    — number of active electrons for SA-CASSCF
#    sa_nstates — number of states to average (minimum 2 for QD-NEVPT2)
#
#  Optional Prism QD-NEVPT2 tuning (passed via qdnevpt2_kwargs):
#    nfrozen           — frozen core orbitals for NEVPT2
#    prism_backend     — 'opt_einsum' (fastest), 'numpy', or 'pytblis'
# ---------------------------------------------------------------------------
n_states = 6

solver = dmet.dmet(
    myInts,
    fragments,
    isTranslationInvariant = False,
    method      = 'QD-NEVPT2',
    mf_real     = mf,              # REQUIRED for QD-NEVPT2
    ncas        = 6,               # active orbitals (6 for water with aug-cc-pvdz)
    nelecas     = 6,               # active electrons
    sa_nstates  = n_states,
    sa_weights  = None,            # uniform weights
    casscf_kwargs = {
        'conv_tol'      : 1e-11,
        'conv_tol_grad' : 1e-6,
    },
    qdnevpt2_kwargs = {
        'prism_backend'  : 'opt_einsum',
        'nfrozen'        : 1,      # freeze 1s oxygen core
        'compute_singles': False,
    },
    SCmethod    = 'NONE',
    print_u     = False,
    print_rdm   = False,
)

# ---------------------------------------------------------------------------
# 6.  Run one-shot DMET
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  One-shot QD-NEVPT2 DMET on H2O")
print("="*60)

E_gs = solver.oneshot(mu_imp=0.0, optimize_mu=False)

print("\n" + "="*60)
print(f"  DMET ground-state energy (QD-NEVPT2): {E_gs:.10f} Ha")
print("="*60)

# ---------------------------------------------------------------------------
# 7.  Print QD-NEVPT2 results for all states
# ---------------------------------------------------------------------------
if solver.qdnevpt2_results:
    res = solver.qdnevpt2_results[0]
    eV  = 27.21138602

    print(f"\nQD-NEVPT2 results ({n_states} states):")
    print(f"  {'State':>5}  {'E_tot (Ha)':>16}  {'E_corr (Ha)':>14}  {'ΔE from GS (eV)':>16}")
    print("  " + "-"*56)
    e_gs_qdnevpt2 = res['e_tot'][0]
    for i, (et, ec) in enumerate(zip(res['e_tot'], res['e_corr'])):
        de_ev = (et - e_gs_qdnevpt2) * eV
        print(f"  {i:>5d}  {et:>16.10f}  {ec:>14.10f}  {de_ev:>+16.4f}")
