# PrismDMET

**PrismDMET** is a Python implementation of Density Matrix Embedding Theory (DMET) for *ab initio* quantum chemistry. Derived from the legacy QC-DMET framework, it adds multi-reference and excited-state solvers and integrates with PySCF, Prism, and Block2.

## Quick Start

```bash
git clone https://github.com/BrycePickett/PrismDMET.git
cd PrismDMET
conda env create -f environment.yml
conda activate prismdmet
pip install -e .
python examples/hydrogen_fci_oneshot.py
```

```python
from pyscf import gto, scf
from prismdmet import LocalIntegrals, DMET, make_fragments

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf = scf.RHF(mol).run()

ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
fragments = make_fragments(mol, ints, [[0, 1]])

d = DMET(ints, fragments, is_translation_invariant=False, method='FCI')
energy = d.oneshot()
print(f"DMET energy = {energy:.8f} Ha")
```

---

## Installation

Requires **Python 3.10+**. The optional C++ backend (`libprismdmet.so`) accelerates the
self-consistency loop and needs a C++11 compiler and CMake.

```bash
conda env create -f environment.yml
conda activate prismdmet
pip install -e .

# Optional C++ backend
cd lib && mkdir build && cd build && cmake .. && make
```

Without the backend, the LSTSQ self-consistency loop falls back to a pure-Python gradient.

---

## Workflow

1. **Mean-field** — run a PySCF calculation: `mf = scf.RHF(mol).run()`
2. **Localize** — `ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')`
   (schemes: `meta_lowdin`, `boys`, `lowdin`, `iao`)
3. **Fragment** — `fragments = make_fragments(mol, ints, [[0, 1], [2, 3]])`
4. **Run** — one-shot or self-consistent:

```python
# One-shot (no u-matrix optimization)
DMET(ints, fragments, False, method='FCI', sc_method='NONE').oneshot()

# Self-consistent (u-matrix optimization via LSTSQ)
DMET(ints, fragments, False, method='FCI', sc_method='LSTSQ').selfconsistent()
```

---

## Available Methods

| Method Key | Solver | Notes |
|------------|--------|-------|
| `'RHF'`   | Restricted HF | Baseline solver |
| `'ROHF'`  | Restricted open-shell HF | For open-shell systems |
| `'UHF'`   | Unrestricted HF | Full spin polarization |
| `'FCI'` / `'ED'` | Full CI (exact diag.) | Exact for small spaces |
| `'DMRG'`  | DMRG via Block2 | Large active spaces |
| `'CC'`    | CCSD + variants | See `CC_E_TYPE` below |
| `'MP2'`   | MP2 | RHF reference only |
| `'EOM-CC'` | EOM-CCSD | Excited states (one-shot only) |
| `'CASSCF'` | CASSCF | Requires `ncas`, `nelecas` |
| `'NEVPT2'` | SC-NEVPT2 | Requires `ncas`, `nelecas` |
| `'QD-NEVPT2'` | QD-NEVPT2 (Prism) | Requires `ncas`, `nelecas`, `sa_nstates >= 2`, Prism backend |
| `'RKS'` / `'ROKS'` / `'UKS'` | KS-DFT | Set `xc='pbe'` etc. |

`CC_E_TYPE` (CC solver): `'LAMBDA'` (default), `'LAMBDA_AMP'`, `'LAMBDA_ZERO'`, `'CASCI'`, `'CCSD(T)'`, `'CCSD(T)_RDM'`.

---

## Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `method` | str | Solver method key (see table above) |
| `sc_method` | str | `'NONE'` (one-shot) or `'LSTSQ'` (self-consistent) |
| `ncas` / `nelecas` | int | Active orbitals / electrons (CASSCF/NEVPT2) |
| `sa_nstates` | int | Number of states for SA-CASSCF |
| `cas_select` | str | Active-space selection: `'natorb'` (recommended for production), `'impurity'`, `'ao_character'`, `'energy'` (legacy default) |
| `ao_labels` | list | AO labels for `cas_select='ao_character'` (e.g. `['0 H 1s']`) |
| `xc` | str | XC functional for DFT solvers (default `'pbe'`) |
| `parallel` | bool | Enable parallel fragment execution |
| `use_symmetry` / `symmetry_map` | bool / dict | Auto-detect or manually map equivalent fragments |
| `eom_type` / `eom_nroots` | str / int | EOM-CCSD variant and number of roots |

---

## Examples

| File | Description | Runtime |
|------|-------------|---------|
| `hydrogen_fci_oneshot.py` | H2 FCI one-shot vs self-consistent equivalence | < 5 s |
| `hydrogen_hf.py` | H2/H3 RHF/ROHF/UHF validation | < 10 s |
| `hydrogen_dft.py` | H2/H3 DFT (PBE) validation | < 10 s |
| `h6_casscf.py` | H6 chain multi-fragment CASSCF | < 15 s |
| `cas_active_space_selection.py` | CASSCF active-space selection (energy/impurity/ao_character) | < 60 s |
| `h10_fci_selfconsistent.py` | H10 ring self-consistent FCI DMET | < 60 s |
| `nitrogen_nevpt2.py` | N2 NEVPT2 | < 30 s |
| `h6_symmetry_test.py` | H6 ring symmetry detection | < 30 s |
| `water_eomcc.py` | Water EOM-CCSD excited states | ~ 2 min |
| `water_qdnevpt2.py` | Water QD-NEVPT2 (requires Prism) | ~ 5 min |

---

## License and Acknowledgments

PrismDMET is distributed under the **GNU General Public License v2 (GPL-v2)**; see `LICENSE`. It is a modernized fork of the original `QC-DMET` code by Sebastian Wouters. Please cite the underlying DMET literature in academic work.
