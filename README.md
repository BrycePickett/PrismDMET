# PrismDMET

**Density Matrix Embedding Theory for ab initio quantum chemistry.**

PrismDMET is a modular Python framework that implements DMET for molecular and
model Hamiltonians, built on top of [PySCF](https://pyscf.org). It supports a
wide range of correlated solvers (FCI, CCSD, CASSCF, NEVPT2, EOM-CCSD, DMRG,
and DFT/HF variants), fragment symmetry detection, OOM-tolerant solver
fallbacks, and parallel fragment execution.

Originally based on [QC-DMET](https://github.com/sebwouters/qc-dmet)
(Wouters et al., 2015), PrismDMET has been extensively modernized for
production-level research.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/BrycePickett/PrismDMET.git
cd PrismDMET
conda env create -f environment.yml
conda activate prismdmet
pip install -e .

# Run the simplest example
python examples/hydrogen_fci_oneshot.py
```

### Minimal Working Example

```python
from pyscf import gto, scf
from prismdmet import LocalIntegrals, DMET, make_fragments

# 1. Build molecule and run mean-field
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf = scf.RHF(mol).run()

# 2. Localize orbitals
ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')

# 3. Define fragments (one fragment = the whole molecule)
fragments = make_fragments(mol, ints, [[0, 1]])

# 4. Run one-shot FCI DMET
d = DMET(ints, fragments, is_translation_invariant=False, method='FCI')
energy = d.oneshot()
print(f"DMET energy = {energy:.8f} Ha")
```

---

## Installation

### Prerequisites

- Python >= 3.10
- PySCF >= 2.0
- NumPy, SciPy
- C++ compiler + LAPACK (for the `libprismdmet.so` extension)
- Optional: [Block2](https://github.com/block-hczhai/block2-preview) for DMRG

### Step 1: Create the Conda environment

```bash
conda env create -f environment.yml
conda activate prismdmet
```

### Step 2: Install PrismDMET

```bash
pip install -e .
```

### Step 3: Build the C++ extension (optional, for gradient calculations)

```bash
cd lib
mkdir build && cd build
cmake ..
make
cd ../..
```

This produces `lib/libprismdmet.so`, which is loaded automatically by the
DMET driver when available.

---

## Workflow

PrismDMET follows a 4-step workflow:

### 1. Mean-field calculation

Run a standard PySCF mean-field calculation (RHF, ROHF, or UHF):

```python
from pyscf import gto, scf
mol = gto.M(atom='...', basis='sto-3g')
mf = scf.RHF(mol).run()
```

### 2. Orbital localization

Construct localized integrals using one of the supported schemes:

```python
from prismdmet import LocalIntegrals
ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
```

Supported localization types: `'meta_lowdin'` (default, robust),
`'boys'`, `'iao'`.

### 3. Fragment definition

Define fragments by grouping atoms:

```python
from prismdmet import make_fragments
# 3 fragments of 2 atoms each
fragments = make_fragments(mol, ints, [[0, 1], [2, 3], [4, 5]])
```

### 4. DMET execution

Run one-shot or self-consistent DMET:

```python
from prismdmet import DMET

# One-shot (no u-matrix optimization)
d = DMET(ints, fragments, False, method='FCI', sc_method='NONE')
energy = d.oneshot()

# Self-consistent (with u-matrix optimization via LSTSQ)
d = DMET(ints, fragments, False, method='FCI', sc_method='LSTSQ')
energy = d.selfconsistent()
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
| `'MP2'`   | MP2 | Fast, RHF reference only |
| `'EOM-CC'` | EOM-CCSD | Excited states (one-shot only) |
| `'CASSCF'` | CASSCF | Requires `ncas`, `nelecas` |
| `'NEVPT2'` | SC-NEVPT2 | Requires `mf_real`, `ncas`, `nelecas` |
| `'QD-NEVPT2'` | QD-NEVPT2 (Prism) | Requires Prism backend |
| `'RKS'`   | Restricted KS-DFT | Set `xc='pbe'` etc. |
| `'ROKS'`  | Restricted open-shell KS | Set `xc='pbe'` etc. |
| `'UKS'`   | Unrestricted KS-DFT | Set `xc='pbe'` etc. |

### CC Energy Types (`CC_E_TYPE`)

| Value | Description |
|-------|-------------|
| `'LAMBDA'` | Standard CCSD Lambda RDM (default) |
| `'LAMBDA_AMP'` | Approximate RDM (lambda = amplitudes) |
| `'LAMBDA_ZERO'` | RDM at zero lambda |
| `'CASCI'` | Total CCSD energy, no projection |
| `'CCSD(T)'` | Perturbative triples on CCSD Lambda-RDM |
| `'CCSD(T)_RDM'` | Full CCSD(T) relaxed RDM |

---

## Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `method` | str | Solver method key (see table above) |
| `sc_method` | str | `'NONE'` (one-shot) or `'LSTSQ'` (self-consistent) |
| `ncas` | int | Active orbitals (CASSCF/NEVPT2) |
| `nelecas` | int | Active electrons (CASSCF/NEVPT2) |
| `sa_nstates` | int | Number of states for SA-CASSCF |
| `mf_real` | object | Converged PySCF mean-field object (required for NEVPT2/QD-NEVPT2) |
| `xc` | str | XC functional for DFT solvers (default `'pbe'`) |
| `parallel` | bool | Enable parallel fragment execution |
| `use_symmetry` | bool | Auto-detect equivalent fragments |
| `symmetry_map` | dict | Manual fragment equivalence mapping |
| `eom_type` | str | EOM-CCSD variant (`'EE-Singlet'`, `'IP'`, `'EA'`, etc.) |
| `eom_nroots` | int | Number of EOM-CCSD roots |

---

## Examples

| File | Description | Runtime |
|------|-------------|---------|
| `hydrogen_fci_oneshot.py` | H2 FCI one-shot vs self-consistent equivalence | < 5 s |
| `hydrogen_hf.py` | H2/H3 RHF/ROHF/UHF validation | < 10 s |
| `hydrogen_dft.py` | H2/H3 DFT (PBE) validation | < 10 s |
| `h6_casscf.py` | H6 chain CASSCF with multi-fragment | < 15 s |
| `h10_fci_selfconsistent.py` | H10 ring self-consistent FCI DMET | < 60 s |
| `nitrogen_nevpt2.py` | N2 NEVPT2 | < 30 s |
| `h6_symmetry_test.py` | H6 ring symmetry detection | < 30 s |
| `water_eomcc.py` | Water EOM-CCSD excited states | ~ 2 min |
| `water_qdnevpt2.py` | Water QD-NEVPT2 (requires Prism) | ~ 5 min |

Legacy examples from QC-DMET are in `examples/legacy/`.

---

## Troubleshooting

**ImportError: No module named 'prismdmet'**
Run `pip install -e .` from the PrismDMET root directory.

**libprismdmet.so not found**
The C++ extension is optional. Build it with `cd lib && mkdir build && cd build && cmake .. && make`.
Without it, the LSTSQ self-consistency loop falls back to a pure-Python gradient.

**MemoryError during solver execution**
PrismDMET has automatic OOM fallback chains (e.g., FCI -> DMRG -> CASSCF -> CC -> MP2).
A RuntimeWarning will indicate when a fallback is triggered.

**NEVPT2/QD-NEVPT2 requires mf_real**
These methods need the original PySCF mean-field object. Pass it as `mf_real=mf`.

---

## License

GNU General Public License, version 2 (GPL-v2).

Based on QC-DMET by Sebastian Wouters (2015).
