# PrismDMET

**PrismDMET** is an advanced, high-performance Python implementation of Density Matrix Embedding Theory (DMET) for *ab initio* quantum chemistry. 

Originally derived from the legacy QC-DMET framework, PrismDMET has been thoroughly modernized and expanded to include state-of-the-art multi-reference and excited-state solvers, specifically architected for seamless integration with modern electronic structure workflows.

## Key Features
*   **Modern Solvers**: Built-in support for high-level correlated methods including `EOM-CCSD`, `CASSCF`, `NEVPT2`, and the highly efficient multi-state `QD-NEVPT2` solver (via the Prism tensor backend).
*   **PySCF Integration**: Fully compatible with PySCF 2.0+, leveraging its robust mean-field orbital localization, integral transformation, and multi-reference modules.
*   **High Performance**: Critical performance bottlenecks, including the exact local 1-RDM constructions, are accelerated using a compiled C++ library.
*   **Clean Portability**: Standardized deployment via modern Conda environments and editable pip installations.

---

## Installation Guide

PrismDMET is designed for Linux-based HPC environments and requires **Python 3.10+**, a modern **C++ compiler** (supporting C++11/14), and **CMake**.

### 1. Environment Setup
We recommend using the provided `environment.yml` to create a clean, reproducible Conda environment containing all required dependencies (NumPy, SciPy, PySCF, etc.).

```bash
conda env create -f environment.yml
conda activate prismdmet
```

### 2. Compile the C++ Backend
PrismDMET relies on a C++ library (`libprismdmet.so`) for rapid tensor manipulations during the DMET self-consistency loop. 

*Note: Ensure your C++ compiler (e.g., `g++` or `icpc`) and CMake are loaded in your module environment before running this step.*

```bash
cd lib
mkdir build && cd build
cmake .. 
make
```

### 3. Install the Python Package
Once the environment is active and the C++ library is built, install the Python package in editable mode from the root repository directory:

```bash
cd ../..
pip install -e .
```

---

## Usage and Examples

PrismDMET provides an intuitive interface for embedding highly correlated quantum chemistry methods into mean-field environments. 

A typical PrismDMET workflow consists of four steps:
1.  **Mean-Field**: Define a standard PySCF `Mole` object and run a mean-field calculation (e.g., RHF).
2.  **Localization**: Localize the canonical orbitals (e.g., using Meta-Löwdin or IAO).
3.  **Fragmentation**: Define atom groups to fragment the molecule into impurities and baths.
4.  **DMET Execution**: Initialize the `dmet` solver object with the selected high-level `method` and run the simulation.

### Included Tutorials
A comprehensive set of clean tutorial examples is provided in the `examples/` directory to help you get started with various solvers:
*   `water_eomccsd.py`: One-shot DMET using the Equation-of-Motion Coupled Cluster solver.
*   `h6_casscf.py`: Core CASSCF solver demonstrating multi-fragment active spaces.
*   `nitrogen_nevpt2.py`: Using the built-in PySCF-based NEVPT2 correlation solver.
*   `water_qdnevpt2.py`: Advanced multi-state QD-NEVPT2 solver using the Prism backend.

*(Run these scripts directly via `python examples/<script_name>.py`)*

---

## License and Acknowledgments
PrismDMET is an open-source project distributed under the **GNU General Public License v2 (GPL-v2)**. Please see the `LICENSE` file for full details.

This project is a modernized fork and continuation of the original `QC-DMET` code authored by Sebastian Wouters. If you utilize PrismDMET in your academic research, please ensure appropriate citations to the underlying DMET theoretical literature.
