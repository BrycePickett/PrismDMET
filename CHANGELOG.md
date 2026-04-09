# Changelog

## [1.2] - 2026-04-09
### Added
- New active-space solvers for DMET loop:
  - `method='CASSCF'`: Supports single and state-averaged calculations with user-defined active spaces (`ncas`, `nelecas`). 
  - `method='EOM-CC'`: Supports all PySCF EOM-CCSD variants.
  - `method='NEVPT2'` and `method='QD-NEVPT2'`: Strong-Contraction and Quasidegenerate NEVPT2 variants operating sequentially on the real representation (`mf_real`).
- New `CC_E_TYPE` options: `CCSD(T)` and `CCSD(T)_RDM` support.
- Configurable dictionary mappings integrated directly into `dmet.__init__` for direct parametric tuning (e.g., `casscf_kwargs`, `eom_kwargs`, `nevpt2_kwargs`).
- Dedicated per-fragment execution artifacts saved on the `dmet` class (`self.eom_results`, `self.cas_results`, etc.).
- Robust evaluation guardrails throwing `RuntimeError` for solvers lacking a compatible loop mapping (strict `oneshot()` compatibility).

## [1.1] - 2026-04-07
### Added
- New `make_fragments` function in `dmet.py` for atom-based fragment definition.
- New `utils.py` module with `silent_stdout` for output suppression.
- New `oneshot()` method in `dmet.py` for one-shot DMET calculations.

### Changed
- Renamed `doselfconsistent()` to `selfconsistent()`
- Post-init attributes (`doDET`, `SCmethod`, `print_u`, etc.) are now keyword arguments in `dmet.__init__`.
- Solvers use `silent_stdout` for safer execution.
- Updated example scripts 


## [1.0] - 2026-03-28
### Added
- New `FCI` solver using PySCF's `pyscf.fci` module
- New `DMRG` solver using [Block2](https://github.com/block-hczhai/block2-preview)

### Changed
- Ported entire codebase from Python 2.7 to Python 3.10
- Updated PySCF API calls for PySCF 2.0+ compatibility
- Centralized all helper scripts and backend solvers into the `src/solvers/` subdirectory
- Replaced depreciated CheMPS2 as default `DMRG` solver with Block2 ([CheMPS2](https://github.com/SebWouters/CheMPS2) remains available as a non-default option).
- `ED` method now calls PySCF FCI

### Removed
- Dependency on deprecated `pyscf.future` module
- Dependency on deprecated `pyscf.tools.rhf_newtonraphson`
- Dependency on deprecated `pyscf.tools.localizer`
