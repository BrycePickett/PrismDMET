"""
Central dispatcher that maps method-key strings to solver implementations.

Every solver module exposes a top-level execute(task) function that accepts a
standardized task dictionary and returns a standardized result dictionary.
SolverFactory.execute(task) delegates to the correct module based on
task['method'], so dmet.py and the parallel worker do not need to know
anything about individual solver signatures.

Task dict required keys:
    method        : str    - solver key
    CONST         : float  - frozen-core constant energy
    dmetOEI       : ndarray (N,N)       - 1e integrals in DMET basis
    dmetFOCK      : ndarray (N,N)       - Fock matrix in DMET basis
    dmetTEI       : ndarray (N,N,N,N)   - 2e integrals in DMET basis
    Norb          : int    - total embedding orbitals (impurity + bath)
    Nel           : int    - electrons in the embedding space
    Nimp          : int    - impurity orbitals (first Nimp of Norb)
    chempot_imp   : float  - chemical potential on the impurity block
    counter       : int    - fragment index
    src_path      : str    - absolute path to src/ for worker sys.path

Task dict optional keys:
    DMguessRHF    : ndarray or None
    CC_E_TYPE     : str
    eom_nroots    : int
    eom_type      : str
    eom_koopmans  : bool
    eom_kwargs    : dict
    ncas          : int
    nelecas       : int
    sa_nstates    : int
    sa_weights    : list
    casscf_kwargs : dict
    mo_guess      : ndarray or None
    ci_guess      : ndarray or None
    OEI_S         : ndarray or None
    nevpt2_kwargs : dict
    qdnevpt2_kwargs: dict
    mol_dumps     : str       - JSON from pyscf.gto.Mole.dumps()
    mf_mo_coeff   : ndarray   - canonical MO coefficients
    mf_mo_energy  : ndarray   - canonical MO energies
    mf_mo_occ     : ndarray   - MO occupations
    mf_e_tot      : float     - total RHF energy

Result dict required keys:
    counter       : int
    energy        : float
    rdm1          : ndarray (N,N)

Result dict optional keys:
    eom_res       : dict
    cas_res       : dict
    qdnevpt2_res  : dict
    nevpt2_res    : dict
    fallback_from : str or None

OOM fallback chain:
    FCI -> DMRG -> CASSCF -> CC -> MP2
    ED  -> DMRG -> CASSCF -> CC -> MP2
    QD-NEVPT2 -> NEVPT2 -> CASSCF -> CC -> MP2
    EOM-CC -> CC -> MP2

When a fallback fires, a RuntimeWarning is printed, task['method'] is
updated, and task['fallback_from'] records the originally requested method.
"""

import sys
import os
import warnings


PARALLEL_ELIGIBLE = frozenset({
    'ED', 'FCI', 'CC', 'MP2', 'EOM-CC', 'DMRG', 'flag_rhf',
    'CASSCF', 'NEVPT2', 'QD-NEVPT2',
})

FALLBACK_CHAIN = {
    'FCI'       : 'DMRG',
    'ED'        : 'DMRG',
    'DMRG'      : 'CASSCF',
    'CASSCF'    : 'CC',
    'QD-NEVPT2' : 'NEVPT2',
    'NEVPT2'    : 'CASSCF',
    'CC'        : 'MP2',
    'EOM-CC'    : 'CC',
}


def _is_oom_like(exc):
    """Return True if the exception should trigger a graceful fallback."""
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(kw in msg for kw in (
            'out of memory', 'oom', 'insufficient memory',
            'cannot allocate', 'lineardependence'))
    return False


def _ensure_src_path(task):
    src_path = task.get('src_path', '')
    if src_path and src_path not in sys.path:
        sys.path.insert(0, src_path)


class SolverFactory:
    """
    Central dispatcher for all DMET fragment solvers.

    Usage:
        result = SolverFactory.execute(task)

    If a solver raises MemoryError or an OOM-like RuntimeError, the factory
    retries with the next cheaper method from FALLBACK_CHAIN. The result dict
    carries 'fallback_from' indicating which method originally failed.
    """

    @staticmethod
    def execute(task):
        """
        Dispatch a fragment task to the correct solver.

        Parameters
        ----------
        task : dict
            Standardized task dict. See module docstring for schema.

        Returns
        -------
        dict with keys: counter, energy, rdm1, plus optional method-specific keys.

        Raises
        ------
        ValueError
            If task['method'] is not a recognized solver key.
        MemoryError / RuntimeError
            If an OOM error occurs and no fallback is defined for the method.
        """
        _ensure_src_path(task)
        method = task['method']

        dispatch = {
            'flag_rhf'    : SolverFactory._run_rhf,
            'ED'          : SolverFactory._run_fci,
            'FCI'         : SolverFactory._run_fci,
            'DMRG'        : SolverFactory._run_dmrg,
            'DMRG-CheMPS2': SolverFactory._run_chemps2,
            'CC'          : SolverFactory._run_cc,
            'MP2'         : SolverFactory._run_mp2,
            'EOM-CC'      : SolverFactory._run_eomcc,
            'CASSCF'      : SolverFactory._run_casscf,
            'QD-NEVPT2'   : SolverFactory._run_qdnevpt2,
            'NEVPT2'      : SolverFactory._run_nevpt2,
        }

        if method not in dispatch:
            raise ValueError(
                f"SolverFactory.execute: unknown method='{method}'. "
                f"Valid keys: {sorted(dispatch.keys())}"
            )

        try:
            return dispatch[method](task)

        except Exception as exc:
            if not _is_oom_like(exc):
                raise

            counter  = task.get('counter', '?')
            fallback = FALLBACK_CHAIN.get(method)

            if fallback is None:
                warnings.warn(
                    f"\nPrismDMET OOM FATAL: fragment {counter}, "
                    f"method '{method}' raised {type(exc).__name__}: {exc}. "
                    f"No fallback defined. Re-raising.",
                    RuntimeWarning, stacklevel=2,
                )
                raise

            warnings.warn(
                f"\nPrismDMET OOM FALLBACK: fragment {counter}, "
                f"method '{method}' raised {type(exc).__name__}: {exc}. "
                f"Retrying with '{fallback}'.",
                RuntimeWarning, stacklevel=2,
            )
            original_method       = task.get('fallback_from', method)
            task                  = dict(task)
            task['method']        = fallback
            task['fallback_from'] = original_method

            result = SolverFactory.execute(task)
            result.setdefault('fallback_from', original_method)
            return result

    @staticmethod
    def _run_rhf(task):
        from solvers import rhf
        energy, rdm1 = rhf.execute(task)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}

    @staticmethod
    def _run_fci(task):
        from solvers import fci
        energy, rdm1 = fci.execute(task)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}

    @staticmethod
    def _run_dmrg(task):
        from solvers import block2
        energy, rdm1 = block2.execute(task)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}

    @staticmethod
    def _run_chemps2(task):
        from solvers import chemps2
        energy, rdm1 = chemps2.execute(task)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}

    @staticmethod
    def _run_cc(task):
        from solvers import cc
        energy, rdm1 = cc.execute(task)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}

    @staticmethod
    def _run_mp2(task):
        from solvers import mp2
        energy, rdm1 = mp2.execute(task)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}

    @staticmethod
    def _run_eomcc(task):
        from solvers import eomcc
        energy, rdm1, eom_res = eomcc.execute(task)
        return {'counter': task['counter'], 'energy': energy,
                'rdm1': rdm1, 'eom_res': eom_res}

    @staticmethod
    def _run_casscf(task):
        from solvers import casscf
        energy, rdm1, cas_res = casscf.execute(task)
        return {'counter': task['counter'], 'energy': energy,
                'rdm1': rdm1, 'cas_res': cas_res}

    @staticmethod
    def _run_qdnevpt2(task):
        from solvers import qdnevpt2
        energy, rdm1, qdnevpt2_res = qdnevpt2.execute(task)
        return {'counter': task['counter'], 'energy': energy,
                'rdm1': rdm1, 'qdnevpt2_res': qdnevpt2_res}

    @staticmethod
    def _run_nevpt2(task):
        from solvers import nevpt2
        energy, rdm1, nevpt2_res = nevpt2.execute(task)
        return {'counter': task['counter'], 'energy': energy,
                'rdm1': rdm1, 'nevpt2_res': nevpt2_res}
