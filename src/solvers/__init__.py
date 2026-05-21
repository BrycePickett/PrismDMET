"""Solver dispatcher: maps method-key strings to solver implementations."""

import warnings


PARALLEL_ELIGIBLE = frozenset({
    'ED', 'FCI', 'CC', 'MP2', 'EOM-CC', 'DMRG', 'flag_rhf',
    'CASSCF', 'NEVPT2', 'QD-NEVPT2',
    'RKS', 'UKS', 'ROKS', 'UHF', 'ROHF',
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
    # DFT fallbacks: unrestricted -> restricted open-shell -> restricted
    'UKS'       : 'ROKS',
    'ROKS'      : 'RKS',
    'UHF'       : 'ROHF',
    'ROHF'      : 'flag_rhf',
}


def _is_oom_like(exc):
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(kw in msg for kw in (
            'out of memory', 'oom', 'insufficient memory',
            'cannot allocate', 'lineardependence'))
    return False



class SolverDispatcher:

    @staticmethod
    def execute(task):
        method = task['method']

        dispatch = {
            'flag_rhf'    : SolverDispatcher._run_rhf,
            'RHF'         : SolverDispatcher._run_rhf,
            'ED'          : SolverDispatcher._run_fci,
            'FCI'         : SolverDispatcher._run_fci,
            'DMRG'        : SolverDispatcher._run_dmrg,
            'CC'          : SolverDispatcher._run_cc,
            'MP2'         : SolverDispatcher._run_mp2,
            'EOM-CC'      : SolverDispatcher._run_eomcc,
            'CASSCF'      : SolverDispatcher._run_casscf,
            'QD-NEVPT2'   : SolverDispatcher._run_qdnevpt2,
            'NEVPT2'      : SolverDispatcher._run_nevpt2,
            # Open-shell HF
            'UHF'         : SolverDispatcher._run_uhf,
            'ROHF'        : SolverDispatcher._run_rohf,
            # DFT
            'RKS'         : SolverDispatcher._run_dft,
            'UKS'         : SolverDispatcher._run_dft,
            'ROKS'        : SolverDispatcher._run_dft,
        }

        if method not in dispatch:
            raise ValueError(
                f"SolverDispatcher.execute: unknown method='{method}'. "
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
                    f"\nPrismdmet OOM FATAL: fragment {counter}, "
                    f"method '{method}' raised {type(exc).__name__}: {exc}. "
                    f"No fallback defined. Re-raising.",
                    RuntimeWarning, stacklevel=2,
                )
                raise

            warnings.warn(
                f"\nPrismdmet OOM FALLBACK: fragment {counter}, "
                f"method '{method}' raised {type(exc).__name__}: {exc}. "
                f"Retrying with '{fallback}'.",
                RuntimeWarning, stacklevel=2,
            )
            original_method       = task.get('fallback_from', method)
            task                  = dict(task)
            task['method']        = fallback
            task['fallback_from'] = original_method

            result = SolverDispatcher.execute(task)
            result.setdefault('fallback_from', original_method)
            return result

    @staticmethod
    def _run_rhf(task):
        from . import rhf
        energy, rdm1 = rhf.execute(task)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}

    @staticmethod
    def _run_fci(task):
        from . import fci
        energy, rdm1 = fci.execute(task)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}

    @staticmethod
    def _run_dmrg(task):
        from . import block2
        energy, rdm1 = block2.execute(task)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}

    @staticmethod
    def _run_cc(task):
        from . import cc
        energy, rdm1 = cc.execute(task)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}

    @staticmethod
    def _run_mp2(task):
        from . import mp2
        energy, rdm1 = mp2.execute(task)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}

    @staticmethod
    def _run_eomcc(task):
        from . import eomcc
        energy, rdm1, eom_res = eomcc.execute(task)
        return {'counter': task['counter'], 'energy': energy,
                'rdm1': rdm1, 'eom_res': eom_res}

    @staticmethod
    def _run_casscf(task):
        from . import casscf
        energy, rdm1, cas_res = casscf.execute(task)
        return {'counter': task['counter'], 'energy': energy,
                'rdm1': rdm1, 'cas_res': cas_res}

    @staticmethod
    def _run_qdnevpt2(task):
        from . import qdnevpt2
        energy, rdm1, qdnevpt2_res = qdnevpt2.execute(task)
        return {'counter': task['counter'], 'energy': energy,
                'rdm1': rdm1, 'qdnevpt2_res': qdnevpt2_res}

    @staticmethod
    def _run_nevpt2(task):
        from . import nevpt2
        energy, rdm1, nevpt2_res = nevpt2.execute(task)
        return {'counter': task['counter'], 'energy': energy,
                'rdm1': rdm1, 'nevpt2_res': nevpt2_res}

    @staticmethod
    def _run_uhf(task):
        from . import uhf
        return uhf.execute(task)

    @staticmethod
    def _run_rohf(task):
        from . import uhf
        return uhf.execute(task)

    @staticmethod
    def _run_dft(task):
        from . import dft
        return dft.execute(task)

