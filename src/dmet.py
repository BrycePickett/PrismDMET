"""PrismDMET core DMET driver."""

from . import prismdmet_helper
import numpy as np
from scipy import optimize
import time
import os
import concurrent.futures
from .solvers import SolverDispatcher
from .fragment_builder import FragmentBuilder


def _fragment_worker(task):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    from prismdmet.solvers import SolverDispatcher
    return SolverDispatcher.execute(task)


class DMET:

    ####################
    # Construction and validation
    ####################

    def __init__( self, integrals, fragments, is_translation_invariant, method='ED',
                  sc_method='LSTSQ', fit_impurity_and_bath=True, use_constrained_optimization=False,
                  use_density_embedding=False, use_density_embedding_no=False, CC_E_TYPE='LAMBDA',
                  print_u=True, print_rdm=True, eom_nroots=3,
                  eom_type='EE-Singlet', eom_koopmans=False, eom_kwargs=None,
                  ncas=None, nelecas=None, sa_nstates=1, sa_weights=None,
                  casscf_kwargs=None, dmrg_kwargs=None,
                  mf_real=None, qdnevpt2_kwargs=None, nevpt2_kwargs=None,
                  use_symmetry=False, symmetry_map=None,
                  parallel=False, max_workers=None, bath_tol=1e-13,
                  xc='pbe', level_shift=0.0, spin_polarized=False,
                  mm_coords=None, mm_charges=None,
                  include_spin_oei=False, cas_select='energy', ao_labels=None,
                  avas_threshold=0.2, spade_gap_tol=0.3, spade_n_fallback=16,
                  embed_level_shift=0.0, rohf_stability=False, cas_multiseed=False,
                  cas_spin=None, cas_spin_shift=0.2, allow_solver_fallback=True,
                  fragment_methods=None,
                  natorb_occ_thresh=0.02, natorb_max_superset=None,
                  deg_tol=1e-3, casci_conv_tol=1e-10,
                  **deprecated_kwargs ):

        import warnings
        # Deprecated keyword aliases (accepted for one release; map old -> new, warn).
        if 'do_det' in deprecated_kwargs:
            warnings.warn("DMET keyword 'do_det' is deprecated; use 'use_density_embedding'.",
                          DeprecationWarning, stacklevel=2)
            use_density_embedding = deprecated_kwargs.pop('do_det')
        if 'do_det_NO' in deprecated_kwargs:
            warnings.warn("DMET keyword 'do_det_NO' is deprecated; use 'use_density_embedding_no'.",
                          DeprecationWarning, stacklevel=2)
            use_density_embedding_no = deprecated_kwargs.pop('do_det_NO')
        if 'fit_imp_bath' in deprecated_kwargs:
            warnings.warn("DMET keyword 'fit_imp_bath' is deprecated; use 'fit_impurity_and_bath'.",
                          DeprecationWarning, stacklevel=2)
            fit_impurity_and_bath = deprecated_kwargs.pop('fit_imp_bath')
        if 'use_constrained_opt' in deprecated_kwargs:
            warnings.warn("DMET keyword 'use_constrained_opt' is deprecated; "
                          "use 'use_constrained_optimization'.",
                          DeprecationWarning, stacklevel=2)
            use_constrained_optimization = deprecated_kwargs.pop('use_constrained_opt')
        if deprecated_kwargs:
            raise TypeError(f"DMET.__init__ got unexpected keyword(s): {sorted(deprecated_kwargs)}")

        self.ints       = integrals
        self.norb       = self.ints.Norbs
        self.fragments  = fragments
        self.umat       = np.zeros([ self.norb, self.norb ], dtype=float)
        self.relaxation = 0.0

        self.NI_hack    = False
        self.method     = method
        self.doSCF      = False
        self.TransInv   = is_translation_invariant
        self.sc_method   = sc_method
        self.CC_E_TYPE   = CC_E_TYPE
        self.eom_nroots  = eom_nroots
        self.eom_type    = eom_type
        self.eom_koopmans = eom_koopmans
        self.eom_kwargs  = eom_kwargs or {}
        self.eom_results   = []   # populated by doexact() when method='EOM-CC'
        self.ncas          = ncas
        self.nelecas       = nelecas
        self.sa_nstates    = sa_nstates
        self.sa_weights    = sa_weights
        self.casscf_kwargs = casscf_kwargs or {}
        self.cas_select    = cas_select
        self.ao_labels     = ao_labels
        self.avas_threshold   = avas_threshold
        self.spade_gap_tol    = spade_gap_tol
        self.spade_n_fallback = spade_n_fallback
        self.embed_level_shift = embed_level_shift   # static level shift on the embedded post-HF ROHF/RHF reference
        self.rohf_stability = rohf_stability
        self.cas_multiseed = cas_multiseed
        self.cas_spin      = cas_spin   # target 2S for CAS states; None disables the spin penalty
        self.cas_spin_shift = cas_spin_shift   # fix_spin_ penalty strength for off-target spins
        self.natorb_occ_thresh   = natorb_occ_thresh    # natorb/spade core-vs-active occupation cutoff
        self.natorb_max_superset = natorb_max_superset  # cap on the natorb superset window size
        self.deg_tol             = deg_tol              # near-degenerate orbital energy tolerance
        self.casci_conv_tol      = casci_conv_tol       # natorb superset CASCI fcisolver.conv_tol
        self.cas_results   = []   # populated by doexact() when method='CASSCF'
        self.dmrg_kwargs   = dmrg_kwargs or {}
        self.dmrg_results  = []   # populated by doexact() when method='DMRG'
        self.mf_real       = mf_real
        self.qdnevpt2_kwargs   = qdnevpt2_kwargs or {}
        self.qdnevpt2_results  = []  # populated by doexact() when method='QD-NEVPT2'
        self.nevpt2_kwargs     = nevpt2_kwargs or {}
        self.nevpt2_results    = []  # populated by doexact() when method='NEVPT2'
        self.dft_results       = []  # populated by doexact() when method is DFT
        self.num_bath_orbs  = None
        self.fit_imp_bath = fit_impurity_and_bath
        self.do_det      = use_density_embedding
        self.do_det_NO   = use_density_embedding_no
        self.NOrotation = None
        self.altcostfunc = use_constrained_optimization
        self.allow_solver_fallback = allow_solver_fallback
        self.fragment_methods = dict(fragment_methods) if fragment_methods else {}
        for _idx, _m in self.fragment_methods.items():
            if not (0 <= _idx < len(fragments)):
                raise ValueError(
                    f"fragment_methods: fragment index {_idx} out of range "
                    f"(have {len(fragments)} fragments).")
            if _m != 'RHF':
                raise ValueError(
                    f"fragment_methods: only 'RHF' is supported per fragment, got '{_m}' "
                    f"for fragment {_idx}.")
        for _i in range(len(fragments)):
            if np.sum(np.asarray(fragments[_i])) < 0:
                warnings.warn(
                    f"RHF via a negative fragment mask (fragment {_i}) is deprecated; "
                    f"pass fragment_methods={{{_i}: 'RHF'}} instead.",
                    DeprecationWarning, stacklevel=2)
        self.oei_s      = None  # spin-dependent 1e potential for open-shell envs
        self.bath_tol   = bath_tol
        self.xc          = xc             # XC functional for DFT solvers
        self.level_shift = level_shift    # static level shift propagated to fragment solvers
        self.spin_polarized = spin_polarized  # enable independent alpha/beta mu optimization
        self.mm_coords  = np.asarray(mm_coords,  dtype=float) if mm_coords  is not None else None
        self.mm_charges = np.asarray(mm_charges, dtype=float) if mm_charges is not None else None

        self._validate_config()
        self._warn_inert_params()

        self.use_symmetry = use_symmetry
        self.symmetry_map = symmetry_map  # user-provided {child_idx: parent_idx} or None
        if self.use_symmetry and self.symmetry_map is None and not is_translation_invariant:
            self.symmetry_map = self._auto_detect_symmetry()

        self.parallel = parallel
        if max_workers is not None:
            self.max_workers = max_workers
        else:
            _env_threads = os.environ.get('PRISMdmet_WORKERS',
                           os.environ.get('SLURM_CPUS_PER_TASK',
                           os.environ.get('OMP_NUM_THREADS', '1')))
            self.max_workers = max(1, int(_env_threads))

        maxiter_frags = 1 if is_translation_invariant else len(fragments)
        self.frag_caches = [None] * maxiter_frags

        self.minFunc    = None
        if self.altcostfunc:
            self.minFunc = 'FOCK_INIT'  # 'oei'
            if self.fit_imp_bath:
                raise ValueError(
                    "use_constrained_optimization=True is incompatible with fit_impurity_and_bath=True; "
                    "set fit_impurity_and_bath=False for constrained optimization.")
            if self.do_det:
                raise ValueError(
                    "use_constrained_optimization=True is incompatible with use_density_embedding=True.")
            if self.sc_method not in {'BFGS', 'NONE'}:
                raise ValueError(
                    "use_constrained_optimization=True requires sc_method in {'BFGS', 'NONE'}.")

        if self.method == 'CC' and self.CC_E_TYPE == 'CASCI':
            if len(self.fragments) != 1:
                raise ValueError(
                    "method='CC' with CC_E_TYPE='CASCI' requires a single impurity cluster.")
        if self.method == 'CC' and self.CC_E_TYPE == 'EOM-CCSD':
            if eom_nroots < 1:
                raise ValueError("eom_nroots must be >= 1 when using CC_E_TYPE='EOM-CCSD'.")

        if self.do_det:
            self.fit_imp_bath = False
            if self.do_det_NO:
                self.NOvecs = None
                self.NOdiag = None

        self.print_u   = print_u
        self.print_rdm = print_rdm

        allOne = self.testclusters()
        if not allOne:
            # Incomplete tiling: impurity orbitals must be the first in the Hamiltonian.
            assert not self.TransInv

        self.energy   = 0.0
        self.imp_1RDM = []
        self.dmetOrbs = []
        self.imp_size = self.make_imp_size()
        self.mu_imp   = 0.0
        self._chempot_imp_beta = 0.0
        self.mask     = self.make_mask()
        self.helper   = prismdmet_helper.PrismDMETHelper( self.ints, self.makelist_H1(), self.altcostfunc, self.minFunc )

        self.time_ed  = 0.0
        self.time_cf  = 0.0
        self.time_func= 0.0
        self.time_grad= 0.0

        # When True, injects 0.5*(F_a - F_b) into embedded CASSCF/NEVPT2 (mrh-style). Default off.
        self.include_spin_oei = include_spin_oei

        # Auto-detect open-shell reference from localintegrals
        if hasattr(self.ints, 'loc_spin_oei'):
            self.oei_s = self.ints.loc_spin_oei()   # None for RHF, ndarray for ROHF/UHF

    @property
    def impClust( self ):
        # Backward-compatible alias for the renamed self.fragments.
        return self.fragments

    def _warn_inert_params( self ):
        '''Warn (not error) when a parameter is set but inert for the chosen method.'''
        import warnings
        _cas_methods = {'CASSCF', 'NEVPT2', 'QD-NEVPT2'}
        if self.ao_labels is not None and self.cas_select not in ('ao_character', 'avas'):
            warnings.warn(
                f"ao_labels is set but cas_select='{self.cas_select}' ignores it "
                "(only 'ao_character'/'avas' use ao_labels).", UserWarning)
        if self.cas_multiseed and self.method not in _cas_methods:
            warnings.warn(
                f"cas_multiseed=True is inert for method='{self.method}' "
                f"(only {sorted(_cas_methods)} use it).", UserWarning)
        if (self.mm_coords is not None or self.mm_charges is not None) \
                and self.method not in ('RKS', 'UKS', 'ROKS'):
            warnings.warn(
                f"mm_coords/mm_charges are set but method='{self.method}' is not a "
                "QM/MM-capable DFT method; they will be ignored.", UserWarning)
        if self.method in _cas_methods and self.cas_select == 'energy':
            warnings.warn(
                "cas_select='energy' (the legacy default) selects the active space by "
                "orbital energy; for production use cas_select='natorb'.", UserWarning)

    def _validate_config( self ):
        '''Fail fast at construction on user-reachable misconfiguration.'''
        if self.TransInv and not self.ints.TI_OK:
            raise ValueError(
                "translation-invariant DMET requires a TI-capable LocalIntegrals "
                "(TI_OK=True); the chosen localization sets TI_OK=False.")

        _valid_methods = {'ED', 'FCI', 'DMRG', 'CC', 'MP2', 'RHF',
                          'EOM-CC', 'CASSCF', 'QD-NEVPT2', 'NEVPT2',
                          'UHF', 'ROHF', 'RKS', 'UKS', 'ROKS'}
        if self.method not in _valid_methods:
            raise ValueError(
                f"DMET: unknown method='{self.method}'. Valid: {sorted(_valid_methods)}")

        if self.sc_method not in {'LSTSQ', 'BFGS', 'NONE'}:
            raise ValueError(
                f"DMET: unknown sc_method='{self.sc_method}'. Valid: ['BFGS', 'LSTSQ', 'NONE']")

        _valid_cc_etypes = {'LAMBDA', 'LAMBDA_AMP', 'LAMBDA_ZERO', 'CASCI',
                            'CCSD(T)', 'CCSD(T)_RDM', 'EOM-CCSD'}
        if self.CC_E_TYPE not in _valid_cc_etypes:
            raise ValueError(
                f"DMET: unknown CC_E_TYPE='{self.CC_E_TYPE}'. Valid: {sorted(_valid_cc_etypes)}")

        _valid_cas = {'energy', 'impurity', 'ao_character', 'avas', 'spade', 'natorb'}
        if self.cas_select not in _valid_cas:
            raise ValueError(
                f"DMET: unknown cas_select='{self.cas_select}'. Valid: {sorted(_valid_cas)}")

        if self.method in ('QD-NEVPT2', 'NEVPT2') and (self.ncas is None or self.nelecas is None):
            raise ValueError(
                f"method='{self.method}' requires ncas and nelecas (active space size).")

        if self.method == 'QD-NEVPT2' and self.sa_nstates < 2:
            raise ValueError(
                "method='QD-NEVPT2' requires sa_nstates >= 2 for state-averaging.")

        if self.cas_select in ('ao_character', 'avas') and self.ao_labels is None:
            raise ValueError(
                f"cas_select='{self.cas_select}' requires ao_labels.")

        if self.method == 'EOM-CC':
            from .solvers.eomcc import _VALID_EOM_TYPES as _EOM_SET
            if self.eom_type not in _EOM_SET:
                raise ValueError(
                    f"DMET: unknown eom_type='{self.eom_type}'. Valid: {sorted(_EOM_SET)}")

    ####################
    # Cluster and mask setup
    ####################

    def testclusters( self ):
    
        quicktest = np.zeros([ self.norb ], dtype=int)
        for item in self.fragments:
            quicktest += np.abs(item)
        assert( np.all( quicktest >= 0 ) )
        assert( np.all( quicktest <= 1 ) )
        allOne = np.all( quicktest == 1 )
        return allOne

    def _auto_detect_symmetry( self ):
        symmetry_map = {}
        seen = {}  # fingerprint -> first fragment index
        for idx, cluster in enumerate(self.fragments):
            fingerprint = int(np.sum(np.abs(cluster)))
            if fingerprint in seen:
                symmetry_map[idx] = seen[fingerprint]
            else:
                seen[fingerprint] = idx
        if symmetry_map:
            n_unique = len(self.fragments) - len(symmetry_map)
            print(f"Prismdmet :: symmetry : Auto-detected {n_unique} unique fragment(s) "
                  f"out of {len(self.fragments)} total. "
                  f"Skipping {len(symmetry_map)} equivalent fragment solve(s).")
        return symmetry_map
            
    def make_imp_size( self ):

        imp_sizes = []
        maxiter = len( self.fragments )
        if self.TransInv:
            maxiter = 1
        for counter in range( maxiter ):
            impurity_orbs = np.abs(self.fragments[ counter ])
            num_imp_orbs = np.sum( impurity_orbs )
            imp_sizes.append( num_imp_orbs )
        imp_sizes = np.array( imp_sizes )
        return imp_sizes

    def makelist_H1( self ):

        h1_terms = []
        if self.do_det: # Do density embedding theory
            if self.TransInv: # Translational invariance assumed
                localsize = self.imp_size[ 0 ]
                for row in range( localsize ):
                    h1 = np.zeros( [ self.norb, self.norb ], dtype=int )
                    for jumper in range( self.norb // localsize ):
                        jumpsquare = localsize * jumper
                        h1[ jumpsquare + row, jumpsquare + row ] = 1
                    h1_terms.append( h1 )
            else: # NO translational invariance assumed
                jumpsquare = 0
                for localsize in self.imp_size:
                    for row in range( localsize ):
                        h1 = np.zeros( [ self.norb, self.norb ], dtype=int )
                        h1[ jumpsquare + row, jumpsquare + row ] = 1
                        h1_terms.append( h1 )
                    jumpsquare += localsize
        else: # Do density MATRIX embedding theory
            if self.TransInv: # Translational invariance assumed
                localsize = self.imp_size[ 0 ]
                for row in range( localsize ):
                    for col in range( row, localsize ):
                        h1 = np.zeros( [ self.norb, self.norb ], dtype=int )
                        for jumper in range( self.norb // localsize ):
                            jumpsquare = localsize * jumper
                            h1[ jumpsquare + row, jumpsquare + col ] = 1
                            h1[ jumpsquare + col, jumpsquare + row ] = 1
                        h1_terms.append( h1 )
            else: # NO translational invariance assumed
                jumpsquare = 0
                for localsize in self.imp_size:
                    for row in range( localsize ):
                        for col in range( row, localsize ):
                            h1 = np.zeros( [ self.norb, self.norb ], dtype=int )
                            h1[ jumpsquare + row, jumpsquare + col ] = 1
                            h1[ jumpsquare + col, jumpsquare + row ] = 1
                            h1_terms.append( h1 )
                    jumpsquare += localsize
        return h1_terms

    def make_mask( self ):

        mask = np.zeros( [ self.norb, self.norb ], dtype=bool )
        if self.do_det: # Do density embedding theory
            jump = 0
            for localsize in self.imp_size: # self.imp_size has length 1 if self.TransInv
                for row in range( localsize ):
                    mask[ jump + row, jump + row ] = True
                jump += localsize
        else: # Do density MATRIX embedding theory
            jump = 0
            for localsize in self.imp_size: # self.imp_size has length 1 if self.TransInv
                for row in range( localsize ):
                    for col in range( row, localsize ):
                        mask[ jump + row, jump + col ] = True
                jump += localsize
        return mask

    ####################
    # DMET driver
    ####################

    def doexact( self, chempot_imp=0.0 ):
    
        one_rdm = self.helper.construct1RDM_loc( self.doSCF, self.umat )
        self.energy   = 0.0
        self.imp_1RDM = []
        self.dmetOrbs = []
        self.frag_energies    = []
        self._spinpol_rdms    = []
        self.dft_results      = []
        self.eom_results      = []
        self.cas_results      = []
        self.qdnevpt2_results = []
        self.nevpt2_results   = []
        self.dmrg_results     = []
        if self.do_det and self.do_det_NO:
            self.NOvecs = []
            self.NOdiag = []

        maxiter = len( self.fragments )
        if self.TransInv:
            maxiter = 1
            
        remainingOrbs = np.ones( [ len( self.fragments[ 0 ] ) ], dtype=float )

        _frag_tasks   = []
        _frag_meta    = []
        _sym_counters = set()

        _builder = FragmentBuilder(
            ints      = self.ints,
            helper    = self.helper,
            fragments = self.fragments,
            method    = self.method,
            num_bath_orbs = self.num_bath_orbs,
            NI_hack   = self.NI_hack,
            umat      = self.umat,
            bath_tol  = self.bath_tol,
            fragment_methods = self.fragment_methods,
        )

        for counter in range( maxiter ):

            # Symmetry skip: handled immediately without a solver.
            if (self.symmetry_map is not None and counter in self.symmetry_map
                    and not self.TransInv):
                _sym_counters.add(counter)
                sym_desc = _builder.build_symmetry_bath(
                    counter, one_rdm, self.symmetry_map[counter])
                self.dmetOrbs.append(sym_desc['loc_2_dmet'][:, :sym_desc['norb_in_imp']])
                _frag_meta.append({
                    'counter'     : counter,
                    'sym_parent'  : sym_desc['sym_parent'],
                    'impurity_orbs': sym_desc['impurity_orbs'],
                })
                continue

            frag = _builder.build(counter, one_rdm, chempot_imp)

            # Unpack frequently-used local names (improves readability below)
            impurity_orbs = frag['impurity_orbs']
            num_imp_orbs   = frag['num_imp_orbs']
            norb_in_imp  = frag['norb_in_imp']
            nelec_in_imp = frag['nelec_in_imp']
            loc_2_dmet     = frag['loc_2_dmet']
            core_1rdm_loc = frag['core_1rdm_loc']
            core_1rdm_dmet= frag['core_1rdm_dmet']
            dmet_oei      = frag['dmet_oei']
            dmet_fock     = frag['dmet_fock']
            dmet_tei      = frag['dmet_tei']
            dm_guess_rhf   = frag['dm_guess_rhf']
            _method_key  = frag['method_key']

            # Populate dmetOrbs for bath-dump and cost-function use
            self.dmetOrbs.append(loc_2_dmet[:, :norb_in_imp])

            print("DMET::exact : Performing a (", norb_in_imp, "orb,",
                  nelec_in_imp, "el ) dmet active space calculation.")

            _frag_meta.append({
                'counter'     : counter,
                'sym_parent'  : None,
                'impurity_orbs': impurity_orbs,
                'num_imp_orbs'  : num_imp_orbs,
                'norb_in_imp' : norb_in_imp,
                'nelec_in_imp': nelec_in_imp,
                'core_1rdm_loc': core_1rdm_loc,
                'method_key'  : _method_key,
                'loc_2_dmet'    : loc_2_dmet,
                'dmet_oei'     : dmet_oei,
                'dmet_fock'    : dmet_fock,
            })

            _mo_guess = None
            _ci_guess = None
            if _method_key == 'CASSCF' and self.frag_caches[counter] is not None:
                cached = self.frag_caches[counter]
                old_mo    = cached['mo_coeff']
                _ci_guess = cached.get('ci', None)
                _ncas  = self.ncas if self.ncas is not None else norb_in_imp
                _ncore = (nelec_in_imp - (self.nelecas if self.nelecas is not None
                                          else nelec_in_imp)) // 2
                if old_mo.shape == (norb_in_imp, norb_in_imp):
                    from .solvers.qcsolver_utils import project_amo_manually
                    _mo_guess, fidelity = project_amo_manually(
                        old_mo, _ncas, _ncore, dmet_fock, norb_in_imp)
                    if np.min(fidelity) < 0.5:
                        print("DMET::CASSCF : Low projection fidelity, discarding CI guess.")
                        _ci_guess = None
                else:
                    print("DMET::CASSCF : MO shape mismatch, starting fresh.")

            # Spin-dependent 1e potential; printed as diagnostic, injected only if include_spin_oei=True.
            _dmet_oei_s = None
            if (_method_key in ('CASSCF', 'NEVPT2', 'QD-NEVPT2')
                    and hasattr(self.ints, 'dmet_oei_s')):
                _vsp = self.ints.dmet_oei_s(loc_2_dmet, norb_in_imp)
                if _vsp is not None:
                    _tag = 'APPLIED' if self.include_spin_oei else 'diagnostic only'
                    print(f"DMET :: spin-oei : ||dmet_oei_s||_F (cluster, {norb_in_imp} orb) "
                          f"= {np.linalg.norm(_vsp):.6f}  [{_tag}]")
                    if self.include_spin_oei:
                        _dmet_oei_s = _vsp

            task = self._build_task(
                counter, _method_key, dmet_oei, dmet_fock, dmet_tei,
                norb_in_imp, nelec_in_imp, num_imp_orbs, chempot_imp,
                dm_guess_rhf, loc_2_dmet,
                mo_guess=_mo_guess, ci_guess=_ci_guess, oei_s=_dmet_oei_s)

            from .solvers import PARALLEL_ELIGIBLE
            _is_parallel_eligible = (
                self.parallel
                and _method_key in PARALLEL_ELIGIBLE
                and not self.do_det_NO  # NO rotation requires in-process state
            )

            if _is_parallel_eligible:
                _frag_tasks.append(task)
            else:
                _frag_meta[-1]['sequential_result'] = self._run_fragment_sequential(task)

        _parallel_results = {}   # counter -> result dict
        if _frag_tasks:
            _nw = min(self.max_workers, len(_frag_tasks))
            print(f"Prismdmet :: parallel : Submitting {len(_frag_tasks)} fragment(s) "
                  f"to {_nw} worker process(es).")
            with concurrent.futures.ProcessPoolExecutor(max_workers=_nw) as pool:
                futures = {pool.submit(_fragment_worker, t): t['counter'] for t in _frag_tasks}
                for fut in concurrent.futures.as_completed(futures):
                    res = fut.result()
                    _parallel_results[res['counter']] = res

        for meta in _frag_meta:
            counter = meta['counter']
            impurity_orbs = meta['impurity_orbs']

            if meta.get('sym_parent') is not None:
                parent = meta['sym_parent']
                print(f"Prismdmet :: symmetry : Fragment {counter} <- fragment {parent} (copied).")
                parent_energy = self.frag_energies[parent]
                parent_rdm    = self.imp_1RDM[parent]
                self.energy += parent_energy
                self.frag_energies.append(parent_energy)
                self.imp_1RDM.append(parent_rdm.copy())
                self._spinpol_rdms.append(self._spinpol_rdms[parent])
                remainingOrbs -= impurity_orbs
                continue

            num_imp_orbs = meta['num_imp_orbs']

            if counter in _parallel_results:
                res = _parallel_results[counter]
            else:
                res = meta['sequential_result']

            IMP_energy = res['energy']
            IMP_1RDM   = res['rdm1']

            if res.get('fallback_from'):
                print(
                    f"Prismdmet :: WARNING : Fragment {counter} — "
                    f"'{res['fallback_from']}' failed (OOM); "
                    f"result computed with '{res.get('method', meta['method_key'])}'. "
                    f"Energy may be less accurate."
                )

            if 'eom_res' in res:
                self.eom_results.append(res['eom_res'])
            if 'cas_res' in res:
                self.frag_caches[counter] = {
                    'mo_coeff': res['cas_res']['mo_coeff'],
                    'ci'      : res['cas_res']['ci'],
                }
                self.cas_results.append(res['cas_res'])
            if 'qdnevpt2_res' in res:
                self.qdnevpt2_results.append(res['qdnevpt2_res'])
            if 'nevpt2_res' in res:
                self.nevpt2_results.append(res['nevpt2_res'])
            if 'dmrg_res' in res:
                self.dmrg_results.append(res['dmrg_res'])
            if 'dft_res' in res:
                self.dft_results.append(res['dft_res'])

            self.energy += IMP_energy
            self.frag_energies.append(IMP_energy)
            self.imp_1RDM.append( IMP_1RDM )
            self._spinpol_rdms.append(
                {'rdm1_alpha': res['rdm1_alpha'], 'rdm1_beta': res['rdm1_beta']}
                if 'rdm1_alpha' in res
                else {'rdm1_alpha': 0.5 * IMP_1RDM, 'rdm1_beta': 0.5 * IMP_1RDM})
            if self.do_det and self.do_det_NO:
                RDMeigenvals, RDMeigenvecs = np.linalg.eigh( IMP_1RDM[ :num_imp_orbs, :num_imp_orbs ] )
                self.NOvecs.append( RDMeigenvecs )
                self.NOdiag.append( RDMeigenvals )

            remainingOrbs -= impurity_orbs
        
        if self.do_det and self.do_det_NO:
            self.NOrotation = self.constructNOrotation()
        
        Nelectrons = 0.0
        for counter in range( maxiter ):
            Nelectrons += np.trace( self.imp_1RDM[counter][ :self.imp_size[counter], :self.imp_size[counter] ] )
        if self.TransInv:
            Nelectrons = Nelectrons * len( self.fragments )
            self.energy = self.energy * len( self.fragments )
            remainingOrbs[:] = 0
            
        # Augment energy with remaining HF contribution for incomplete impurity tilings
        if ( np.sum( remainingOrbs ) != 0 ):

            if ( self.CC_E_TYPE == 'CASCI' ):
                # CASCI energy omits the frozen-core constant; add 0.5 * Tr[core1RDM (oei+fock)]
                assert( maxiter == 1 )
                transfo = np.eye( self.norb, dtype=float )
                totalOEI  = self.ints.dmet_oei(  transfo, self.norb )
                totalFOCK = self.ints.dmet_fock( transfo, self.norb, core_1rdm_loc )
                self.energy += 0.5 * np.einsum( 'ij,ij->', core_1rdm_loc, totalOEI + totalFOCK )
                Nelectrons = np.trace( self.imp_1RDM[ 0 ] ) + np.trace( core_1rdm_loc ) # Because full active space is used to compute the energy
            else:
                assert (np.array_equal(self.ints.active, np.ones([self.ints.mol.nao_nr()], dtype=int)))

                # Global-Fock mean-field energy E = 0.5 Tr[gamma (h + F)] of the
                # orbitals no fragment covers (as in libDMET/Vayesta/mrh).
                # activeOEI and activeFOCK already carry the QM/MM potential and
                # DFT veff, so no environment SCF is needed; running one on the
                # bare molecule would drop the MM field and use HF exchange.
                impOrbs = remainingOrbs == 1
                h_plus_F = self.ints.activeOEI + self.ints.activeFOCK

                ImpEnergy = \
                     0.25 * np.einsum('ji,ij->', one_rdm[:, impOrbs], h_plus_F[impOrbs, :]) \
                   + 0.25 * np.einsum('ji,ij->', one_rdm[impOrbs, :], h_plus_F[:, impOrbs])

                self.energy += ImpEnergy
                Nelectrons += np.trace(one_rdm[np.ix_(impOrbs, impOrbs)])

            remainingOrbs[ remainingOrbs==1 ] -= 1
        assert( np.all( remainingOrbs == 0 ) )
            
        self.energy += self.ints.const()
        return Nelectrons

    def _build_task(self, counter, method_key, dmet_oei, dmet_fock, dmet_tei,
                    norb_in_imp, nelec_in_imp, num_imp_orbs, chempot_imp,
                    dm_guess_rhf, loc_2_dmet, mo_guess=None, ci_guess=None, oei_s=None):
        _is_dft = method_key in ('RKS', 'UKS', 'ROKS')
        _is_avas_like = self.cas_select in ('ao_character', 'avas')
        _dft_mol_info = {}
        if _is_dft:
            _dft_mol_info = {
                'dft_mol_dumps'        : self.ints.mol.dumps(),
                'ao2loc'               : self.ints.ao2loc,
                'loc_2_dmet'           : loc_2_dmet[:, :norb_in_imp],
                'mm_coords'            : self.mm_coords,
                'mm_charges'           : self.mm_charges,
                'nel_total'            : self.ints.Nelec,
                'dm_canonical_alpha'   : self.ints.fullDMao_alpha,
                'dm_canonical_beta'    : self.ints.fullDMao_beta,
                'fock_canonical_alpha' : self.ints.fullFOCKao_alpha,
                'fock_canonical_beta'  : self.ints.fullFOCKao_beta,
                'level_shift'          : self.level_shift,
                'use_density_fit'      : self.ints.use_density_fit,
                'df_auxbasis'          : self.ints.df_auxbasis,
            }
        return {
            'counter'           : counter,
            'method'            : method_key,
            'const'             : 0.0,
            'dmet_oei'          : dmet_oei,
            'dmet_fock'         : dmet_fock,
            'dmet_tei'          : dmet_tei,
            'norb'              : norb_in_imp,
            'nel'               : nelec_in_imp,
            'nimp'              : num_imp_orbs,
            'chempot_imp'       : chempot_imp,
            'chempot_imp_beta'  : self._chempot_imp_beta if self.spin_polarized else None,
            'dm_guess_rhf'      : dm_guess_rhf,
            'CC_E_TYPE'         : self.CC_E_TYPE,
            'eom_nroots'        : self.eom_nroots,
            'eom_type'          : self.eom_type,
            'eom_koopmans'      : self.eom_koopmans,
            'eom_kwargs'        : self.eom_kwargs,
            'ncas'              : self.ncas,
            'nelecas'           : self.nelecas,
            'sa_nstates'        : self.sa_nstates,
            'sa_weights'        : self.sa_weights,
            'cas_select'        : self.cas_select,
            'ao_labels'         : self.ao_labels,
            'ao_mol_dumps'      : self.ints.mol.dumps() if _is_avas_like else None,
            'ao2eo'             : (self.ints.ao2loc @ loc_2_dmet[:, :norb_in_imp]) if _is_avas_like else None,
            'avas_threshold'    : self.avas_threshold,
            'spade_gap_tol'     : self.spade_gap_tol,
            'spade_n_fallback'  : self.spade_n_fallback,
            'natorb_occ_thresh'   : self.natorb_occ_thresh,
            'natorb_max_superset' : self.natorb_max_superset,
            'deg_tol'             : self.deg_tol,
            'casci_conv_tol'      : self.casci_conv_tol,
            'embed_level_shift' : self.embed_level_shift,
            'rohf_stability'    : self.rohf_stability,
            'cas_multiseed'     : self.cas_multiseed,
            'cas_spin'          : self.cas_spin,
            'cas_spin_shift'    : self.cas_spin_shift,
            'casscf_kwargs'     : self.casscf_kwargs,
            'dmrg_kwargs'       : self.dmrg_kwargs,
            'mo_guess'          : mo_guess,
            'ci_guess'          : ci_guess,
            'oei_s'             : oei_s,
            'nevpt2_kwargs'     : self.nevpt2_kwargs,
            'qdnevpt2_kwargs'   : self.qdnevpt2_kwargs,
            'xc'                : self.xc,
            'spin'              : self.ints.mol.spin,
            'spin_polarized'    : self.spin_polarized,
            'allow_solver_fallback' : self.allow_solver_fallback,
            **_dft_mol_info,
        }

    def _run_fragment_sequential(self, task):
        result = SolverDispatcher.execute(task)
        if 'cas_res' in result:
            cas_res = result['cas_res']
            self.frag_caches[task['counter']] = {
                'mo_coeff': cas_res['mo_coeff'],
                'ci'      : cas_res['ci'],
            }
        return result

    ####################
    # u-matrix fitting: cost functions and gradients
    ####################

    def constructNOrotation( self ):
    
        myNOrotation = np.zeros( [ self.norb, self.norb ], dtype=float )
        jumpsquare = 0
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            myNOrotation[ jumpsquare : jumpsquare + self.imp_size[ count ], jumpsquare : jumpsquare + self.imp_size[ count ] ] = self.NOvecs[ count ]
            jumpsquare += self.imp_size[ count ]
        for count in range( jumpsquare, self.norb ):
            myNOrotation[ count, count ] = 1.0
        if self.TransInv:
            size = self.imp_size[ 0 ]
            for it in range( 1, self.norb // size ):
                myNOrotation[ it*size:(it+1)*size, it*size:(it+1)*size ] = myNOrotation[ 0:size, 0:size ]
        return myNOrotation
        
    def costfunction( self, newumatflat ):

        return np.linalg.norm( self.rdm_differences( newumatflat ) )**2

    def alt_costfunction( self, newumatflat ):

        newumatsquare_loc = self.flat2square( newumatflat )
        one_rdm_loc = self.helper.construct1RDM_loc( self.doSCF, newumatsquare_loc )

        errors    = self.rdm_differences_bis( newumatflat )
        errors_sq = self.flat2square (errors)

        if self.minFunc == 'oei' :
            e_fun = np.trace( np.dot(self.ints.loc_oei(), one_rdm_loc) )
        elif self.minFunc == 'FOCK_INIT' :
            e_fun = np.trace( np.dot(self.ints.loc_fock(), one_rdm_loc) )
        e_cstr = np.sum( newumatsquare_loc * errors_sq )
        return -e_fun-e_cstr
        
    def costfunction_derivative( self, newumatflat ):
        
        errors = self.rdm_differences( newumatflat )
        error_derivs = self.rdm_differences_derivative( newumatflat )
        thegradient = np.zeros([ len( newumatflat ) ], dtype=float)
        for counter in range( len( newumatflat ) ):
            thegradient[ counter ] = 2 * np.sum( np.multiply( error_derivs[ : , counter ], errors ) )
        return thegradient

    def alt_costfunction_derivative( self, newumatflat ):
        
        errors = self.rdm_differences_bis( newumatflat )
        return -errors
    
    def rdm_differences( self, newumatflat ):
    
        start_func = time.time()
    
        newumatsquare_loc = self.flat2square( newumatflat )
        one_rdm_loc = self.helper.construct1RDM_loc( self.doSCF, newumatsquare_loc )
        
        n_cluster_orbs = 0
        for count in range(len(self.imp_size)):
            if self.do_det: # Do density embedding theory: fit only impurity
                n_cluster_orbs += self.imp_size[ count ]
                assert not self.fit_imp_bath
            else: # Do density MATRIX embedding theory
                if self.fit_imp_bath:
                    n_cluster_orbs += self.dmetOrbs[count].shape[1] * self.dmetOrbs[count].shape[1]
                else:
                    n_cluster_orbs += self.imp_size[ count ] * self.imp_size[ count ]
        errors = np.zeros( [ n_cluster_orbs ], dtype=float )
        
        jump = 0
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            if self.fit_imp_bath:
                mf_1RDM = np.dot( np.dot( self.dmetOrbs[ count ].T, one_rdm_loc ), self.dmetOrbs[ count ] )
                ed_1RDM = self.imp_1RDM[count]
            else:
                mf_1RDM = (one_rdm_loc[:,np.flatnonzero(self.fragments[count])])[np.flatnonzero(self.fragments[count]),:]
                ed_1RDM = self.imp_1RDM[count][:self.imp_size[count],:self.imp_size[count]]
            if self.do_det: # Do density embedding theory
                if self.do_det_NO: # Work in the NO basis
                    rdm_mismatch = np.diag( np.dot( np.dot( self.NOvecs[ count ].T, mf_1RDM ), self.NOvecs[ count ] ) ) - self.NOdiag[ count ]
                else: # Work in the lattice basis
                    rdm_mismatch = np.diag( mf_1RDM - ed_1RDM )
                errors[ jump : jump + len( rdm_mismatch ) ] = rdm_mismatch
                jump += len( rdm_mismatch )
            else: # Do density MATRIX embedding theory
                rdm_mismatch = mf_1RDM - ed_1RDM
                squaresize = rdm_mismatch.shape[0] * rdm_mismatch.shape[1]
                errors[ jump : jump + squaresize ] = np.reshape( rdm_mismatch, squaresize, order='F' )
                jump += squaresize
        assert ( jump == n_cluster_orbs )
        
        stop_func = time.time()
        self.time_func += ( stop_func - start_func )
        
        return errors
        
    def rdm_differences_bis( self, newumatflat ):
    
        start_func = time.time()
    
        newumatsquare_loc = self.flat2square( newumatflat )
        one_rdm_loc = self.helper.construct1RDM_loc( self.doSCF, newumatsquare_loc )

        n_cluster_orbs = 0
        jump = 0
        for count in range(len(self.imp_size)):
            # n_cluster_orbs += self.imp_size[ count ] * self.imp_size[ count ]
            mask_t = self.mask[ np.ix_(list(range(jump,jump+self.imp_size[count])),list(range(jump,jump+self.imp_size[count]))) ]
            n_cluster_orbs += np.count_nonzero( mask_t )
            jump += self.imp_size[count]
        errors = np.zeros( [ n_cluster_orbs ], dtype=float )
        
        jump = 0
        jumpc = 0
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            mf_1RDM = (one_rdm_loc[:,np.flatnonzero(self.fragments[count])])[np.flatnonzero(self.fragments[count]),:]
            ed_1RDM = self.imp_1RDM[count][:self.imp_size[count],:self.imp_size[count]]
            rdm_mismatch = mf_1RDM - ed_1RDM
            mask_t = self.mask[ np.ix_(list(range(jumpc,jumpc+self.imp_size[count])),list(range(jumpc,jumpc+self.imp_size[count]))) ]
            squaresize = np.count_nonzero( mask_t )
            errors[ jump : jump + squaresize ] = np.reshape( rdm_mismatch[mask_t], squaresize, order='F' )
            jump  += squaresize
            jumpc += self.imp_size[count]
        assert ( jump == n_cluster_orbs )
        
        stop_func = time.time()
        self.time_func += ( stop_func - start_func )
        
        return errors

    def rdm_differences_derivative( self, newumatflat ):
        
        start_grad = time.time()
        
        newumatsquare_loc = self.flat2square( newumatflat )
        RDMderivs_rot = self.helper.construct1RDM_response( self.doSCF, newumatsquare_loc, self.NOrotation )
        
        n_cluster_orbs = 0
        for count in range(len(self.imp_size)):
            if self.do_det: # Do density embedding theory: fit only impurity
                n_cluster_orbs += self.imp_size[ count ]
                assert not self.fit_imp_bath
            else: # Do density MATRIX embedding theory
                if self.fit_imp_bath:
                    n_cluster_orbs += self.dmetOrbs[count].shape[1] * self.dmetOrbs[count].shape[1]
                else:
                    n_cluster_orbs += self.imp_size[ count ] * self.imp_size[ count ]
        
        gradient = []
        for countgr in range( len( newumatflat ) ):
            error_deriv = np.zeros( [ n_cluster_orbs ], dtype=float )
            jump = 0
            jumpsquare = 0
            for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
                if self.fit_imp_bath:
                    local_derivative = np.dot( np.dot( self.dmetOrbs[ count ].T, RDMderivs_rot[ countgr, :, : ] ), self.dmetOrbs[ count ] )
                else:
                    if self.do_det and self.do_det_NO:
                        local_derivative = RDMderivs_rot[ countgr, jumpsquare : jumpsquare + self.imp_size[ count ],\
                                                                   jumpsquare : jumpsquare + self.imp_size[ count ] ]
                        jumpsquare += self.imp_size[ count ]
                    else:
                        local_derivative = ((RDMderivs_rot[ countgr, :, : ])[:,np.flatnonzero(self.fragments[count])])[np.flatnonzero(self.fragments[count]),:]
                if self.do_det: # Do density embedding theory
                    local_derivative = np.diag( local_derivative )
                    error_deriv[ jump : jump + len( local_derivative ) ] = local_derivative
                    jump += len( local_derivative )
                else: # Do density MATRIX embedding theory
                    squaresize = local_derivative.shape[0] * local_derivative.shape[1]
                    error_deriv[ jump : jump + squaresize ] = np.reshape( local_derivative, squaresize, order='F' )
                    jump += squaresize
            assert ( jump == n_cluster_orbs )
            gradient.append( error_deriv )
        gradient = np.array( gradient ).T
        
        stop_grad = time.time()
        self.time_grad += ( stop_grad - start_grad )
        
        return gradient
        
    def verify_gradient( self, umatflat ):
    
        gradient = self.costfunction_derivative( umatflat )
        cost_reference = self.costfunction( umatflat )
        gradientbis = np.zeros( [ len( gradient ) ], dtype=float )
        stepsize = 1e-7
        for cnt in range( len( gradient ) ):
            umatbis = np.array( umatflat, copy=True )
            umatbis[cnt] += stepsize
            costbis = self.costfunction( umatbis )
            gradientbis[ cnt ] = ( costbis - cost_reference ) / stepsize
        print("   Norm( gradient difference ) =", np.linalg.norm( gradient - gradientbis ))
        print("   Norm( gradient )            =", np.linalg.norm( gradient ))
        
    def hessian_eigenvalues( self, umatflat ):
    
        stepsize = 1e-7
        gradient_reference = self.costfunction_derivative( umatflat )
        hessian = np.zeros( [ len( umatflat ), len( umatflat ) ], dtype=float )
        for cnt in range( len( umatflat ) ):
            umat_perturbed = umatflat.copy()
            umat_perturbed[ cnt ] += stepsize
            gradient = self.costfunction_derivative( umat_perturbed )
            hessian[ :, cnt ] = ( gradient - gradient_reference ) / stepsize
        hessian = 0.5 * ( hessian + hessian.T )
        eigvals, eigvecs = np.linalg.eigh( hessian )
        idx = eigvals.argsort()
        eigvals = eigvals[ idx ]
        print("Hessian eigenvalues =", eigvals)
        
    def flat2square( self, umatflat ):
    
        umatsquare = np.zeros( [ self.norb, self.norb ], dtype=float )
        umatsquare[ self.mask ] = umatflat
        umatsquare = umatsquare.T
        umatsquare[ self.mask ] = umatflat
        if self.TransInv:
            size = self.imp_size[ 0 ]
            for it in range( 1, self.norb // size ):
                umatsquare[ it*size:(it+1)*size, it*size:(it+1)*size ] = umatsquare[ 0:size, 0:size ]

        if self.NOrotation is not None:
            umatsquare = np.dot( np.dot( self.NOrotation, umatsquare ), self.NOrotation.T )
        return umatsquare
        
    def square2flat( self, umatsquare ):
    
        umatsquare_bis = np.array( umatsquare, copy=True )
        if self.NOrotation is not None:
            umatsquare_bis = np.dot( np.dot( self.NOrotation.T, umatsquare_bis ), self.NOrotation )
        umatflat = umatsquare_bis[ self.mask ]
        return umatflat
        
    # Maximum physically reasonable chemical potential (Eh). Newton steps beyond
    # this indicate the optimizer has lost contact with the electron-count response
    # surface — common for DFT-in-DFT where impurity orbitals are fully occupied.
    _MU_MAX = 10.0

    def numeleccostfunction( self, chempot_imp ):
        if abs(chempot_imp) > self._MU_MAX:
            raise RuntimeError(
                f"mu optimization diverged: |mu| = {abs(chempot_imp):.2f} Eh "
                f"exceeds threshold {self._MU_MAX} Eh. "
                "For DFT-in-DFT DMET, optimize_mu=True is not recommended."
            )
        Nelec_dmet   = self.doexact( chempot_imp )
        Nelec_target = self.ints.Nelec
        print("      (chemical potential , number of electrons) = (", chempot_imp, "," , Nelec_dmet ,")")
        return Nelec_dmet - Nelec_target

    def numeleccostfunction_spinpol( self, chempot_pair ):
        mu_a, mu_b = float(chempot_pair[0]), float(chempot_pair[1])
        if max(abs(mu_a), abs(mu_b)) > self._MU_MAX:
            raise RuntimeError(
                f"mu optimization diverged: |mu| = ({abs(mu_a):.2f}, {abs(mu_b):.2f}) Eh "
                f"exceeds threshold {self._MU_MAX} Eh. "
                "For DFT-in-DFT DMET, optimize_mu=True is not recommended."
            )
        self._chempot_imp_beta = mu_b
        Nelec_total = self.doexact(mu_a)
        Nelec_a = sum(np.trace(rdm['rdm1_alpha'][:s, :s])
                      for rdm, s in zip(self._spinpol_rdms, self.imp_size))
        Nelec_b = Nelec_total - Nelec_a
        target_a = self.ints.Nelec_alpha if hasattr(self.ints, 'Nelec_alpha') \
                   else self.ints.Nelec / 2.0
        target_b = self.ints.Nelec_beta if hasattr(self.ints, 'Nelec_beta') \
                   else self.ints.Nelec / 2.0
        print(f"      (mu_a, mu_b, N_a, N_b) = ({mu_a:.6f}, {mu_b:.6f}, "
              f"{Nelec_a:.4f}, {Nelec_b:.4f})")
        return np.array([Nelec_a - target_a, Nelec_b - target_b])

    ####################
    # Self-consistency, one-shot, and output
    ####################

    def selfconsistent( self ):

        if self.method in ('EOM-CC', 'QD-NEVPT2', 'NEVPT2'):
            _labels = {
                'EOM-CC'    : ("EOM-CCSD",  "provides excited-state energies, not a ground-state u-matrix"),
                'QD-NEVPT2' : ("QD-NEVPT2", "perturbative correction on a fixed CASSCF reference — "
                               "self-consistent u-matrix iteration is not defined for this method"),
                'NEVPT2'    : ("NEVPT2",    "perturbative correction on a fixed CASSCF reference — "
                               "self-consistent u-matrix iteration is not defined for this method"),
            }
            label, reason = _labels[self.method]
            raise RuntimeError(
                f"method='{self.method}' is only compatible with one-shot dmet (oneshot()). "
                f"Self-consistent dmet with {label} is not supported: {reason}."
            )

        iteration = 0
        u_diff = 1.0
        convergence_threshold = 1e-5
        print("RHF energy =", self.ints.fullEhf)
        
        while ( u_diff > convergence_threshold ):
        
            iteration += 1
            print("dmet iteration", iteration)
            umat_old = np.array( self.umat, copy=True )
            rdm_old = self.transform_ed_1rdm() # At the very first iteration, this matrix will be zero
            
            # Find the chemical potential for the correlated impurity problem
            start_ed = time.time()
            if (( self.method == 'CC' ) and ( self.CC_E_TYPE == 'CASCI' )):
                self.mu_imp = 0.0
                self.doexact( self.mu_imp )
            try:
                self.mu_imp = optimize.newton( self.numeleccostfunction, self.mu_imp )
            except RuntimeError:
                print("Warning: newton solver for chemical potential did not perfectly converge. Proceeding with last evaluated chemical potential.")
            print("   Chemical potential =", self.mu_imp)
            stop_ed = time.time()
            self.time_ed += ( stop_ed - start_ed )
            print("   Energy =", self.energy)
            if ( self.sc_method != 'NONE' and not(self.altcostfunc) ):
                self.hessian_eigenvalues( self.square2flat( self.umat ) )

            # Optimize the u-matrix
            start_cf = time.time()
            if ( self.altcostfunc and self.sc_method == 'BFGS' ):
                result = optimize.minimize( self.alt_costfunction, self.square2flat( self.umat ), jac=self.alt_costfunction_derivative, options={'disp': False} )
                self.umat = self.flat2square( result.x )
            elif ( self.sc_method == 'LSTSQ' ):
                result = optimize.leastsq( self.rdm_differences, self.square2flat( self.umat ), Dfun=self.rdm_differences_derivative, factor=0.1 )
                self.umat = self.flat2square( result[ 0 ] )
            elif ( self.sc_method == 'BFGS' ):
                result = optimize.minimize( self.costfunction, self.square2flat( self.umat ), jac=self.costfunction_derivative, options={'disp': False} )
                self.umat = self.flat2square( result.x )
            self.umat = self.umat - np.eye( self.umat.shape[ 0 ] ) * np.average( np.diag( self.umat ) )  # Remove arbitrary global shift
            if ( self.altcostfunc ):
                print("   Cost function after convergence =", self.alt_costfunction( self.square2flat( self.umat ) ))
            else:
                print("   Cost function after convergence =", self.costfunction( self.square2flat( self.umat ) ))
            stop_cf = time.time()
            self.time_cf += ( stop_cf - start_cf )

            if self.print_u:
                self.print_umat()
            if self.print_rdm:
                self.print_1rdm()

            # Convergence check
            u_diff   = np.linalg.norm( umat_old - self.umat )
            rdm_diff = np.linalg.norm( rdm_old - self.transform_ed_1rdm() )
            self.umat = self.relaxation * umat_old + ( 1.0 - self.relaxation ) * self.umat
            print("   2-norm of difference old and new u-mat =", u_diff)
            print("   2-norm of difference old and new 1-RDM =", rdm_diff)
            
            if ( self.sc_method == 'NONE' ):
                u_diff = 0.1 * convergence_threshold # Do only 1 iteration
        
        print("Time cf func =", self.time_func)
        print("Time cf grad =", self.time_grad)
        print("Time dmet ed =", self.time_ed)
        print("Time dmet cf =", self.time_cf)
        
        return self.energy

    def doselfconsistent(self):
        from warnings import warn
        warn("'doselfconsistent()' is deprecated; use 'selfconsistent()'", DeprecationWarning, stacklevel=2)
        return self.selfconsistent()

    def print_umat( self ):
    
        print("The u-matrix =")
        squarejumper = 0
        for localsize in self.imp_size: # self.imp_size has length 1 if self.TransInv
            print(self.umat[ squarejumper:squarejumper+localsize , squarejumper:squarejumper+localsize ])
            squarejumper += localsize
    
    def print_1rdm( self ):
    
        print("The ED 1-RDM of the impurities ( + baths ) =")
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            print(self.imp_1RDM[ count ])
            
    def transform_ed_1rdm( self ):
    
        result = np.zeros( [self.umat.shape[0], self.umat.shape[0]], dtype=float )
        squarejumper = 0
        for count in range( len( self.imp_1RDM ) ): # self.imp_size has length 1 if self.TransInv
            localsize = self.imp_size[ count ]
            result[ squarejumper:squarejumper+localsize , squarejumper:squarejumper+localsize ] = self.imp_1RDM[ count ][ :localsize , :localsize ]
            squarejumper += localsize
        return result
        
    def dump_bath_orbs( self, filename, impnumber=0 ):
        from pyscf import tools
        from pyscf.tools import molden
        with open( filename, 'w' ) as thefile:
            molden.header( self.ints.mol, thefile )
            molden.orbital_coeff( self.ints.mol, thefile, np.dot( self.ints.ao2loc, self.dmetOrbs[impnumber] ) )

    def dump_natural_orbitals( self, filename, impnumber=0, fmt='molden', orbital_indices=None ):
        # Natural orbitals of the impurity 1-RDM, back-transformed to AOs for molden/cube.
        if not self.imp_1RDM:
            raise RuntimeError("dump_natural_orbitals: run oneshot()/selfconsistent() first.")
        occ, vecs = np.linalg.eigh( self.imp_1RDM[impnumber] )
        order = np.argsort( occ )[::-1]
        occ   = occ[order]
        mo_ao = np.dot( self.ints.ao2loc, np.dot( self.dmetOrbs[impnumber], vecs[:, order] ) )
        if fmt == 'molden':
            from pyscf.tools import molden
            with open( filename, 'w' ) as thefile:
                molden.header( self.ints.mol, thefile )
                molden.orbital_coeff( self.ints.mol, thefile, mo_ao, occ=occ )
        elif fmt == 'cube':
            from pyscf.tools import cubegen
            if orbital_indices is None:
                orbital_indices = [i for i, n in enumerate(occ) if 1e-2 < n < 2.0 - 1e-2]
            base = filename[:-5] if filename.endswith('.cube') else filename
            for i in orbital_indices:
                cubegen.orbital( self.ints.mol, f"{base}_no{i}_occ{occ[i]:.3f}.cube", mo_ao[:, i] )
        else:
            raise ValueError(f"dump_natural_orbitals: unknown fmt='{fmt}'. Use 'molden' or 'cube'.")
        return occ

    def dump_ntos( self, filename, impnumber=0, initial_state=0, target_state=1,
                   fmt='molden', n_pairs=None, nx=60, ny=60, nz=60 ):
        # NTOs for the initial_state -> target_state transition, back-transformed to AOs.
        import pyscf.fci.direct_spin1 as fci_spin1
        if self.qdnevpt2_results:
            if impnumber >= len( self.qdnevpt2_results ):
                raise IndexError( f"dump_ntos: impnumber={impnumber} out of range." )
            res     = self.qdnevpt2_results[impnumber]
            mc      = res['mc']
            ci      = mc.ci if isinstance( mc.ci, list ) else [mc.ci]
            mo      = mc.mo_coeff
            ncore   = mc.ncore
            ncas    = mc.ncas
            nelecas = mc.nelecas
        elif self.cas_results:
            if impnumber >= len( self.cas_results ):
                raise IndexError( f"dump_ntos: impnumber={impnumber} out of range." )
            res     = self.cas_results[impnumber]
            ci      = res['ci'] if isinstance( res['ci'], list ) else [res['ci']]
            mo      = res['mo_coeff']
            ncore   = res['ncore']
            ncas    = res['ncas']
            nelecas = res['nelecas']
        else:
            raise RuntimeError( "dump_ntos: run oneshot()/selfconsistent() with CASSCF or QD-NEVPT2 first." )
        if initial_state >= len(ci) or target_state >= len(ci):
            raise ValueError( f"dump_ntos: state index out of range (nstates={len(ci)})." )

        tdm = fci_spin1.trans_rdm1( ci[target_state], ci[initial_state], ncas, nelecas )
        U, s, Vh = np.linalg.svd( tdm, full_matrices=False )
        weights  = s ** 2
        omega    = np.sum( weights )
        w_norm   = weights / ( omega + 1e-30 )
        pr       = 1.0 / ( np.sum( w_norm ** 2 ) + 1e-30 )
        entropy  = -np.sum( w_norm * np.log( w_norm + 1e-16 ) )

        print( f"NTOs S{initial_state} -> S{target_state} (impurity {impnumber}):" )
        print( f"  Singular values:         {np.round(s, 6).tolist()}" )
        print( f"  Weights (s^2):           {np.round(weights, 6).tolist()}" )
        print( f"  Sum of weights (Omega):  {omega:.6f}" )
        print( f"  Participation ratio:     {pr:.6f}" )
        print( f"  Entanglement entropy:    {entropy:.6f}" )
        print( f"  Entangled states (Z):    {np.exp(entropy):.6f}" )

        # Active-space MOs back-transformed to real AOs (same chain as dump_natural_orbitals).
        cas_mo_ao  = self.ints.ao2loc @ self.dmetOrbs[impnumber] @ mo[:, ncore:ncore + ncas]
        C_hole     = cas_mo_ao @ U        # hole NTOs (left singular vectors)
        C_particle = cas_mo_ao @ Vh.T     # particle NTOs (right singular vectors)

        # Phase consistency: largest-magnitude AO sets hole sign (PRISM convention).
        for k in range( C_hole.shape[1] ):
            idx = np.argmax( np.abs( C_hole[:, k] ) )
            if C_hole[idx, k] < 0:
                C_hole[:, k]     *= -1
                C_particle[:, k] *= -1

        n_write = C_hole.shape[1] if n_pairs is None else min( n_pairs, C_hole.shape[1] )

        if fmt == 'molden':
            from pyscf.tools import molden
            C_nto   = np.zeros( (C_hole.shape[0], 2 * n_write) )
            C_nto[:, 0::2] = C_hole[:, :n_write]
            C_nto[:, 1::2] = C_particle[:, :n_write]
            occ_nto = np.repeat( weights[:n_write], 2 )
            with open( filename, 'w' ) as thefile:
                molden.header( self.ints.mol, thefile )
                molden.orbital_coeff( self.ints.mol, thefile, C_nto, occ=occ_nto )
        elif fmt == 'cube':
            from pyscf.tools import cubegen
            base = filename[:-5] if filename.endswith( '.cube' ) else filename
            for k in range( n_write ):
                cubegen.orbital( self.ints.mol,
                                 f"{base}_S{initial_state}S{target_state}_nto{k+1}_hole.cube",
                                 C_hole[:, k], nx=nx, ny=ny, nz=nz )
                cubegen.orbital( self.ints.mol,
                                 f"{base}_S{initial_state}S{target_state}_nto{k+1}_particle.cube",
                                 C_particle[:, k], nx=nx, ny=ny, nz=nz )
        else:
            raise ValueError( f"dump_ntos: unknown fmt='{fmt}'. Use 'molden' or 'cube'." )
        return weights, C_hole, C_particle

    def onedm_solution_rhf(self):
        return self.helper.construct1RDM_loc( self.doSCF, self.umat )

    def oneshot( self, mu_imp=0.0, optimize_mu=False ):
        if optimize_mu:
            if self.spin_polarized:
                mu0 = np.array(mu_imp) if hasattr(mu_imp, '__len__') \
                      else np.array([float(mu_imp), float(mu_imp)])
                try:
                    sol = optimize.fsolve(
                        self.numeleccostfunction_spinpol,
                        mu0,
                        full_output=True,
                    )
                    mu_a, mu_b = sol[0]
                    self.mu_imp = mu_a
                    self._chempot_imp_beta = mu_b
                    print(f"  Spin-polarized mu: alpha={mu_a:.8f}, beta={mu_b:.8f}")
                except Exception as exc:
                    print(f"Warning: spin-polarized mu optimization failed ({exc}). "
                          f"Falling back to spin-symmetric mu.")
                    self.mu_imp = float(mu_imp) if not hasattr(mu_imp, '__len__') \
                                  else float(mu_imp[0])
                    self._chempot_imp_beta = self.mu_imp
            else:
                try:
                    self.mu_imp = optimize.newton( self.numeleccostfunction, mu_imp )
                except RuntimeError:
                    print("Warning: Newton solver for mu_imp did not converge. Falling back to initial mu.")
                    self.mu_imp = mu_imp
        else:
            self.mu_imp = mu_imp

        self.doexact( self.mu_imp )
        return self.energy


def make_fragments( mol, myInts, atom_groups ):
    ao_slices = mol.aoslice_by_atom()  # shape (natm, 4): (shl0, shl1, ao0, ao1)
    Norbs = myInts.Norbs
    impurity_clusters = []
    covered = np.zeros(Norbs, dtype=int)

    for group in atom_groups:
        mask = np.zeros(Norbs, dtype=int)
        for atom_idx in group:
            ao_start = ao_slices[atom_idx, 2]
            ao_stop  = ao_slices[atom_idx, 3]
            mask[ao_start:ao_stop] = 1
        impurity_clusters.append(mask)
        covered += mask

    if np.any(covered > 1):
        raise ValueError("make_fragments: overlapping atom groups detected.")

    return impurity_clusters
