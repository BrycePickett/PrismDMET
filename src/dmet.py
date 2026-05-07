"""
Prismdmet core dmet driver.
Built on QC-dmet (Wouters et al., 2015) under GPL-v2.
"""

import local_integrals
import prismdmet_helper
import numpy as np
from scipy import optimize
import time
import os
import concurrent.futures
from solvers import SolverDispatcher
from fragment_builder import FragmentBuilder

# Methods that can be run in parallel worker processes (all inputs are plain numpy arrays)
_PARALLEL_METHODS = frozenset({'ED', 'FCI', 'CC', 'MP2', 'EOM-CC', 'DMRG', 'flag_rhf'})


def _fragment_worker(task):
    """
    Module-level worker for parallel fragment solving via ProcessPoolExecutor.
    Must be a top-level function (not a method) to be picklable.

    All inputs and outputs are plain numpy arrays or Python scalars — no
    PySCF objects are passed across process boundaries.

    Delegates entirely to SolverDispatcher.execute(task), which is the single
    authoritative dispatch point for all solver methods. The returned dict
    from SolverDispatcher is already in the standardised format expected by
    doexact() Phase 3.

    Parameters
    ----------
    task : dict
        Standardised task dict. See solvers/__init__.py for full schema.
        At minimum: method, const, dmet_oei, dmet_fock, dmet_tei, norb, nel,
        nimp, chempot_imp, counter, src_path.

    Returns
    -------
    dict with keys 'counter', 'energy', 'rdm1', plus optional method-specific
    keys ('eom_res', etc.) — as returned by SolverDispatcher.execute().
    """
    # Pin BLAS to 1 thread inside workers to prevent over-subscription
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    # Ensure the src/ directory is on sys.path so solver modules are importable
    # inside the spawned subprocess (SolverDispatcher._ensure_src_path handles this).
    import sys
    src_path = task.get('src_path', '')
    if src_path and src_path not in sys.path:
        sys.path.insert(0, src_path)

    from solvers import SolverDispatcher
    return SolverDispatcher.execute(task)


class DMET:

    def __init__( self, the_ints, impurity_clusters, is_translation_invariant, method='ED',
                  sc_method='LSTSQ', fit_imp_bath=True, use_constrained_opt=False,
                  do_det=False, do_det_NO=False, CC_E_TYPE='LAMBDA',
                  print_u=True, print_rdm=True, eom_nroots=3,
                  eom_type='EE-Singlet', eom_koopmans=False, eom_kwargs=None,
                  ncas=None, nelecas=None, sa_nstates=1, sa_weights=None,
                  casscf_kwargs=None,
                  mf_real=None, qdnevpt2_kwargs=None, nevpt2_kwargs=None,
                  use_symmetry=False, symmetry_map=None,
                  parallel=False, max_workers=None, bath_tol=1e-13,
                  xc='pbe', spin_polarized=False ):

        if ( is_translation_invariant == True ):
            assert( the_ints.TI_OK == True )

        _valid_methods = {'ED', 'FCI', 'DMRG', 'DMRG-CheMPS2', 'CC', 'MP2', 'RHF',
                          'EOM-CC', 'CASSCF', 'QD-NEVPT2', 'NEVPT2',
                          'UHF', 'ROHF', 'RKS', 'UKS', 'ROKS'}
        assert method in _valid_methods, \
            f"DMET: unknown method='{method}'. Valid: {sorted(_valid_methods)}"
        if method in ('QD-NEVPT2', 'NEVPT2'):
            if mf_real is None:
                raise ValueError(
                    f"method='{method}' requires mf_real: a converged RHF object on the "
                    f"REAL physical molecule. Pass it as mf_real=mf to DMET.__init__."
                )
            if ncas is None or nelecas is None:
                raise ValueError(
                    f"method='{method}' requires ncas and nelecas (active space size)."
                )
        if method == 'QD-NEVPT2':
            if sa_nstates < 2:
                raise ValueError(
                    "method='QD-NEVPT2' requires sa_nstates >= 2 for state-averaging."
                )
        assert (( sc_method == 'LSTSQ' ) or ( sc_method == 'BFGS' ) or ( sc_method == 'NONE' ))
        _valid_cc_etypes = {'LAMBDA', 'LAMBDA_AMP', 'LAMBDA_ZERO', 'CASCI', 'CCSD(T)', 'CCSD(T)_RDM', 'EOM-CCSD'}
        assert CC_E_TYPE in _valid_cc_etypes, f"DMET: unknown CC_E_TYPE='{CC_E_TYPE}'. Valid: {_valid_cc_etypes}"

        self.ints       = the_ints
        self.norb       = self.ints.Norbs
        self.impClust   = impurity_clusters
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
        self.cas_results   = []   # populated by doexact() when method='CASSCF'
        self.mf_real       = mf_real
        self.qdnevpt2_kwargs   = qdnevpt2_kwargs or {}
        self.qdnevpt2_results  = []  # populated by doexact() when method='QD-NEVPT2'
        self.nevpt2_kwargs     = nevpt2_kwargs or {}
        self.nevpt2_results    = []  # populated by doexact() when method='NEVPT2'
        self.dft_results       = []  # populated by doexact() when method is DFT
        self.BATH_ORBS  = None
        self.fit_imp_bath = fit_imp_bath
        self.do_det      = do_det
        self.do_det_NO   = do_det_NO
        self.NOrotation = None
        self.altcostfunc = use_constrained_opt
        self.oei_s      = None  # spin-dependent 1e potential for open-shell envs
        self.bath_tol   = bath_tol
        self.xc         = xc           # XC functional for DFT solvers
        self.spin_polarized = spin_polarized  # enable independent alpha/beta mu optimization

        # --- Efficiency: symmetry mapping ---
        self.use_symmetry = use_symmetry
        self.symmetry_map = symmetry_map  # user-provided {child_idx: parent_idx} or None
        if self.use_symmetry and self.symmetry_map is None and not is_translation_invariant:
            self.symmetry_map = self._auto_detect_symmetry()

        # --- Efficiency: parallel fragment solving ---
        self.parallel = parallel
        if max_workers is not None:
            self.max_workers = max_workers
        else:
            # Auto-detect from environment
            _env_threads = os.environ.get('PRISMdmet_WORKERS',
                           os.environ.get('SLURM_CPUS_PER_TASK',
                           os.environ.get('OMP_NUM_THREADS', '1')))
            self.max_workers = max(1, int(_env_threads))

        # Fragment caches for warm-restarting CASSCF across SC iterations.
        # Each entry is None (first iteration) or a dict with 'mo_coeff' and 'ci'.
        maxiter_frags = 1 if is_translation_invariant else len(impurity_clusters)
        self.frag_caches = [None] * maxiter_frags

        self.minFunc    = None
        if self.altcostfunc:
            self.minFunc = 'FOCK_INIT'  # 'oei'
            assert (self.fit_imp_bath == False)
            assert (self.do_det == False)
            assert (self.sc_method == 'BFGS' or self.sc_method == 'NONE')

        if (( self.method == 'CC' ) and ( self.CC_E_TYPE == 'CASCI' )):
            assert( len( self.impClust ) == 1 )
        if (( self.method == 'CC' ) and ( self.CC_E_TYPE == 'EOM-CCSD' )):
            assert eom_nroots >= 1, "eom_nroots must be >= 1 when using CC_E_TYPE='EOM-CCSD'"
        if self.method == 'EOM-CC':
            from solvers.eomcc import _VALID_EOM_TYPES as _EOM_SET
            if eom_type not in _EOM_SET:
                raise ValueError(
                    f"DMET: unknown eom_type='{eom_type}'. Valid: {sorted(_EOM_SET)}"
                )

        if ( self.do_det == True ):
            # DET only fits impurity diagonal; see Bulik, PRB 89, 035140 (2014)
            self.fit_imp_bath = False
            if ( self.do_det_NO == True ):
                self.NOvecs = None
                self.NOdiag = None

        self.print_u   = print_u
        self.print_rdm = print_rdm

        allOne = self.testclusters()
        if ( allOne == False ):
            # Incomplete tiling: impurity orbitals must be the first in the Hamiltonian.
            assert( self.TransInv == False )

        self.energy   = 0.0
        self.imp_1RDM = []
        self.dmetOrbs = []
        self.imp_size = self.make_imp_size()
        self.mu_imp   = 0.0
        self.mask     = self.make_mask()
        self.helper   = prismdmet_helper.PrismDMETHelper( self.ints, self.makelist_H1(), self.altcostfunc, self.minFunc )

        self.time_ed  = 0.0
        self.time_cf  = 0.0
        self.time_func= 0.0
        self.time_grad= 0.0

        # Auto-detect open-shell reference from localintegrals
        if hasattr(self.ints, 'loc_spin_oei'):
            self.oei_s = self.ints.loc_spin_oei()   # None for RHF, ndarray for ROHF/UHF

        np.set_printoptions(precision=3, linewidth=160)
        
    def testclusters( self ):
    
        quicktest = np.zeros([ self.norb ], dtype=int)
        for item in self.impClust:
            quicktest += np.abs(item)
        assert( np.all( quicktest >= 0 ) )
        assert( np.all( quicktest <= 1 ) )
        allOne = np.all( quicktest == 1 )
        return allOne

    def _auto_detect_symmetry( self ):
        """
        Auto-detect equivalent fragments by comparing their impurity orbital
        sizes. Two fragments are equivalent if they have the same number of
        impurity orbitals. The first fragment of each unique size is the
        'parent'; all subsequent identical-size fragments map to it.

        Returns
        -------
        symmetry_map : dict
            Mapping {child_fragment_idx : parent_fragment_idx}.
            Empty dict if all fragments are unique.
        """
        symmetry_map = {}
        seen = {}  # fingerprint -> first fragment index
        for idx, cluster in enumerate(self.impClust):
            fingerprint = int(np.sum(np.abs(cluster)))
            if fingerprint in seen:
                symmetry_map[idx] = seen[fingerprint]
            else:
                seen[fingerprint] = idx
        if symmetry_map:
            n_unique = len(self.impClust) - len(symmetry_map)
            print(f"Prismdmet :: symmetry : Auto-detected {n_unique} unique fragment(s) "
                  f"out of {len(self.impClust)} total. "
                  f"Skipping {len(symmetry_map)} equivalent fragment solve(s).")
        return symmetry_map
            
    def make_imp_size( self ):
    
        thearray = []
        maxiter = len( self.impClust )
        if ( self.TransInv == True ):
            maxiter = 1
        for counter in range( maxiter ):
            impurity_orbs = np.abs(self.impClust[ counter ])
            num_imp_orbs = np.sum( impurity_orbs )
            thearray.append( num_imp_orbs )
        thearray = np.array( thearray )
        return thearray

    def makelist_H1( self ):
    
        theH1 = []
        if ( self.do_det == True ): # Do density embedding theory
            if ( self.TransInv == True ): # Translational invariance assumed
                localsize = self.imp_size[ 0 ]
                for row in range( localsize ):
                    H1 = np.zeros( [ self.norb, self.norb ], dtype=int )
                    for jumper in range( self.norb // localsize ):
                        jumpsquare = localsize * jumper
                        H1[ jumpsquare + row, jumpsquare + row ] = 1
                    theH1.append( H1 )
            else: # NO translational invariance assumed
                jumpsquare = 0
                for localsize in self.imp_size:
                    for row in range( localsize ):
                        H1 = np.zeros( [ self.norb, self.norb ], dtype=int )
                        H1[ jumpsquare + row, jumpsquare + row ] = 1
                        theH1.append( H1 )
                    jumpsquare += localsize
        else: # Do density MATRIX embedding theory
            if ( self.TransInv == True ): # Translational invariance assumed
                localsize = self.imp_size[ 0 ]
                for row in range( localsize ):
                    for col in range( row, localsize ):
                        H1 = np.zeros( [ self.norb, self.norb ], dtype=int )
                        for jumper in range( self.norb // localsize ):
                            jumpsquare = localsize * jumper
                            H1[ jumpsquare + row, jumpsquare + col ] = 1
                            H1[ jumpsquare + col, jumpsquare + row ] = 1
                        theH1.append( H1 )
            else: # NO translational invariance assumed
                jumpsquare = 0
                for localsize in self.imp_size:
                    for row in range( localsize ):
                        for col in range( row, localsize ):
                            H1 = np.zeros( [ self.norb, self.norb ], dtype=int )
                            H1[ jumpsquare + row, jumpsquare + col ] = 1
                            H1[ jumpsquare + col, jumpsquare + row ] = 1
                            theH1.append( H1 )
                    jumpsquare += localsize
        return theH1
        
    def make_mask( self ):
    
        themask = np.zeros( [ self.norb, self.norb ], dtype=bool )
        if ( self.do_det == True ): # Do density embedding theory
            jump = 0
            for localsize in self.imp_size: # self.imp_size has length 1 if self.TransInv
                for row in range( localsize ):
                    themask[ jump + row, jump + row ] = True
                jump += localsize
        else: # Do density MATRIX embedding theory
            jump = 0
            for localsize in self.imp_size: # self.imp_size has length 1 if self.TransInv
                for row in range( localsize ):
                    for col in range( row, localsize ):
                        themask[ jump + row, jump + col ] = True
                jump += localsize
        return themask
        
    def doexact( self, chempot_imp=0.0 ):
    
        one_rdm = self.helper.construct1RDM_loc( self.doSCF, self.umat )
        self.energy   = 0.0
        self.imp_1RDM = []
        self.dmetOrbs = []
        self.frag_energies = []  # per-fragment energies for symmetry reuse
        if ( self.do_det == True ) and ( self.do_det_NO == True ):
            self.NOvecs = []
            self.NOdiag = []
        
        maxiter = len( self.impClust )
        if ( self.TransInv == True ):
            maxiter = 1
            
        remainingOrbs = np.ones( [ len( self.impClust[ 0 ] ) ], dtype=float )
               # ---------------------------------------------------------------
        # Phase 1: Build all fragment tasks (bath + integrals).
        # Always sequential — all fragments read from the same one_rdm.
        # ---------------------------------------------------------------
        _frag_tasks   = []   # picklable task dicts for parallel workers
        _frag_meta    = []   # per-fragment metadata needed after solve
        _sym_counters = set()  # counters handled by symmetry skip

        _src_path = os.path.dirname(os.path.abspath(__file__))

        # Instantiate fragment_builder once per doexact() call.
        # It holds lightweight references to ints and helper; it does NOT copy
        # the large integral tensors.
        _builder = FragmentBuilder(
            ints     = self.ints,
            helper   = self.helper,
            impClust = self.impClust,
            method   = self.method,
            BATH_ORBS= self.BATH_ORBS,
            NI_hack  = self.NI_hack,
            umat     = self.umat,
            bath_tol = self.bath_tol,
        )

        for counter in range( maxiter ):

            # --- Symmetry skip (handled immediately, no solver needed) ---
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

            # --- Build embedding Hamiltonian via fragment_builder -----------
            frag = _builder.build(counter, one_rdm, chempot_imp)

            # Unpack frequently-used local names (improves readability below)
            flag_rhf     = frag['flag_rhf']
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

            # ---------------------------------------------------------------
            # Build the full task dict for SolverDispatcher / parallel worker.
            #
            # All keys are plain numpy arrays or Python scalars so the task
            # dict is always picklable regardless of method.  PySCF's mf_real
            # object is represented as serialized arrays + mol JSON dump; the
            # solver's execute() wrapper reconstructs a dummy mf inside the
            # worker process.  See solvers/nevpt2._reconstruct_mf_from_task.
            # ---------------------------------------------------------------

            # For NEVPT2 methods: build full-molecule MO guess anchored to
            # the dmet embedding orbitals (must happen in the main process
            # where mf_real is available).
            _mo_guess = None
            if _method_key in ('QD-NEVPT2', 'NEVPT2') and self.mf_real is not None:
                _mo_guess, _, _ = self._build_full_mo_guess(
                    loc_2_dmet, norb_in_imp, core_1rdm_dmet)

            # Serialize mf_real if available. The solver modules' execute()
            # wrappers detect 'mol_dumps' in the task and reconstruct a
            # minimal RHF object without re-running any SCF iterations.
            _mf_serial = {}
            if self.mf_real is not None:
                _mf_serial = {
                    'mol_dumps'    : self.mf_real.mol.dumps(),
                    'mf_mo_coeff'  : self.mf_real.mo_coeff,
                    'mf_mo_energy' : self.mf_real.mo_energy,
                    'mf_mo_occ'    : self.mf_real.mo_occ,
                    'mf_e_tot'     : self.mf_real.e_tot,
                }

            # DFT solvers need the real molecule (for XC grid) and the
            # AO-to-localized-orbital transformation to back-transform the
            # embedding density to AO space for correct XC evaluation.
            _dft_mol_info = {}
            if _method_key in ('RKS', 'UKS', 'ROKS'):
                _dft_mol_info = {
                    'dft_mol_dumps' : self.ints.mol.dumps(),
                    'ao2loc'        : self.ints.ao2loc,
                    'loc_2_dmet'    : loc_2_dmet,
                }

            task = {
                # --- Core identity ------------------------------------------
                'counter'       : counter,
                'method'        : _method_key,
                'src_path'      : _src_path,
                # --- Embedding integrals (always picklable) -----------------
                'const'         : 0.0,
                'dmet_oei'       : dmet_oei,
                'dmet_fock'      : dmet_fock,
                'dmet_tei'       : dmet_tei,
                'norb'          : norb_in_imp,
                'nel'           : nelec_in_imp,
                'nimp'          : num_imp_orbs,
                'chempot_imp'   : chempot_imp,
                'dm_guess_rhf'    : dm_guess_rhf,
                # --- CC / EOM-CC options ------------------------------------
                'CC_E_TYPE'     : self.CC_E_TYPE,
                'eom_nroots'    : self.eom_nroots,
                'eom_type'      : self.eom_type,
                'eom_koopmans'  : self.eom_koopmans,
                'eom_kwargs'    : self.eom_kwargs,
                # --- CASSCF / NEVPT2 active-space options -------------------
                'ncas'          : self.ncas,
                'nelecas'       : self.nelecas,
                'sa_nstates'    : self.sa_nstates,
                'sa_weights'    : self.sa_weights,
                'casscf_kwargs' : self.casscf_kwargs,
                'mo_guess'      : _mo_guess,
                'nevpt2_kwargs' : self.nevpt2_kwargs,
                'qdnevpt2_kwargs': self.qdnevpt2_kwargs,
                # --- DFT / open-shell options --------------------------------
                'xc'            : self.xc,
                'spin'          : nelec_in_imp % 2,
                'spin_polarized': self.spin_polarized,
                # --- DFT mol and localization info --------------------------
                **_dft_mol_info,
                # --- Serialized physical molecule (NEVPT2 / QD-NEVPT2) -----
                # Keys present only when self.mf_real is not None.
                # Solver wrappers detect 'mol_dumps' to choose the
                # parallel-safe reconstruction path.
                **_mf_serial,
            }

            # ---------------------------------------------------------------
            # Route the task: parallel or sequential.
            #
            # All methods in PARALLEL_ELIGIBLE can now run in a worker
            # process because the task dict is fully picklable.
            # Methods that require in-process state (e.g. do_det_NO's
            # NOrotation tracking) must remain sequential.
            # ---------------------------------------------------------------
            from solvers import PARALLEL_ELIGIBLE
            _is_parallel_eligible = (
                self.parallel
                and _method_key in PARALLEL_ELIGIBLE
                and not self.do_det_NO  # NO rotation requires in-process state
            )

            if _is_parallel_eligible:
                _frag_tasks.append(task)
            else:
                # Sequential path: run immediately in the main process.
                # _run_fragment_sequential builds its own task dict (with
                # the live mf_real for CASSCF cache handling), so we pass
                # the CASSCF warm-restart keys explicitly rather than the
                # full serialized task dict.  This preserves frag_caches
                # side effects that are only needed in-process.
                _frag_meta[-1]['sequential_result'] = self._run_fragment_sequential(
                    counter, _method_key, flag_rhf, dmet_oei, dmet_fock, dmet_tei,
                    norb_in_imp, nelec_in_imp, num_imp_orbs, chempot_imp,
                    dm_guess_rhf, loc_2_dmet, mo_guess=_mo_guess)

        # ---------------------------------------------------------------
        # Phase 2: Run parallel tasks via ProcessPoolExecutor.
        # ---------------------------------------------------------------
        _parallel_results = {}   # counter -> result dict
        if _frag_tasks:
            _nw = min(self.max_workers, len(_frag_tasks))
            print(f"Prismdmet :: parallel : Submitting {len(_frag_tasks)} fragment(s) "
                  f"to {_nw} worker process(es).")
            ctx = concurrent.futures.get_context('spawn') if hasattr(concurrent.futures, 'get_context') else None
            _Executor = concurrent.futures.ProcessPoolExecutor
            with _Executor(max_workers=_nw) as pool:
                futures = {pool.submit(_fragment_worker, t): t['counter'] for t in _frag_tasks}
                for fut in concurrent.futures.as_completed(futures):
                    res = fut.result()
                    _parallel_results[res['counter']] = res

        # ---------------------------------------------------------------
        # Phase 3: Collect results in order and update state.
        # ---------------------------------------------------------------
        for meta in _frag_meta:
            counter = meta['counter']
            impurity_orbs = meta['impurity_orbs']

            if meta.get('sym_parent') is not None:
                # Symmetry-copied fragment — parent must already be in frag_energies/imp_1RDM
                parent = meta['sym_parent']
                print(f"Prismdmet :: symmetry : Fragment {counter} <- fragment {parent} (copied).")
                parent_energy = self.frag_energies[parent]
                parent_rdm    = self.imp_1RDM[parent]
                self.energy += parent_energy
                self.frag_energies.append(parent_energy)
                self.imp_1RDM.append(parent_rdm.copy())
                remainingOrbs -= impurity_orbs
                continue

            num_imp_orbs = meta['num_imp_orbs']

            # Retrieve result from whichever path handled this fragment.
            if counter in _parallel_results:
                res = _parallel_results[counter]
            else:
                res = meta['sequential_result']

            IMP_energy = res['energy']
            IMP_1RDM   = res['rdm1']

            # Log OOM fallback if one occurred.
            if res.get('fallback_from'):
                print(
                    f"Prismdmet :: WARNING : Fragment {counter} — "
                    f"'{res['fallback_from']}' failed (OOM); "
                    f"result computed with '{res.get('method', meta['method_key'])}'. "
                    f"Energy may be less accurate."
                )

            # Unpack method-specific result artefacts.
            if 'eom_res' in res:
                self.eom_results.append(res['eom_res'])
            if 'cas_res' in res:
                # CASSCF warm-restart cache: store mo_coeff + ci for next
                # dmet iteration.  This is still meaningful even if parallel
                # because the workers return the numpy arrays, not the live mc.
                self.frag_caches[counter] = {
                    'mo_coeff': res['cas_res']['mo_coeff'],
                    'ci'      : res['cas_res']['ci'],
                }
                self.cas_results.append(res['cas_res'])
            if 'qdnevpt2_res' in res:
                self.qdnevpt2_results.append(res['qdnevpt2_res'])
            if 'nevpt2_res' in res:
                self.nevpt2_results.append(res['nevpt2_res'])
            if 'dft_res' in res:
                self.dft_results.append(res['dft_res'])

            self.energy += IMP_energy
            self.frag_energies.append(IMP_energy)
            self.imp_1RDM.append( IMP_1RDM )
            if ( self.do_det == True ) and ( self.do_det_NO == True ):
                RDMeigenvals, RDMeigenvecs = np.linalg.eigh( IMP_1RDM[ :num_imp_orbs, :num_imp_orbs ] )
                self.NOvecs.append( RDMeigenvecs )
                self.NOdiag.append( RDMeigenvals )

            remainingOrbs -= impurity_orbs
        
        if ( self.do_det == True ) and ( self.do_det_NO == True ):
            self.NOrotation = self.constructNOrotation()
        
        Nelectrons = 0.0
        for counter in range( maxiter ):
            Nelectrons += np.trace( self.imp_1RDM[counter][ :self.imp_size[counter], :self.imp_size[counter] ] )
        if ( self.TransInv == True ):
            Nelectrons = Nelectrons * len( self.impClust )
            self.energy = self.energy * len( self.impClust )
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

                from pyscf import scf
                from types import MethodType
                mol_ = self.ints.mol
                mf_  = scf.RHF(mol_)

                impOrbs = remainingOrbs==1
                xorb = np.dot(mf_.get_ovlp(), self.ints.ao2loc)
                hc  = -chempot_imp * np.dot(xorb[:,impOrbs], xorb[:,impOrbs].T)
                dm0 = np.dot(self.ints.ao2loc, np.dot(one_rdm, self.ints.ao2loc.T))

                def mf_hcore (self, mol=None):
                    if mol is None: mol = self.mol
                    return scf.hf.get_hcore(mol) + hc
                mf_.get_hcore = MethodType(mf_hcore, mf_)
                mf_.scf(dm0)
                assert (mf_.converged)

                rdm1 = mf_.make_rdm1()
                jk   = mf_.get_veff(dm=rdm1)

                xorb = np.dot(mf_.get_ovlp(), self.ints.ao2loc)
                rdm1 = np.dot(xorb.T, np.dot(rdm1, xorb))
                oei  = np.dot(self.ints.ao2loc.T, np.dot(mf_.get_hcore()-hc, self.ints.ao2loc))
                jk   = np.dot(self.ints.ao2loc.T, np.dot(jk, self.ints.ao2loc))

                ImpEnergy = \
                   + 0.50 * np.einsum('ji,ij->', rdm1[:,impOrbs], oei[impOrbs,:]) \
                   + 0.50 * np.einsum('ji,ij->', rdm1[impOrbs,:], oei[:,impOrbs]) \
                   + 0.25 * np.einsum('ji,ij->', rdm1[:,impOrbs], jk[impOrbs,:]) \
                   + 0.25 * np.einsum('ji,ij->', rdm1[impOrbs,:], jk[:,impOrbs])
                self.energy += ImpEnergy
                Nelectrons += np.trace(rdm1[np.ix_(impOrbs,impOrbs)])

            remainingOrbs[ remainingOrbs==1 ] -= 1
        assert( np.all( remainingOrbs == 0 ) )
            
        self.energy += self.ints.const()
        return Nelectrons

    def _build_full_mo_guess(self, loc_2_dmet, norb_in_imp, core_1rdm_dmet):
        """
        Construct a full-molecule MO coefficient matrix (nAO x nMO) with the
        dmet embedding orbitals placed exactly in the CASSCF active window.

        Layout (columns left to right):
            [frozen_core | env_core | dmet_active | env_virt | frozen_virt]

        Parameters
        ----------
        loc_2_dmet : ndarray (Norbs_active, Norbs_active)
            The unitary from localintegrals LMO basis → dmet embedding basis.
            Columns 0..norb_in_imp-1  : embedding (imp + bath)
            Columns norb_in_imp..end  : environment (core + virtual)
        norb_in_imp : int
            Number of embedding orbitals (impurity + bath).
        core_1rdm_dmet : ndarray (Norbs_active - norb_in_imp,)
            Diagonal of the environment 1-RDM in the dmet basis.
            Entries ≈ 2 are environment core; entries ≈ 0 are environment virtual.

        Returns
        -------
        mo_guess : ndarray (nAO, nMO_real)
            Column-ordered MO matrix suitable for PySCF mc.kernel(mo_guess).
        ncas_check : int
            Number of active orbitals (==norb_in_imp, for a sanity check).
        ncore_check : int
            Number of doubly-occupied (core) orbitals before the active window.
        """
        mf  = self.mf_real
        ao2loc = self.ints.ao2loc      # shape: (nAO, Norbs_active)
        active_mask = self.ints.active # 1 for LMO-active orbs, 0 for frozen

        nAO  = ao2loc.shape[0]
        nMO  = mf.mo_coeff.shape[1]

        # ── Frozen orbitals (outside localintegrals active space) ──────────────
        # active_mask is over canonical MO indices (same as nAO for full-valence
        # calculations, or a subset for frozen-core). Frozen MO indices are those
        # where active_mask == 0.
        frozen_idx  = np.where(active_mask == 0)[0]
        active_idx  = np.where(active_mask == 1)[0]
        frozen_occ  = frozen_idx[mf.mo_occ[frozen_idx] > 0]
        frozen_virt = frozen_idx[mf.mo_occ[frozen_idx] == 0]
        mo_frozen_core = mf.mo_coeff[:, frozen_occ]   # (nAO, N_frozen_core)
        mo_frozen_virt = mf.mo_coeff[:, frozen_virt]  # (nAO, N_frozen_virt)

        # ── Environment orbitals (inside localintegrals active space, outside dmet) ──
        # loc_2_dmet[:, norb_in_imp:] are the environment LMOs. Their occupations
        # come from core_1rdm_dmet[norb_in_imp:]: ~2 → core, ~0 → virtual.
        env_lmos = loc_2_dmet[:, norb_in_imp:]              # (Norbs_active, Nenv)
        env_occ_diag  = core_1rdm_dmet[norb_in_imp:]      # environment occupations only
        env_occ_mask  = env_occ_diag > 1.0               # approximately 2
        env_virt_mask = env_occ_diag < 1.0               # approximately 0

        # Transform environment LMOs from LMO basis back to AO basis
        mo_env_core = ao2loc @ env_lmos[:, env_occ_mask]   # (nAO, Nenv_core)
        mo_env_virt = ao2loc @ env_lmos[:, env_virt_mask]  # (nAO, Nenv_virt)

        # ── Active (dmet embedding) orbitals ──────────────────────────────────
        # loc_2_dmet[:, :norb_in_imp] are the imp+bath LMOs.
        dmet_lmos    = loc_2_dmet[:, :norb_in_imp]       # (Norbs_active, norb_in_imp)
        mo_active    = ao2loc @ dmet_lmos              # (nAO, norb_in_imp)

        # ── Stack into [frozen_core | env_core | active | env_virt | frozen_virt] ──
        mo_guess = np.hstack([
            mo_frozen_core,
            mo_env_core,
            mo_active,
            mo_env_virt,
            mo_frozen_virt,
        ])

        ncore_check = mo_frozen_core.shape[1] + mo_env_core.shape[1]
        ncas_check  = norb_in_imp

        if mo_guess.shape[1] != nMO:
            raise RuntimeError(
                f"_build_full_mo_guess: column count mismatch. "
                f"Built {mo_guess.shape[1]} MOs but mf_real has {nMO}. "
                f"Frozen-core={mo_frozen_core.shape[1]}, env_core={mo_env_core.shape[1]}, "
                f"active={ncas_check}, env_virt={mo_env_virt.shape[1]}, "
                f"frozen_virt={mo_frozen_virt.shape[1]}"
            )

        print(f"Prismdmet :: mo_guess : Built full MO guess for NEVPT2 solver.")
        print(f"  Layout: {mo_frozen_core.shape[1]} frozen_core | {mo_env_core.shape[1]} env_core "
              f"| {ncas_check} active | {mo_env_virt.shape[1]} env_virt "
              f"| {mo_frozen_virt.shape[1]} frozen_virt")
        return mo_guess, ncas_check, ncore_check

    def _run_fragment_sequential(self, counter, method_key, flag_rhf,
                                   dmet_oei, dmet_fock, dmet_tei,
                                   norb_in_imp, nelec_in_imp, num_imp_orbs,
                                   chempot_imp, dm_guess_rhf, loc_2_dmet,
                                   mo_guess=None):
        """
        Run a single fragment solver sequentially and return a standardised
        result dict via SolverDispatcher.execute(task).

        All solver-specific argument marshalling is done here by building
        the task dict, then delegating to SolverDispatcher. This means the
        sequential path and the parallel worker path share exactly the same
        dispatch logic — there is no duplication.

        Special handling for CASSCF warm-restart (MO/CI guess injection
        from frag_caches) is performed before the task dict is built, since
        this requires access to self.frag_caches and project_amo_manually.

        Returns
        -------
        dict with keys: 'energy', 'rdm1', and optionally 'cas_res',
        'eom_res', 'qdnevpt2_res', 'nevpt2_res' — as returned by
        SolverDispatcher.execute().
        """
        # ------------------------------------------------------------------
        # CASSCF-specific pre-processing: warm-restart cache injection
        # Must happen before task dict is built because it requires
        # self.frag_caches, project_amo_manually, and the local dmet_fock.
        # ------------------------------------------------------------------
        _mo_guess_cas, _ci_guess_cas = mo_guess, None
        if method_key == 'CASSCF' and self.frag_caches[counter] is not None:
            cached = self.frag_caches[counter]
            old_mo    = cached['mo_coeff']
            _ci_guess_cas = cached.get('ci', None)
            _ncas  = self.ncas    if self.ncas    is not None else norb_in_imp
            _ncore = (nelec_in_imp - (self.nelecas if self.nelecas is not None
                                      else nelec_in_imp)) // 2
            if old_mo.shape == (norb_in_imp, norb_in_imp):
                from solvers.qcsolver_utils import project_amo_manually
                _mo_guess_cas, fidelity = project_amo_manually(
                    old_mo, _ncas, _ncore, dmet_fock, norb_in_imp)
                if np.min(fidelity) < 0.5:
                    print("DMET::CASSCF : Low projection fidelity, discarding CI guess.")
                    _ci_guess_cas = None
            else:
                print("DMET::CASSCF : MO shape mismatch, starting fresh.")
                _mo_guess_cas = None

        # Spin oei for open-shell environments (CASSCF only)
        _dmet_oei_s = None
        if method_key == 'CASSCF':
            _dmet_oei_s = (self.ints.dmet_oei_s(loc_2_dmet, norb_in_imp)
                           if hasattr(self.ints, 'dmet_oei_s') else self.oei_s)

        # ------------------------------------------------------------------
        # Build the standardised task dict understood by SolverDispatcher.
        # All solver-specific optional keys use .get() with safe defaults
        # so that SolverDispatcher can remain decoupled from dmet.py internals.
        # ------------------------------------------------------------------
        task = {
            # Core identity
            'counter'       : counter,
            'method'        : method_key if not flag_rhf else 'flag_rhf',
            # Embedding integrals (numpy arrays only — picklable)
            'const'         : 0.0,
            'dmet_oei'       : dmet_oei,
            'dmet_fock'      : dmet_fock,
            'dmet_tei'       : dmet_tei,
            'norb'          : norb_in_imp,
            'nel'           : nelec_in_imp,
            'nimp'          : num_imp_orbs,
            'chempot_imp'   : chempot_imp,
            'dm_guess_rhf'    : dm_guess_rhf,
            # CC / EOM-CC options
            'CC_E_TYPE'     : self.CC_E_TYPE,
            'eom_nroots'    : self.eom_nroots,
            'eom_type'      : self.eom_type,
            'eom_koopmans'  : self.eom_koopmans,
            'eom_kwargs'    : self.eom_kwargs,
            # CASSCF options (including warm-restart artefacts computed above)
            'ncas'          : self.ncas,
            'nelecas'       : self.nelecas,
            'sa_nstates'    : self.sa_nstates,
            'sa_weights'    : self.sa_weights,
            'casscf_kwargs' : self.casscf_kwargs,
            'mo_guess'      : _mo_guess_cas,
            'ci_guess'      : _ci_guess_cas,
            'oei_s'         : _dmet_oei_s,
            # DFT / open-shell options
            'xc'            : self.xc,
            'spin'          : nelec_in_imp % 2,
            'spin_polarized': self.spin_polarized,
            # DFT mol and localization info (sequential path has live objects)
            'dft_mol_dumps' : self.ints.mol.dumps() if method_key in ('RKS', 'UKS', 'ROKS') else None,
            'ao2loc'        : self.ints.ao2loc       if method_key in ('RKS', 'UKS', 'ROKS') else None,
            'loc_2_dmet'    : loc_2_dmet             if method_key in ('RKS', 'UKS', 'ROKS') else None,
            # NEVPT2 / QD-NEVPT2: live PySCF objects (not picklable; sequential only)
            'mf_real'       : self.mf_real,
            'nevpt2_kwargs' : self.nevpt2_kwargs,
            'qdnevpt2_kwargs': self.qdnevpt2_kwargs,
        }

        # ------------------------------------------------------------------
        # Delegate to SolverDispatcher — single authoritative dispatch.
        # ------------------------------------------------------------------
        result = SolverDispatcher.execute(task)

        # ------------------------------------------------------------------
        # Post-process: extract CASSCF warm-restart artefacts and store them
        # back into frag_caches. SolverDispatcher returns 'cas_res' when
        # method='CASSCF'; we unpack it here because frag_caches is owned
        # by dmet, not by the solver.
        # ------------------------------------------------------------------
        if 'cas_res' in result:
            cas_res = result['cas_res']
            self.frag_caches[counter] = {
                'mo_coeff': cas_res['mo_coeff'],
                'ci'      : cas_res['ci'],
            }

        return result

    def constructNOrotation( self ):
    
        myNOrotation = np.zeros( [ self.norb, self.norb ], dtype=float )
        jumpsquare = 0
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            myNOrotation[ jumpsquare : jumpsquare + self.imp_size[ count ], jumpsquare : jumpsquare + self.imp_size[ count ] ] = self.NOvecs[ count ]
            jumpsquare += self.imp_size[ count ]
        for count in range( jumpsquare, self.norb ):
            myNOrotation[ count, count ] = 1.0
        if ( self.TransInv == True ):
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
        # e_cstr = np.sum( newumatflat * errors )    # not correct, but gives correct verify_gradient results
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
        
        thesize = 0
        for count in range(len(self.imp_size)):
            if ( self.do_det == True ): # Do density embedding theory: fit only impurity
                thesize += self.imp_size[ count ]
                assert ( self.fit_imp_bath == False )
            else: # Do density MATRIX embedding theory
                if ( self.fit_imp_bath == True ):
                    thesize += self.dmetOrbs[count].shape[1] * self.dmetOrbs[count].shape[1]
                else:
                    thesize += self.imp_size[ count ] * self.imp_size[ count ]
        errors = np.zeros( [ thesize ], dtype=float )
        
        jump = 0
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            if ( self.fit_imp_bath == True ):
                mf_1RDM = np.dot( np.dot( self.dmetOrbs[ count ].T, one_rdm_loc ), self.dmetOrbs[ count ] )
                ed_1RDM = self.imp_1RDM[count]
            else:
                mf_1RDM = (one_rdm_loc[:,np.flatnonzero(self.impClust[count])])[np.flatnonzero(self.impClust[count]),:]
                ed_1RDM = self.imp_1RDM[count][:self.imp_size[count],:self.imp_size[count]]
            if ( self.do_det == True ): # Do density embedding theory
                if ( self.do_det_NO == True ): # Work in the NO basis
                    theerror = np.diag( np.dot( np.dot( self.NOvecs[ count ].T, mf_1RDM ), self.NOvecs[ count ] ) ) - self.NOdiag[ count ]
                else: # Work in the lattice basis
                    theerror = np.diag( mf_1RDM - ed_1RDM )
                errors[ jump : jump + len( theerror ) ] = theerror
                jump += len( theerror )
            else: # Do density MATRIX embedding theory
                theerror = mf_1RDM - ed_1RDM
                squaresize = theerror.shape[0] * theerror.shape[1]
                errors[ jump : jump + squaresize ] = np.reshape( theerror, squaresize, order='F' )
                jump += squaresize
        assert ( jump == thesize )
        
        stop_func = time.time()
        self.time_func += ( stop_func - start_func )
        
        return errors
        
    def rdm_differences_bis( self, newumatflat ):
    
        start_func = time.time()
    
        newumatsquare_loc = self.flat2square( newumatflat )
        one_rdm_loc = self.helper.construct1RDM_loc( self.doSCF, newumatsquare_loc )

        thesize = 0
        jump = 0
        for count in range(len(self.imp_size)):
            # thesize += self.imp_size[ count ] * self.imp_size[ count ]
            mask_t = self.mask[ np.ix_(list(range(jump,jump+self.imp_size[count])),list(range(jump,jump+self.imp_size[count]))) ]
            thesize += np.count_nonzero( mask_t )
            jump += self.imp_size[count]
        errors = np.zeros( [ thesize ], dtype=float )
        
        jump = 0
        jumpc = 0
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            mf_1RDM = (one_rdm_loc[:,np.flatnonzero(self.impClust[count])])[np.flatnonzero(self.impClust[count]),:]
            ed_1RDM = self.imp_1RDM[count][:self.imp_size[count],:self.imp_size[count]]
            theerror = mf_1RDM - ed_1RDM
            # squaresize = theerror.shape[0] * theerror.shape[1]
            # errors[ jump : jump + squaresize ] = np.reshape( theerror, squaresize, order='F' )
            mask_t = self.mask[ np.ix_(list(range(jumpc,jumpc+self.imp_size[count])),list(range(jumpc,jumpc+self.imp_size[count]))) ]
            squaresize = np.count_nonzero( mask_t )
            errors[ jump : jump + squaresize ] = np.reshape( theerror[mask_t], squaresize, order='F' )
            jump  += squaresize
            jumpc += self.imp_size[count]
        assert ( jump == thesize )
        
        stop_func = time.time()
        self.time_func += ( stop_func - start_func )
        
        return errors

    def rdm_differences_derivative( self, newumatflat ):
        
        start_grad = time.time()
        
        newumatsquare_loc = self.flat2square( newumatflat )
        RDMderivs_rot = self.helper.construct1RDM_response( self.doSCF, newumatsquare_loc, self.NOrotation )
        
        thesize = 0
        for count in range(len(self.imp_size)):
            if ( self.do_det == True ): # Do density embedding theory: fit only impurity
                thesize += self.imp_size[ count ]
                assert ( self.fit_imp_bath == False )
            else: # Do density MATRIX embedding theory
                if ( self.fit_imp_bath == True ):
                    thesize += self.dmetOrbs[count].shape[1] * self.dmetOrbs[count].shape[1]
                else:
                    thesize += self.imp_size[ count ] * self.imp_size[ count ]
        
        gradient = []
        for countgr in range( len( newumatflat ) ):
            error_deriv = np.zeros( [ thesize ], dtype=float )
            jump = 0
            jumpsquare = 0
            for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
                if ( self.fit_imp_bath == True ):
                    local_derivative = np.dot( np.dot( self.dmetOrbs[ count ].T, RDMderivs_rot[ countgr, :, : ] ), self.dmetOrbs[ count ] )
                else:
                    if ( self.do_det == True ) and ( self.do_det_NO == True ):
                        local_derivative = RDMderivs_rot[ countgr, jumpsquare : jumpsquare + self.imp_size[ count ],\
                                                                   jumpsquare : jumpsquare + self.imp_size[ count ] ]
                        jumpsquare += self.imp_size[ count ]
                    else:
                        local_derivative = ((RDMderivs_rot[ countgr, :, : ])[:,np.flatnonzero(self.impClust[count])])[np.flatnonzero(self.impClust[count]),:]
                if ( self.do_det == True ): # Do density embedding theory
                    local_derivative = np.diag( local_derivative )
                    error_deriv[ jump : jump + len( local_derivative ) ] = local_derivative
                    jump += len( local_derivative )
                else: # Do density MATRIX embedding theory
                    squaresize = local_derivative.shape[0] * local_derivative.shape[1]
                    error_deriv[ jump : jump + squaresize ] = np.reshape( local_derivative, squaresize, order='F' )
                    jump += squaresize
            assert ( jump == thesize )
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
            gradient = umatflat.copy()
            gradient[ cnt ] += stepsize
            gradient = self.costfunction_derivative( gradient )
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
        if ( self.TransInv == True ):
            size = self.imp_size[ 0 ]
            for it in range( 1, self.norb // size ):
                umatsquare[ it*size:(it+1)*size, it*size:(it+1)*size ] = umatsquare[ 0:size, 0:size ]

        if ( self.NOrotation != None ):
            umatsquare = np.dot( np.dot( self.NOrotation, umatsquare ), self.NOrotation.T )
        return umatsquare
        
    def square2flat( self, umatsquare ):
    
        umatsquare_bis = np.array( umatsquare, copy=True )
        if ( self.NOrotation != None ):
            umatsquare_bis = np.dot( np.dot( self.NOrotation.T, umatsquare_bis ), self.NOrotation )
        umatflat = umatsquare_bis[ self.mask ]
        return umatflat
        
    def numeleccostfunction( self, chempot_imp ):
        
        Nelec_dmet   = self.doexact( chempot_imp )
        Nelec_target = self.ints.Nelec
        print("      (chemical potential , number of electrons) = (", chempot_imp, "," , Nelec_dmet ,")")
        return Nelec_dmet - Nelec_target

    def numeleccostfunction_spinpol( self, chempot_pair ):
        """Cost function for spin-polarized DMET (independent alpha/beta mu).

        Parameters
        ----------
        chempot_pair : array-like, shape (2,)
            [mu_alpha, mu_beta].

        Returns
        -------
        ndarray shape (2,) — [error_alpha, error_beta].
        """
        mu_a, mu_b = float(chempot_pair[0]), float(chempot_pair[1])
        # Run doexact with alpha mu; beta mu is stored for the solvers to read.
        self._chempot_imp_beta = mu_b
        Nelec_total = self.doexact(mu_a)
        # Collect alpha and beta electron counts from stored spin-polarized RDMs
        Nelec_a = sum(
            np.trace(rdm.get('rdm1_alpha', rdm.get('rdm1', np.zeros((1,1))))[:s, :s])
            for rdm, s in zip(self._spinpol_rdms, self.imp_size)
        ) if hasattr(self, '_spinpol_rdms') else Nelec_total / 2.0
        Nelec_b = Nelec_total - Nelec_a
        target_a = self.ints.Nelec_alpha if hasattr(self.ints, 'Nelec_alpha') \
                   else self.ints.Nelec / 2.0
        target_b = self.ints.Nelec_beta if hasattr(self.ints, 'Nelec_beta') \
                   else self.ints.Nelec / 2.0
        print(f"      (mu_a, mu_b, N_a, N_b) = ({mu_a:.6f}, {mu_b:.6f}, "
              f"{Nelec_a:.4f}, {Nelec_b:.4f})")
        return np.array([Nelec_a - target_a, Nelec_b - target_b])

    def selfconsistent( self ):

        if self.method in ('EOM-CC', 'QD-NEVPT2', 'NEVPT2'):
            _labels = {
                'EOM-CC'    : ("EOM-CCSD",  "provides excited-state energies, not a ground-state u-matrix"),
                'QD-NEVPT2' : ("QD-NEVPT2", "Prism operates on the real molecular integrals and cannot be "
                               "embedded in the u-matrix self-consistency loop"),
                'NEVPT2'    : ("NEVPT2",    "NEVPT2 operates on the real molecular integrals and cannot be "
                               "embedded in the u-matrix self-consistency loop"),
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
            except RuntimeError as e:
                print("Warning: newton solver for chemical potential did not perfectly converge. Proceeding with last evaluated chemical potential.")
                pass
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
            print("******************************************************")
            
            if ( self.sc_method == 'NONE' ):
                u_diff = 0.1 * convergence_threshold # Do only 1 iteration
        
        print("Time cf func =", self.time_func)
        print("Time cf grad =", self.time_grad)
        print("Time dmet ed =", self.time_ed)
        print("Time dmet cf =", self.time_cf)
        
        return self.energy

    def doselfconsistent(self):
        """Deprecated alias for selfconsistent()"""
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
    
    def onedm_solution_rhf(self):
        return self.helper.construct1RDM_loc( self.doSCF, self.umat )

    def oneshot( self, mu_imp=0.0, optimize_mu=False ):
        """
        Perform a single dmet active space calculation.

        Parameters
        ----------
        mu_imp : float or array-like of length 2
            Chemical potential applied to the impurity orbitals.
            * float     — single spin-symmetric mu (default 0.0).
            * [mu_a, mu_b] — only used when optimize_mu=True and
              spin_polarized=True; seeds the 2D root search.
        optimize_mu : bool
            If True, optimize the chemical potential to match the target
            number of electrons. If False, use the fixed mu_imp.
            When spin_polarized=True, runs a 2D root search for
            (mu_alpha, mu_beta).

        Returns
        -------
        energy : float
            The correlated energy from the active space solvers.
        """
        if optimize_mu:
            from scipy import optimize
            if self.spin_polarized:
                # 2D root search: find [mu_alpha, mu_beta] independently.
                mu0 = np.array(mu_imp) if hasattr(mu_imp, '__len__') \
                      else np.array([float(mu_imp), float(mu_imp)])
                try:
                    sol = optimize.fsolve(
                        self.numeleccostfunction_spinpol,
                        mu0,
                        full_output=True,
                    )
                    mu_a, mu_b = sol[0]
                    print(f"  Spin-polarized mu: alpha={mu_a:.8f}, beta={mu_b:.8f}")
                except Exception as exc:
                    print(f"Warning: spin-polarized mu optimization failed ({exc}). "
                          f"Falling back to spin-symmetric mu.")
                    self.mu_imp = float(mu_imp) if not hasattr(mu_imp, '__len__') \
                                  else float(mu_imp[0])
                    self.doexact(self.mu_imp)
            else:
                try:
                    self.mu_imp = optimize.newton( self.numeleccostfunction, mu_imp )
                except RuntimeError:
                    print("Warning: Newton solver for mu_imp did not converge. Using last value.")
        else:
            self.mu_imp = mu_imp
            self.doexact( self.mu_imp )

        return self.energy


# ---------------------------------------------------------------------------
# Standalone helper: fragment construction by atom groups
# ---------------------------------------------------------------------------

def make_fragments( mol, myInts, atom_groups ):
    '''
    Build the impurity_clusters list required by DMET.__init__ by specifying
    groups of atom indices rather than raw orbital indices.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        The PySCF Mole object used for the mean-field calculation.
    myInts : local_integrals.localintegrals
        The localintegrals object for the system.
    atom_groups : list of lists
        Each sub-list contains the integer indices (0-based) of the atoms
        that form one impurity fragment. Every atom must appear in exactly
        one group; all atoms must be covered.

        Examples
        --------
        # 10-atom H ring, 2 atoms per impurity:
        atom_groups = [[0,1],[2,3],[4,5],[6,7],[8,9]]

        # Single large impurity containing atoms 0, 2, and 4:
        atom_groups = [[0,2,4]]

    Returns
    -------
    impurity_clusters : list of np.ndarray
        List of integer arrays of length Norbs, with 1 where the orbital
        belongs to the impurity and 0 elsewhere.

    Notes
    -----
    Uses pyscf.gto.Mole.aoslice_by_atom() to map atom indices to AO ranges,
    making the mapping robust across all basis sets.
    '''
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

    if not np.all(covered >= 0):
        raise ValueError("make_fragments: overlapping atom groups detected.")

    return impurity_clusters
