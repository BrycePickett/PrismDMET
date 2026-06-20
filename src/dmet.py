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

    def __init__( self, the_ints, impurity_clusters, is_translation_invariant, method='ED',
                  sc_method='LSTSQ', fit_imp_bath=True, use_constrained_opt=False,
                  do_det=False, do_det_NO=False, CC_E_TYPE='LAMBDA',
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
                  embed_level_shift=0.0, rohf_stability=False, cas_multiseed=False ):

        if is_translation_invariant:
            assert the_ints.TI_OK

        _valid_methods = {'ED', 'FCI', 'DMRG', 'CC', 'MP2', 'RHF',
                          'EOM-CC', 'CASSCF', 'QD-NEVPT2', 'NEVPT2',
                          'UHF', 'ROHF', 'RKS', 'UKS', 'ROKS'}
        assert method in _valid_methods, \
            f"DMET: unknown method='{method}'. Valid: {sorted(_valid_methods)}"
        if method in ('QD-NEVPT2', 'NEVPT2'):
            if ncas is None or nelecas is None:
                raise ValueError(
                    f"method='{method}' requires ncas and nelecas (active space size)."
                )
        if method == 'QD-NEVPT2':
            if sa_nstates < 2:
                raise ValueError(
                    "method='QD-NEVPT2' requires sa_nstates >= 2 for state-averaging."
                )
        assert sc_method in {'LSTSQ', 'BFGS', 'NONE'}
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
        self.cas_select    = cas_select
        self.ao_labels     = ao_labels
        self.avas_threshold   = avas_threshold
        self.spade_gap_tol    = spade_gap_tol
        self.spade_n_fallback = spade_n_fallback
        self.embed_level_shift = embed_level_shift   # static level shift on the embedded post-HF ROHF/RHF reference
        self.rohf_stability = rohf_stability
        self.cas_multiseed = cas_multiseed
        self.cas_results   = []   # populated by doexact() when method='CASSCF'
        self.dmrg_kwargs   = dmrg_kwargs or {}
        self.dmrg_results  = []   # populated by doexact() when method='DMRG'
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
        self.xc          = xc             # XC functional for DFT solvers
        self.level_shift = level_shift    # static level shift propagated to fragment solvers
        self.spin_polarized = spin_polarized  # enable independent alpha/beta mu optimization
        self.mm_coords  = np.asarray(mm_coords,  dtype=float) if mm_coords  is not None else None
        self.mm_charges = np.asarray(mm_charges, dtype=float) if mm_charges is not None else None

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

        maxiter_frags = 1 if is_translation_invariant else len(impurity_clusters)
        self.frag_caches = [None] * maxiter_frags

        self.minFunc    = None
        if self.altcostfunc:
            self.minFunc = 'FOCK_INIT'  # 'oei'
            assert not self.fit_imp_bath
            assert not self.do_det
            assert self.sc_method in {'BFGS', 'NONE'}

        if (( self.method == 'CC' ) and ( self.CC_E_TYPE == 'CASCI' )):
            assert( len( self.impClust ) == 1 )
        if (( self.method == 'CC' ) and ( self.CC_E_TYPE == 'EOM-CCSD' )):
            assert eom_nroots >= 1, "eom_nroots must be >= 1 when using CC_E_TYPE='EOM-CCSD'"
        if self.method == 'EOM-CC':
            from .solvers.eomcc import _VALID_EOM_TYPES as _EOM_SET
            if eom_type not in _EOM_SET:
                raise ValueError(
                    f"DMET: unknown eom_type='{eom_type}'. Valid: {sorted(_EOM_SET)}"
                )

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

    def testclusters( self ):
    
        quicktest = np.zeros([ self.norb ], dtype=int)
        for item in self.impClust:
            quicktest += np.abs(item)
        assert( np.all( quicktest >= 0 ) )
        assert( np.all( quicktest <= 1 ) )
        allOne = np.all( quicktest == 1 )
        return allOne

    def _auto_detect_symmetry( self ):
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
        if self.TransInv:
            maxiter = 1
        for counter in range( maxiter ):
            impurity_orbs = np.abs(self.impClust[ counter ])
            num_imp_orbs = np.sum( impurity_orbs )
            thearray.append( num_imp_orbs )
        thearray = np.array( thearray )
        return thearray

    def makelist_H1( self ):
    
        theH1 = []
        if self.do_det: # Do density embedding theory
            if self.TransInv: # Translational invariance assumed
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
            if self.TransInv: # Translational invariance assumed
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
        if self.do_det: # Do density embedding theory
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

        maxiter = len( self.impClust )
        if self.TransInv:
            maxiter = 1
            
        remainingOrbs = np.ones( [ len( self.impClust[ 0 ] ) ], dtype=float )

        _frag_tasks   = []
        _frag_meta    = []
        _sym_counters = set()

        _src_path = os.path.dirname(os.path.abspath(__file__))
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

            # DFT solvers need the real molecule (for XC grid) and the
            # AO-to-localized-orbital transformation to back-transform the
            # embedding density to AO space for correct XC evaluation.
            _dft_mol_info = {}
            if _method_key in ('RKS', 'UKS', 'ROKS'):
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

            task = {
                'counter'       : counter,
                'method'        : _method_key,
                'src_path'      : _src_path,
                'const'         : 0.0,
                'dmet_oei'       : dmet_oei,
                'dmet_fock'      : dmet_fock,
                'dmet_tei'       : dmet_tei,
                'norb'          : norb_in_imp,
                'nel'           : nelec_in_imp,
                'nimp'          : num_imp_orbs,
                'chempot_imp'   : chempot_imp,
                'chempot_imp_beta': self._chempot_imp_beta if self.spin_polarized else None,
                'dm_guess_rhf'    : dm_guess_rhf,
                'CC_E_TYPE'     : self.CC_E_TYPE,
                'eom_nroots'    : self.eom_nroots,
                'eom_type'      : self.eom_type,
                'eom_koopmans'  : self.eom_koopmans,
                'eom_kwargs'    : self.eom_kwargs,
                'ncas'          : self.ncas,
                'nelecas'       : self.nelecas,
                'sa_nstates'    : self.sa_nstates,
                'sa_weights'    : self.sa_weights,
                'cas_select'    : self.cas_select,
                'ao_labels'     : self.ao_labels,
                'ao_mol_dumps'  : self.ints.mol.dumps() if self.cas_select in ('ao_character', 'avas') else None,
                'ao2eo'         : (self.ints.ao2loc @ loc_2_dmet[:, :norb_in_imp]) if self.cas_select in ('ao_character', 'avas') else None,
                'avas_threshold'  : self.avas_threshold,
                'spade_gap_tol'   : self.spade_gap_tol,
                'spade_n_fallback': self.spade_n_fallback,
                'embed_level_shift': self.embed_level_shift,
                'rohf_stability'  : self.rohf_stability,
                'cas_multiseed'   : self.cas_multiseed,
                'casscf_kwargs'   : self.casscf_kwargs,
                'dmrg_kwargs'     : self.dmrg_kwargs,
                'mo_guess'      : _mo_guess,
                'oei_s'         : _dmet_oei_s,
                'nevpt2_kwargs' : self.nevpt2_kwargs,
                'qdnevpt2_kwargs': self.qdnevpt2_kwargs,
                'xc'            : self.xc,
                'spin'          : self.ints.mol.spin,
                'spin_polarized': self.spin_polarized,
                **_dft_mol_info,
            }

            from .solvers import PARALLEL_ELIGIBLE
            _is_parallel_eligible = (
                self.parallel
                and _method_key in PARALLEL_ELIGIBLE
                and not self.do_det_NO  # NO rotation requires in-process state
            )

            if _is_parallel_eligible:
                _frag_tasks.append(task)
            else:
                _frag_meta[-1]['sequential_result'] = self._run_fragment_sequential(
                    counter, _method_key, dmet_oei, dmet_fock, dmet_tei,
                    norb_in_imp, nelec_in_imp, num_imp_orbs, chempot_imp,
                    dm_guess_rhf, loc_2_dmet, mo_guess=_mo_guess, oei_s=_dmet_oei_s)

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

    def _run_fragment_sequential(self, counter, method_key,
                                   dmet_oei, dmet_fock, dmet_tei,
                                   norb_in_imp, nelec_in_imp, num_imp_orbs,
                                   chempot_imp, dm_guess_rhf, loc_2_dmet,
                                   mo_guess=None, oei_s=None):
        _mo_guess_cas, _ci_guess_cas = mo_guess, None
        if method_key == 'CASSCF' and self.frag_caches[counter] is not None:
            cached = self.frag_caches[counter]
            old_mo    = cached['mo_coeff']
            _ci_guess_cas = cached.get('ci', None)
            _ncas  = self.ncas    if self.ncas    is not None else norb_in_imp
            _ncore = (nelec_in_imp - (self.nelecas if self.nelecas is not None
                                      else nelec_in_imp)) // 2
            if old_mo.shape == (norb_in_imp, norb_in_imp):
                from .solvers.qcsolver_utils import project_amo_manually
                _mo_guess_cas, fidelity = project_amo_manually(
                    old_mo, _ncas, _ncore, dmet_fock, norb_in_imp)
                if np.min(fidelity) < 0.5:
                    print("DMET::CASSCF : Low projection fidelity, discarding CI guess.")
                    _ci_guess_cas = None
            else:
                print("DMET::CASSCF : MO shape mismatch, starting fresh.")
                _mo_guess_cas = None

        task = {
            'counter'       : counter,
            'method'        : method_key,
            'const'         : 0.0,
            'dmet_oei'       : dmet_oei,
            'dmet_fock'      : dmet_fock,
            'dmet_tei'       : dmet_tei,
            'norb'          : norb_in_imp,
            'nel'           : nelec_in_imp,
            'nimp'          : num_imp_orbs,
            'chempot_imp'   : chempot_imp,
            'chempot_imp_beta': self._chempot_imp_beta if self.spin_polarized else None,
            'dm_guess_rhf'    : dm_guess_rhf,
            'CC_E_TYPE'     : self.CC_E_TYPE,
            'eom_nroots'    : self.eom_nroots,
            'eom_type'      : self.eom_type,
            'eom_koopmans'  : self.eom_koopmans,
            'eom_kwargs'    : self.eom_kwargs,
            'ncas'          : self.ncas,
            'nelecas'       : self.nelecas,
            'sa_nstates'    : self.sa_nstates,
            'sa_weights'    : self.sa_weights,
            'cas_select'    : self.cas_select,
            'ao_labels'     : self.ao_labels,
            'ao_mol_dumps'  : self.ints.mol.dumps() if self.cas_select in ('ao_character', 'avas') else None,
            'ao2eo'         : (self.ints.ao2loc @ loc_2_dmet[:, :norb_in_imp]) if self.cas_select in ('ao_character', 'avas') else None,
            'avas_threshold'  : self.avas_threshold,
            'spade_gap_tol'   : self.spade_gap_tol,
            'spade_n_fallback': self.spade_n_fallback,
            'embed_level_shift': self.embed_level_shift,
            'rohf_stability'  : self.rohf_stability,
            'casscf_kwargs'   : self.casscf_kwargs,
            'dmrg_kwargs'   : self.dmrg_kwargs,
            'mo_guess'      : _mo_guess_cas,
            'ci_guess'      : _ci_guess_cas,
            'oei_s'         : oei_s,
            'xc'            : self.xc,
            'spin'          : self.ints.mol.spin,
            'spin_polarized': self.spin_polarized,
            'dft_mol_dumps'      : self.ints.mol.dumps() if method_key in ('RKS', 'UKS', 'ROKS') else None,
            'ao2loc'             : self.ints.ao2loc       if method_key in ('RKS', 'UKS', 'ROKS') else None,
            'loc_2_dmet'         : loc_2_dmet[:, :norb_in_imp] if method_key in ('RKS', 'UKS', 'ROKS') else None,
            'mm_coords'          : self.mm_coords  if method_key in ('RKS', 'UKS', 'ROKS') else None,
            'mm_charges'         : self.mm_charges if method_key in ('RKS', 'UKS', 'ROKS') else None,
            'nel_total'          : self.ints.Nelec if method_key in ('RKS', 'UKS', 'ROKS') else None,
            'dm_canonical_alpha'   : self.ints.fullDMao_alpha   if method_key in ('RKS', 'UKS', 'ROKS') else None,
            'dm_canonical_beta'    : self.ints.fullDMao_beta    if method_key in ('RKS', 'UKS', 'ROKS') else None,
            'fock_canonical_alpha' : self.ints.fullFOCKao_alpha if method_key in ('RKS', 'UKS', 'ROKS') else None,
            'fock_canonical_beta'  : self.ints.fullFOCKao_beta  if method_key in ('RKS', 'UKS', 'ROKS') else None,
            'level_shift'          : self.level_shift           if method_key in ('RKS', 'UKS', 'ROKS') else 0.0,
            'use_density_fit'      : self.ints.use_density_fit  if method_key in ('RKS', 'UKS', 'ROKS') else False,
            'df_auxbasis'          : self.ints.df_auxbasis      if method_key in ('RKS', 'UKS', 'ROKS') else None,
            'nevpt2_kwargs'        : self.nevpt2_kwargs,
            'qdnevpt2_kwargs'      : self.qdnevpt2_kwargs,
        }

        result = SolverDispatcher.execute(task)
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
        
        thesize = 0
        for count in range(len(self.imp_size)):
            if self.do_det: # Do density embedding theory: fit only impurity
                thesize += self.imp_size[ count ]
                assert not self.fit_imp_bath
            else: # Do density MATRIX embedding theory
                if self.fit_imp_bath:
                    thesize += self.dmetOrbs[count].shape[1] * self.dmetOrbs[count].shape[1]
                else:
                    thesize += self.imp_size[ count ] * self.imp_size[ count ]
        errors = np.zeros( [ thesize ], dtype=float )
        
        jump = 0
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            if self.fit_imp_bath:
                mf_1RDM = np.dot( np.dot( self.dmetOrbs[ count ].T, one_rdm_loc ), self.dmetOrbs[ count ] )
                ed_1RDM = self.imp_1RDM[count]
            else:
                mf_1RDM = (one_rdm_loc[:,np.flatnonzero(self.impClust[count])])[np.flatnonzero(self.impClust[count]),:]
                ed_1RDM = self.imp_1RDM[count][:self.imp_size[count],:self.imp_size[count]]
            if self.do_det: # Do density embedding theory
                if self.do_det_NO: # Work in the NO basis
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
            if self.do_det: # Do density embedding theory: fit only impurity
                thesize += self.imp_size[ count ]
                assert not self.fit_imp_bath
            else: # Do density MATRIX embedding theory
                if self.fit_imp_bath:
                    thesize += self.dmetOrbs[count].shape[1] * self.dmetOrbs[count].shape[1]
                else:
                    thesize += self.imp_size[ count ] * self.imp_size[ count ]
        
        gradient = []
        for countgr in range( len( newumatflat ) ):
            error_deriv = np.zeros( [ thesize ], dtype=float )
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
                        local_derivative = ((RDMderivs_rot[ countgr, :, : ])[:,np.flatnonzero(self.impClust[count])])[np.flatnonzero(self.impClust[count]),:]
                if self.do_det: # Do density embedding theory
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
