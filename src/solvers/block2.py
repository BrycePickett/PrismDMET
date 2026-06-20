import numpy as np
import os
from ..utils import silent_stdout, nullcontext

def solve( const, oei, fock, tei, norb, nel, nimp, chempot_imp=0.0, printoutput=False,
           bond_dims=None, noises=None, thrds=None, n_sweeps=5, n_threads=None ):
    '''Solve the impurity problem with Block2 DMRG. Returns (impurity_energy, rdm1, dmrg_res).'''

    # Import block2 here so that QC-dmet can run without it if not using DMRG
    try:
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    except ImportError:
        raise ImportError(
            "Block2 is not installed. Please install it via: pip install block2"
        )

    if n_threads is None:
        n_threads = int(os.environ.get('OMP_NUM_THREADS', 1))
    if bond_dims is None:
        bond_dims = [250] * n_sweeps
    if noises is None:
        noises = [0] * n_sweeps
    if thrds is None:
        thrds = [1e-8] * n_sweeps

    fock_copy = fock.copy()
    if chempot_imp != 0.0:
        for orb in range(nimp):
            fock_copy[ orb, orb ] -= chempot_imp

    with silent_stdout() if not printoutput else nullcontext():
        driver = DMRGDriver(
            scratch=os.path.join(os.getcwd(), '.block2_scratch'),
            symm_type=SymmetryTypes.SU2,
            n_threads=n_threads
        )
        driver.initialize_system(n_sites=norb, n_elec=nel, spin=nel % 2)

        # Fiedler reordering reduces MPS entanglement; block2 auto-reverses it in get_1pdm/get_2pdm.
        mpo = driver.get_qc_mpo(h1e=fock_copy, g2e=tei, iprint=0, reorder='fiedler')

        ket = driver.get_random_mps(tag='gs', bond_dim=bond_dims[0], nroots=1)
        EnergyDMRG = driver.dmrg(mpo, ket, n_sweeps=n_sweeps,
                                 bond_dims=bond_dims, noises=noises,
                                 thrds=thrds, iprint=0)

        _dw = getattr(driver._dmrg, 'discarded_weights', None)
        discarded_weight = float(_dw[-1]) if _dw is not None and len(_dw) else None

        rdm1 = driver.get_1pdm(ket)
        # Block2 2-RDM convention: transpose (0,3,1,2) maps to PySCF convention.
        rdm2 = driver.get_2pdm(ket).transpose(0, 3, 1, 2)

    impurity_energy = const
    impurity_energy += 0.5 * np.einsum('ij,ij->', rdm1[:nimp,:], oei[:nimp,:] + fock[:nimp,:])
    impurity_energy += 0.5 * np.einsum('ijkl,ijkl->', rdm2[:nimp,:,:,:], tei[:nimp,:,:,:])

    dmrg_res = {
        'energy'           : float(EnergyDMRG),
        'discarded_weight' : discarded_weight,
        'bond_dims'        : list(bond_dims),
        'noises'           : list(noises),
        'n_sweeps'         : n_sweeps,
        'n_threads'        : n_threads,
    }
    return (impurity_energy, rdm1, dmrg_res)


def execute(task):
    """SolverDispatcher entry point for the Block2 DMRG solver."""
    dmrg_kwargs = dict(task.get('dmrg_kwargs') or {})
    return solve(
        task['const'],
        task['dmet_oei'],
        task['dmet_fock'],
        task['dmet_tei'],
        task['norb'],
        task['nel'],
        task['nimp'],
        chempot_imp=task.get('chempot_imp', 0.0),
        **dmrg_kwargs,
    )
