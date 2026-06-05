import numpy as np
import os
from ..utils import silent_stdout, nullcontext

def solve( const, oei, fock, tei, norb, nel, nimp, chempot_imp=0.0, printoutput=False ):
    """Solve the impurity problem with Block2 DMRG. Returns (impurity_energy, rdm1)."""

    # Import block2 here so that QC-dmet can run without it if not using DMRG
    try:
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    except ImportError:
        raise ImportError(
            "Block2 is not installed. Please install it via: pip install block2"
        )

    fock_copy = fock.copy()
    if chempot_imp != 0.0:
        for orb in range(nimp):
            fock_copy[ orb, orb ] -= chempot_imp

    with silent_stdout() if not printoutput else nullcontext():
        driver = DMRGDriver(
            scratch=os.path.join(os.getcwd(), '.block2_scratch'),
            symm_type=SymmetryTypes.SU2,
            n_threads=1
        )
        driver.initialize_system(n_sites=norb, n_elec=nel, spin=0)

        # Fiedler reordering reduces MPS entanglement; block2 auto-reverses it in get_1pdm/get_2pdm.
        mpo = driver.get_qc_mpo(h1e=fock_copy, g2e=tei, iprint=0, reorder='fiedler')

        bond_dims = [250, 250, 250, 250, 250]
        noises    = [0,   0,   0,   0,   0  ]
        thrds     = [1e-8]*5

        ket = driver.get_random_mps(tag='gs', bond_dim=bond_dims[0], nroots=1)
        EnergyDMRG = driver.dmrg(mpo, ket, n_sweeps=5,
                                 bond_dims=bond_dims, noises=noises,
                                 thrds=thrds, iprint=0)

        rdm1 = driver.get_1pdm(ket)
        # Block2 2-RDM convention: transpose (0,3,1,2) maps to PySCF convention.
        rdm2 = driver.get_2pdm(ket).transpose(0, 3, 1, 2)

    impurity_energy = const
    impurity_energy += 0.5 * np.einsum('ij,ij->', rdm1[:nimp,:], oei[:nimp,:] + fock[:nimp,:])
    impurity_energy += 0.5 * np.einsum('ijkl,ijkl->', rdm2[:nimp,:,:,:], tei[:nimp,:,:,:])

    return (impurity_energy, rdm1)


def execute(task):
    """SolverDispatcher entry point for the Block2 DMRG solver."""
    return solve(
        task['const'],
        task['dmet_oei'],
        task['dmet_fock'],
        task['dmet_tei'],
        task['norb'],
        task['nel'],
        task['nimp'],
        task.get('chempot_imp', 0.0),
    )
