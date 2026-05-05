import numpy as np
import os
from utils import silent_stdout, nullcontext

def solve( CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, chempot_imp=0.0, printoutput=False ):
    """
    Solves the impurity problem using Block2 DMRG.
    Returns (ImpurityEnergy, OneRDM).
    """

    # Import block2 here so that QC-DMET can run without it if not using DMRG
    try:
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    except ImportError:
        raise ImportError(
            "Block2 is not installed. Please install it via: pip install block2"
        )

    FOCKcopy = FOCK.copy()
    if chempot_imp != 0.0:
        for orb in range(Nimp):
            FOCKcopy[ orb, orb ] -= chempot_imp

    with silent_stdout() if not printoutput else nullcontext():
        driver = DMRGDriver(
            scratch=os.path.join(os.getcwd(), '.block2_scratch'),
            symm_type=SymmetryTypes.SU2,
            n_threads=1
        )
        driver.initialize_system(n_sites=Norb, n_elec=Nel, spin=0)

        # Fiedler orbital reordering reduces 1D entanglement in the MPS chain.
        # The block2 driver reverses the permutation automatically in get_1pdm/get_2pdm.
        mpo = driver.get_qc_mpo(h1e=FOCKcopy, g2e=TEI, iprint=0, reorder='fiedler')

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

    ImpurityEnergy = CONST
    ImpurityEnergy += 0.5 * np.einsum('ij,ij->', rdm1[:Nimp,:], OEI[:Nimp,:] + FOCK[:Nimp,:])
    ImpurityEnergy += 0.5 * np.einsum('ijkl,ijkl->', rdm2[:Nimp,:,:,:], TEI[:Nimp,:,:,:])

    return (ImpurityEnergy, rdm1)


# ---------------------------------------------------------------------------
# SolverFactory entry point
# ---------------------------------------------------------------------------

def execute(task):
    """
    SolverFactory-compatible wrapper for the Block2 DMRG solver.

    Unpacks the standardised task dict and calls solve().

    Parameters
    ----------
    task : dict
        Must contain: CONST, dmetOEI, dmetFOCK, dmetTEI, Norb, Nel, Nimp,
        chempot_imp.

    Returns
    -------
    (ImpurityEnergy, rdm1) — same as solve().
    """
    return solve(
        task['CONST'],
        task['dmetOEI'],
        task['dmetFOCK'],
        task['dmetTEI'],
        task['Norb'],
        task['Nel'],
        task['Nimp'],
        task.get('chempot_imp', 0.0),
    )
