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
        # QC-DMET assumes closed-shell singlets; SU2 symmetry allows Block2 to
        # natively spin-trace the RDMs to match PySCF convention exactly.
        driver = DMRGDriver(
            scratch=os.path.join(os.getcwd(), '.block2_scratch'),
            symm_type=SymmetryTypes.SU2,
            n_threads=1
        )
        driver.initialize_system(n_sites=Norb, n_elec=Nel, spin=0)

        # Setup the QC MPO (Block2 expects chemist's notation TEI)
        mpo = driver.get_qc_mpo(h1e=FOCKcopy, g2e=TEI, iprint=0)

        # DMRG sweeps: bond_dim=250 is sufficient for typical DMET impurity spaces
        bond_dims = [250, 250, 250, 250, 250]
        noises     = [0,   0,   0,   0,   0  ]
        thrds      = [1e-8]*5

        ket = driver.get_random_mps(tag='gs', bond_dim=bond_dims[0], nroots=1)
        EnergyDMRG = driver.dmrg(mpo, ket, n_sweeps=5,
                                 bond_dims=bond_dims, noises=noises,
                                 thrds=thrds, iprint=0)

        # 1-RDM: SU2 get_1pdm returns spin-traced RDM identical to PySCF
        rdm1 = driver.get_1pdm(ket)
        # 2-RDM: Block2 returns <i+ j+ k l>; PySCF wants <i+ k+ l j>.
        # Transpose (0,3,1,2) maps Block2 -> PySCF convention.
        rdm2 = driver.get_2pdm(ket).transpose(0, 3, 1, 2)

    ImpurityEnergy = CONST
    ImpurityEnergy += 0.5 * np.einsum('ij,ij->', rdm1[:Nimp,:], OEI[:Nimp,:] + FOCK[:Nimp,:])
    ImpurityEnergy += 0.5 * np.einsum('ijkl,ijkl->', rdm2[:Nimp,:,:,:], TEI[:Nimp,:,:,:])

    return (ImpurityEnergy, rdm1)
