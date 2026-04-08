import numpy as np
import qcdmet_paths
from pyscf import gto, scf, ao2mo, fci
from utils import silent_stdout, nullcontext

def solve( CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, chempot_imp=0.0, printoutput=False ):

    FOCKcopy = FOCK.copy()
    if (chempot_imp != 0.0):
        for orb in range(Nimp):
            FOCKcopy[ orb, orb ] -= chempot_imp

    mol = gto.Mole()
    mol.build( verbose=0 )
    mol.atom.append(('H', (0, 0, 0)))
    mol.nelectron = Nel
    mol.incore_anyway = True
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: FOCKcopy
    mf.get_ovlp = lambda *args: np.eye( Norb )
    mf._eri = ao2mo.restore(8, TEI, Norb)

    with silent_stdout() if not printoutput else nullcontext():
        mf.scf()

        assert( Nel % 2 == 0 )
        cisolver = fci.direct_spin0.FCI()
        cisolver.verbose = 0
        cisolver.max_cycle = 200
        cisolver.conv_tol = 1e-12
        EnergyFCI, FCIvector = cisolver.kernel( FOCKcopy, TEI, Norb, Nel, ecore=CONST )
        TwoRDM = cisolver.make_rdm2( FCIvector, Norb, Nel )

    OneRDM = np.einsum( 'ijkk->ij', TwoRDM ) / ( Nel - 1 )

    ImpurityEnergy = CONST
    ImpurityEnergy += 0.5 * np.einsum( 'ij,ij->', OneRDM[:Nimp,:], OEI[:Nimp,:] + FOCK[:Nimp,:] )
    ImpurityEnergy += 0.5 * np.einsum( 'ijkl,ijkl->', TwoRDM[:Nimp,:,:,:], TEI[:Nimp,:,:,:] )
    return ( ImpurityEnergy, OneRDM )
