import numpy as np
from pyscf import fci
from ..utils import silent_stdout, nullcontext

def solve( const, oei, fock, tei, norb, nel, nimp, chempot_imp=0.0, printoutput=False ):

    fock_copy = fock.copy()
    if (chempot_imp != 0.0):
        for orb in range(nimp):
            fock_copy[ orb, orb ] -= chempot_imp

    if nel % 2 == 0:
        fci_nel = nel
        cisolver = fci.direct_spin0.FCI()
    else:
        fci_nel = ((nel + 1) // 2, nel // 2)
        cisolver = fci.direct_spin1.FCI()

    with silent_stdout() if not printoutput else nullcontext():
        cisolver.verbose = 0
        cisolver.max_cycle = 200
        cisolver.conv_tol = 1e-12
        EnergyFCI, FCIvector = cisolver.kernel( fock_copy, tei, norb, fci_nel, ecore=const )
        two_rdm = cisolver.make_rdm2( FCIvector, norb, fci_nel )

    one_rdm = np.einsum( 'ijkk->ij', two_rdm ) / ( nel - 1 )

    impurity_energy = const
    impurity_energy += 0.5 * np.einsum( 'ij,ij->', one_rdm[:nimp,:], oei[:nimp,:] + fock[:nimp,:] )
    impurity_energy += 0.5 * np.einsum( 'ijkl,ijkl->', two_rdm[:nimp,:,:,:], tei[:nimp,:,:,:] )
    return ( impurity_energy, one_rdm )


def execute(task):
    """SolverDispatcher entry point for the FCI solver."""
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
