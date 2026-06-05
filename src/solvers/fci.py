import numpy as np
from pyscf import gto, scf, ao2mo, fci
from ..utils import silent_stdout, nullcontext

def solve( const, oei, fock, tei, norb, nel, nimp, chempot_imp=0.0, printoutput=False ):

    fock_copy = fock.copy()
    if (chempot_imp != 0.0):
        for orb in range(nimp):
            fock_copy[ orb, orb ] -= chempot_imp

    mol = gto.Mole()
    mol.build( verbose=0 )
    mol.atom.append(('H', (0, 0, 0)))
    mol.nelectron = nel
    mol.incore_anyway = True
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: fock_copy
    mf.get_ovlp = lambda *args: np.eye( norb )
    mf._eri = ao2mo.restore(8, tei, norb)

    if nel % 2 == 0:
        fci_nel = nel
    else:
        fci_nel = ((nel + 1) // 2, nel // 2)

    with silent_stdout() if not printoutput else nullcontext():
        if nel % 2 == 0:
            mf.scf()
            cisolver = fci.direct_spin0.FCI()
        else:
            mol.spin = 1
            mf = scf.ROHF(mol)
            mf.get_hcore = lambda *args: fock_copy
            mf.get_ovlp  = lambda *args: np.eye(norb)
            mf._eri      = ao2mo.restore(8, tei, norb)
            mf.scf()
            cisolver = fci.direct_spin1.FCI()
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
