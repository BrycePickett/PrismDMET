'''
    QC-dmet: a python implementation of density matrix embedding theory for ab initio quantum chemistry
    Copyright (C) 2015 Sebastian Wouters
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

import numpy as np
from pyscf import gto, scf, ao2mo, mp
from ..utils import silent_stdout, nullcontext

def solve( const, oei, fock, tei, norb, nel, nimp, dm_guess_rhf, chempot_imp=0.0, printoutput=True ):

    ctx = silent_stdout() if not printoutput else nullcontext()

    # Augment the fock operator with the chemical potential
    fock_copy = fock.copy()
    if (chempot_imp != 0.0):
        for orb in range(nimp):
            fock_copy[ orb, orb ] -= chempot_imp

    if nel % 2 != 0:
        raise NotImplementedError(
            'MP2 solver requires an even number of electrons (RHF reference only). '
            'Use CC or FCI for open-shell embedding problems.'
        )

    with ctx:
        mol = gto.Mole()
        mol.build(verbose=0)
        mol.atom.append(('H', (0, 0, 0)))
        mol.nelectron = nel
        mf = scf.RHF( mol )
        mf.get_hcore = lambda *args: fock_copy
        mf.get_ovlp = lambda *args: np.eye( norb )
        mf._eri = ao2mo.restore(8, tei, norb)
        mf.scf( dm_guess_rhf )

        # Get the MP2 solution. PySCF's response densities include the HF
        # reference, so E = Tr[h dm1] + 0.5*dm2.eri reproduces E(MP2) exactly.
        myMP2 = mp.MP2( mf )
        E_MP2, T_MP2 = myMP2.kernel()
        OneRDM_mo = myMP2.make_rdm1()
        TwoRDM_mo = myMP2.make_rdm2() # 2-RDM is stored in chemistry notation!
        one_rdm_loc = np.dot(mf.mo_coeff, np.dot( OneRDM_mo, mf.mo_coeff.T ))
        TwoRDM_loc = np.einsum('ai,ijkl->ajkl', mf.mo_coeff, TwoRDM_mo )
        TwoRDM_loc = np.einsum('bj,ajkl->abkl', mf.mo_coeff, TwoRDM_loc)
        TwoRDM_loc = np.einsum('ck,abkl->abcl', mf.mo_coeff, TwoRDM_loc)
        TwoRDM_loc = np.einsum('dl,abcl->abcd', mf.mo_coeff, TwoRDM_loc)
    
    # Half-projector: 0.5*(oei + fock) avoids double-counting JK.
    impurity_energy = const
    impurity_energy += 0.5 * np.einsum( 'ij,ij->', one_rdm_loc[:nimp,:], oei[:nimp,:] + fock[:nimp,:] )
    impurity_energy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:nimp,:,:,:], tei[:nimp,:,:,:] )
    impurity_energy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:,:nimp,:,:], tei[:,:nimp,:,:] )
    impurity_energy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:,:,:nimp,:], tei[:,:,:nimp,:] )
    impurity_energy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:,:,:,:nimp], tei[:,:,:,:nimp] )
    return ( impurity_energy, one_rdm_loc )


def execute(task):
    """SolverDispatcher entry point for the MP2 solver."""
    return solve(
        task['const'],
        task['dmet_oei'],
        task['dmet_fock'],
        task['dmet_tei'],
        task['norb'],
        task['nel'],
        task['nimp'],
        task.get('dm_guess_rhf'),
        task.get('chempot_imp', 0.0),
    )
