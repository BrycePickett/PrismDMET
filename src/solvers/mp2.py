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
import ctypes
import rhf
import local_integrals
from pyscf import gto, scf, ao2mo, mp
from utils import silent_stdout, nullcontext

def solve( const, oei, fock, tei, norb, nel, nimp, dm_guess_rhf, chempot_imp=0.0, printoutput=True ):

    ctx = silent_stdout() if not printoutput else nullcontext()

    # Augment the fock operator with the chemical potential
    fock_copy = fock.copy()
    if (chempot_imp != 0.0):
        for orb in range(nimp):
            fock_copy[ orb, orb ] -= chempot_imp

    with ctx:
        # Get the RHF solution
        assert( nel % 2 == 0 )
        mol = gto.Mole()
        mol.build(verbose=0)
        mol.atom.append(('H', (0, 0, 0)))
        mol.nelectron = nel
        mf = scf.RHF( mol )
        mf.get_hcore = lambda *args: fock_copy
        mf.get_ovlp = lambda *args: np.eye( norb )
        mf._eri = ao2mo.restore(8, tei, norb)
        mf.scf( dm_guess_rhf )
        DMrhf = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
        Erhf  = const + np.einsum('ij,ij->', fock_copy, DMrhf)
        Erhf += 0.5 * np.einsum('ijkl,ij,kl->', tei, DMrhf, DMrhf) - 0.25 * np.einsum('ijkl,ik,jl->', tei, DMrhf, DMrhf)
        numPairs = nel // 2
        print("mp2::solve : RHF homo-lumo gap =", mf.mo_energy[numPairs] - mf.mo_energy[numPairs-1])

        # Get the MP2 solution
        myMP2 = mp.MP2( mf )
        E_MP2, T_MP2 = myMP2.kernel()
        OneRDM_mo = np.zeros( [norb, norb], dtype=float )
        TwoRDM_mo = myMP2.make_rdm2() # 2-RDM is stored in chemistry notation!

        # Reconstruct HF reference contributions
        Etotal = Erhf + E_MP2
        for orb1 in range(numPairs):
            OneRDM_mo[orb1, orb1] += 2.0
            for orb2 in range(numPairs):
                TwoRDM_mo[orb1,orb1,orb2,orb2] += 4.0
                TwoRDM_mo[orb1,orb2,orb1,orb2] -= 2.0
        one_rdm_loc = np.dot(mf.mo_coeff, np.dot( OneRDM_mo, mf.mo_coeff.T ))
        TwoRDM_loc = np.einsum('ai,ijkl->ajkl', mf.mo_coeff, TwoRDM_mo )
        TwoRDM_loc = np.einsum('bj,ajkl->abkl', mf.mo_coeff, TwoRDM_loc)
        TwoRDM_loc = np.einsum('ck,abkl->abcl', mf.mo_coeff, TwoRDM_loc)
        TwoRDM_loc = np.einsum('dl,abcl->abcd', mf.mo_coeff, TwoRDM_loc)
        Etotal2 = Erhf - 0.5 * np.einsum('ijkl,ij,kl->', tei, DMrhf, DMrhf) + 0.25 * np.einsum('ijkl,ik,jl->', tei, DMrhf, DMrhf) + 0.5 * np.einsum('ijkl,ijkl->', tei, TwoRDM_loc)
        print("Etotal  =", Etotal)
        print("Etotal2 =", Etotal2)
    
    # To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( oei + fock ) = oei + 0.5 * JK
    impurity_energy = const
    impurity_energy += 0.5 * np.einsum( 'ij,ij->', DMrhf[:nimp,:], oei[:nimp,:] + fock[:nimp,:] ) # To be consistent with the energy formula above, this should be the HF RDM !!!
    impurity_energy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:nimp,:,:,:], tei[:nimp,:,:,:] )
    impurity_energy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:,:nimp,:,:], tei[:,:nimp,:,:] )
    impurity_energy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:,:,:nimp,:], tei[:,:,:nimp,:] )
    impurity_energy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:,:,:,:nimp], tei[:,:,:,:nimp] )
    return ( impurity_energy, one_rdm_loc )


# ---------------------------------------------------------------------------
# solver_dispatcher entry point
# ---------------------------------------------------------------------------

def execute(task):
    """
    solver_dispatcher-compatible wrapper for the MP2 solver.

    Unpacks the standardised task dict and calls solve().

    Parameters
    ----------
    task : dict
        Must contain: const, dmet_oei, dmet_fock, dmet_tei, norb, nel, nimp,
        dm_guess_rhf, chempot_imp.

    Returns
    -------
    (impurity_energy, one_rdm_loc) — same as solve().
    """
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
