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
from pyscf import ao2mo, gto, scf

def solve( const, oei, fock, tei, norb, nel, nimp, dm_guess_rhf, chempot_imp=0.0 ):

    fock_copy = fock.copy()
    if (chempot_imp != 0.0):
        for orb in range(nimp):
            fock_copy[ orb, orb ] -= chempot_imp
    
    mol = gto.Mole()
    mol.build( verbose=0 )
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = nel
    mol.incore_anyway = True
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: fock_copy
    mf.get_ovlp = lambda *args: np.eye( norb )
    mf._eri = ao2mo.restore(8, tei, norb)
    mf.scf( dm_guess_rhf )
    dm_loc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    if not mf.converged:
        mf = mf.newton()
        mf.scf( dm_loc )
        dm_loc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )

    
    e_rhf = mf.e_tot
    RDM1 = mf.make_rdm1()
    JK   = mf.get_veff(None, dm=RDM1)
 
    # Half-projector: 0.5*(oei + fock) avoids double-counting JK.
    impurity_energy = const \
                   + 0.25 * np.einsum('ji,ij->', RDM1[:,:nimp], fock[:nimp,:] + oei[:nimp,:]) \
                   + 0.25 * np.einsum('ji,ij->', RDM1[:nimp,:], fock[:,:nimp] + oei[:,:nimp]) \
                   + 0.25 * np.einsum('ji,ij->', RDM1[:,:nimp], JK[:nimp,:]) \
                   + 0.25 * np.einsum('ji,ij->', RDM1[:nimp,:], JK[:,:nimp])
    
    return ( impurity_energy, RDM1 )


# ---------------------------------------------------------------------------
# SolverDispatcher entry point
# ---------------------------------------------------------------------------

def execute(task):
    """
    SolverDispatcher-compatible wrapper for the RHF solver.

    Unpacks the standardised task dict and calls solve().

    Parameters
    ----------
    task : dict
        Must contain: const, dmet_oei, dmet_fock, dmet_tei, norb, nel, nimp,
        chempot_imp, dm_guess_rhf.

    Returns
    -------
    (impurity_energy, RDM1) — same as solve().
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
