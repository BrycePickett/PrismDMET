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

import sys
import local_integrals, dmet, ring_helper
from dmet import make_fragments
from pyscf import gto, scf
from pyscf.cc import ccsd
import numpy as np

###  Disclaimer: run one of the three cases for root following
casenumber = 3 # 1, 2 or 3

if ( casenumber == 1 ):
    thecases = np.arange( 3.6, 2.88, -0.1 )
if ( casenumber == 2 ):
    thecases = np.arange( 2.4, 2.92, +0.1 )
if ( casenumber == 3 ):
    thecases = np.arange( 2.4, 1.78, -0.1 )
    
print("Bond lengths (Angstrom) =", thecases)

dm_guess = None
for bl in thecases:

    nat = 30
    mol = gto.Mole()
    mol.atom = []
    r = 0.5 * bl / np.sin(np.pi/nat)
    for i in range(nat):
        theta = i * (2*np.pi/nat)
        mol.atom.append(('Be', (r*np.cos(theta), r*np.sin(theta), 0)))

    mol.basis = { 'Be': 'sto-6g', }
    mol.build(verbose=0)

    mf = scf.RHF(mol)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=dm_guess)

    dm_guess = np.dot( np.dot( mf.mo_coeff, np.diag( mf.mo_occ ) ), mf.mo_coeff.T )

    if ( False ):   
        ccsolver = ccsd.CCSD( mf )
        ccsolver.verbose = 5
        e_corr, t1, t2 = ccsolver.ccsd()
        e_ccsd = mf.e_tot + e_corr
        print("e_ccsd for bondlength ",bl," =", e_ccsd)

    #elif ( bl < 3.35 ):
    else:
        #localization_type = 'meta_lowdin'
        #localization_type = 'boys'
        localization_type = 'meta_lowdin'
        # careful with 'iao'; the generic IAO scheme implemented in QC-dmet will not reproduce
        # results in the manuscript, which use a more careful IAO construction
        rotation = np.eye( mol.nao_nr(), dtype=float )
        for i in range(nat):
            theta  = i * (2*np.pi/nat)
            offset = 5 * i # 5 basisfunctions in sto-6g
            # Order of AO: 3s 2p 1d
            rotation[ offset+2:offset+5,  offset+2:offset+5  ] = ring_helper.p_functions( theta )
        assert( np.linalg.norm( np.dot( rotation, rotation.T ) - np.eye( rotation.shape[0] ) ) < 1e-6 )
        my_ints = local_integrals.local_integrals( mf, list(range( mol.nao_nr())), localization_type, rotation )
        if (( localization_type == 'meta_lowdin' ) or ( localization_type == 'iao' )):
            my_ints.TI_OK = True
        my_ints.molden( 'Be-loc.molden' )

        # Build fragments with the helper: 1 atom per impurity
        atom_groups = [ [i] for i in range(nat) ]
        impurity_clusters = make_fragments( mol, my_ints, atom_groups )

        if (( localization_type == 'meta_lowdin' ) or ( localization_type == 'iao' )):
            isTranslationInvariant = True
        else:
            isTranslationInvariant = False # Boys TI is not OK

        method = 'CC'
        SCmethod = 'NONE' # NONE or LSTSQ for no self-consistency or least-squares fitting of the u-matrix, respectively
        thedmet = dmet.dmet( my_ints, impurity_clusters, isTranslationInvariant,
                             method=method, SCmethod=SCmethod )
        thedmet.selfconsistent()
        #thedmet.dump_bath_orbs( 'Be-bathorbs.molden' )

