'''
    QC-DMET: a python implementation of density matrix embedding theory for ab initio quantum chemistry
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

# C12H25Br + Cl-  -->  C12H25Cl + Br-

import sys
sys.path.append('../src')
import local_integrals, dmet, qcdmet_paths
from dmet import make_fragments
from pyscf import gto, scf, symm
from pyscf.cc import ccsd
import numpy as np
import sn2_struct_bis as sn2_structures_bis

#############
#   Input   #
#############
thestructure = 0                    # 'reactants' or 'products' or any integer in the range [-9, 10] (boundaries included)
cluster_sizes = np.arange( 1, 7 )   # Number of carbon atoms per cluster
localization = 'iao'                # 'iao' or 'meta_lowdin' or 'boys'
single_impurity = True              # Single impurity vs. partitioning
one_bath_orb_per_bond = True        # Sun & Chan, JCTC 10, 3784 (2014) [ http://dx.doi.org/10.1021/ct500512f ]
casci_energy_formula = True         # CASCI or DMET energy formula

#######################
#   Parse the input   #
#######################
thebasis1 = 'cc-pvdz'       # Basis set for H and C
thebasis2 = 'aug-cc-pvdz'   # Basis set for Cl and Br
mol = sn2_structures_bis.structure( thestructure, thebasis1, thebasis2 )
mf = scf.RHF( mol )
mf.verbose = 4
mf.scf()

if ( False ):
    from pyscf.tools import molden
    with open( 'sn2-mo.molden', 'w' ) as thefile:
        molden.header( mol, thefile )
        molden.orbital_coeff( mol, thefile, mf.mo_coeff )

if ( False ):
    ccsolver = ccsd.CCSD( mf )
    ccsolver.verbose = 5
    ECORR, t1, t2 = ccsolver.ccsd()
    ECCSD = mf.e_tot + ECORR
    print("ERHF  for structure", thestructure, "=", mf.e_tot)
    print("ECCSD for structure", thestructure, "=", ECCSD)
    
if ( True ):
    myInts = local_integrals.localintegrals( mf, list(range( mol.nao_nr())), 'iao' )
    myInts.molden( 'sn2-loc.molden' )
    
    # Define physical units by atom index for sn2_bis (Cl, Br, C12H25 chain): 
    # Unit 0: (C, H, H, Cl, Br) -> Atoms [0,1,2,3,4]
    # Units 1-10: (C, H, H) 
    # Unit 11: (C, H, H, H)
    atom_units = [ [0,1,2,3,4] ] + [ [5+3*i, 6+3*i, 7+3*i] for i in range(10) ] + [ [35, 36, 37, 38] ]

    for carbons_in_cluster in cluster_sizes:
        if ( casci_energy_formula ): # Do only 1 impurity at the edge
            atom_groups = [ [item for sub in atom_units[0:carbons_in_cluster] for item in sub] ]
        else: # Partition the whole system
            atom_groups = []
            for i in range(0, len(atom_units), carbons_in_cluster):
                group = [item for sub in atom_units[i : i+carbons_in_cluster] for item in sub]
                atom_groups.append(group)
        
        impurityClusters = make_fragments( mol, myInts, atom_groups )

        # Apply freezing (RHF solver) for non-edge impurities if using single_impurity partitioning
        if ( not casci_energy_formula and single_impurity ):
            for i in range(1, len(impurityClusters)):
                impurityClusters[i] *= -1

        theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant=False, 
                             method='CC', SCmethod='NONE',
                             CC_E_TYPE='CASCI' if casci_energy_formula else 'CCSD' )

        if ( one_bath_orb_per_bond == True ):
            theDMET.BATH_ORBS = 2 * np.ones( [ len(impurityClusters) ], dtype=int )
            theDMET.BATH_ORBS[ 0 ] = 1
            theDMET.BATH_ORBS[ len(impurityClusters) - 1 ] = 1
            
        the_energy = theDMET.selfconsistent()
        print("######  DMET(", carbons_in_cluster,"C , CCSD ) /", thebasis1, "/", thebasis2, " =", the_energy)
