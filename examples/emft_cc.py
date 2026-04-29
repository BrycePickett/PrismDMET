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

import sys
import local_integrals, dmet
from dmet import make_fragments
from pyscf import gto, scf, symm
from pyscf.cc import ccsd
import numpy as np

'''
   thestructure can be
      * 'reactants'
      * 'products'
'''
thestructure = 'reactants'
thebasis1 = 'cc-pvdz'
thebasis2 = 'aug-cc-pvdz'

###########################
#   Build the molecules   #
###########################

mol1 = gto.Mole()
mol2 = gto.Mole()

if ( thestructure == 'reactants' ):
    
    # 1-chlorodecane [http://pubs.acs.org/doi/abs/10.1021/ct5011032]
    mol1.atom = '''
         Cl        6.074014   -0.375527    0.000000
          H        4.587791    1.234857   -0.888004
          H        4.587791    1.234861    0.887998
          C        4.535917    0.606269   -0.000001
          H        3.322910   -0.929788   -0.878770
          H        3.322911   -0.929785    0.878774
          C        3.299764   -0.277657    0.000001
          H        1.996337    1.211680    0.877653
          H        1.996337    1.211677   -0.877656
          C        2.008115    0.553293   -0.000001
          H        0.753591   -0.969910   -0.876887
          H        0.753592   -0.969908    0.876889
          C        0.741064   -0.310560    0.000001
          H       -0.565286    1.168692    0.876901
          H       -0.565286    1.168693   -0.876900
          C       -0.554527    0.508674    0.000000
          H       -1.812063   -1.012283   -0.876749
          H       -1.812062   -1.012285    0.876746
          C       -1.823023   -0.352093    0.000000
          H       -3.130077    1.126541    0.876796
          H       -3.130077    1.126544   -0.876791
          C       -3.119345    0.466230    0.000001
          H       -4.378389   -1.054203   -0.876747
          H       -4.378389   -1.054207    0.876742
          C       -4.388205   -0.393603   -0.000001
          H       -5.695612    1.083112    0.876169
          H       -5.695612    1.083116   -0.876164
          C       -5.685388    0.423431    0.000001
          H       -6.982821   -1.090173   -0.882800
          H       -7.853542    0.167696    0.000001
          H       -6.982821   -1.090177    0.882796
          C       -6.947283   -0.444115   -0.000001
    '''

    # OH- [http://webbook.nist.gov/cgi/cbook.cgi?ID=C14280309&Units=SI&Mask=1000#Diatomic]
    mol2.atom = '''
          O        0.000000    0.000000    0.000000
          H        0.000000    0.000000    0.970000
    '''

if ( thestructure == 'products' ):

    # 1-decanol [http://pubs.acs.org/doi/abs/10.1021/ct5011032]
    mol1.atom = '''
          O       -6.326993   -0.342551   -0.085075
          H       -6.343597   -0.932869    0.674603
          H       -5.172280    1.115811   -0.856164
          H       -5.163379    1.078922    0.909461
          C       -5.143976    0.447568    0.008436
          H       -3.870394   -1.068468    0.848317
          H       -3.886864   -1.023249   -0.909394
          C       -3.866782   -0.389712   -0.015871
          H       -2.586810    1.131129   -0.857112
          H       -2.589011    1.097064    0.896328
          C       -2.587307    0.454246    0.006651
          H       -1.301618   -1.058312    0.856288
          H       -1.301459   -1.025748   -0.896865
          C       -1.302629   -0.382041   -0.008161
          H        0.024656    1.137733   -0.856268
          H       -0.024980    1.105335    0.896879
          C       -0.022767    0.461248    0.008048
          H        1.263962   -1.051267    0.856601
          H        1.264619   -1.017717   -0.896424
          C        1.262430   -0.374150   -0.007297
          H        2.541053    1.146168   -0.854028
          H        2.540990    1.111571    0.899047
          C        2.542888    0.468512    0.009449
          H        3.830289   -1.045369    0.855777
          H        3.831311   -1.009068   -0.897251
          C        3.828011   -0.366637   -0.007063
          H        5.107586    1.152957   -0.850500
          H        5.107033    1.115433    0.901323
          C        5.109507    0.474593    0.011259
          H        6.434418   -1.033802    0.861582
          H        7.282187    0.259647    0.008368
          H        6.435992   -0.994161   -0.903519
          C        6.387550   -0.368849   -0.006524
    '''

    mol2.atom = '''
          Cl        0.000000    0.000000    0.000000
    '''

mol1.basis = { 'H':  thebasis1,
               'C':  thebasis1,
               'O':  thebasis2,
               'Cl': thebasis2 }
mol2.basis = { 'H':  thebasis1,
               'C':  thebasis1,
               'O':  thebasis2,
               'Cl': thebasis2 }
mol1.symmetry = 0
mol2.symmetry = 0
mol1.charge =  0
mol2.charge = -1
mol1.spin = 0
mol2.spin = 0
mol1.build()
mol2.build()

####################################
#   Perform the RHF calculations   #
####################################
mf1 = scf.RHF( mol1 )
mf1.verbose = 4
mf1.scf()
mf2 = scf.RHF( mol2 )
mf2.verbose = 4
mf2.scf()

######################
#   CCSD reference   #
######################

# Always calculate the CCSD energy of the small molecule (mol2)
ccsolver2 = ccsd.CCSD( mf2 )
ccsolver2.verbose = 5
ECORR2, t1, t2 = ccsolver2.ccsd()
ERHF2 = mf2.e_tot
ECCSD2 = ERHF2 + ECORR2

############
#   DMET   #
############

if ( True ):
    # Perform DMET on the large system (mol1)
    myInts = local_integrals.localintegrals( mf1, list(range( mol1.nao_nr())), 'meta_lowdin' )
    myInts.molden( 'emft-loc.molden' )
    
    # Define physical units by atom index: 
    if ( thestructure == 'reactants' ): # 1-chlorodecane
        # Unit 0: [Cl], Unit 1: [H,H,C], Units 2-9: [H,H,C], Unit 10: [H,H,H,C]
        atom_units = [ [0], [1,2,3] ] + [ [4+3*i, 5+3*i, 6+3*i] for i in range(8) ] + [ [28,29,30,31] ]
    else: # 1-decanol
        # Unit 0: [O,H], Unit 1: [H,H,C], Units 2-9: [H,H,C], Unit 10: [H,H,H,C]
        atom_units = [ [0,1], [2,3,4] ] + [ [5+3*i, 6+3*i, 7+3*i] for i in range(8) ] + [ [29,30,31,32] ] # Wait, products decanol atom count...

    for n_carbon_units in range( 0, 5 ): # 0, 1, 2, 3, 4 units + Unit 0
        atom_groups = [ [item for sub in atom_units[0 : n_carbon_units+1] for item in sub] ]
        impurityClusters = make_fragments( mol1, myInts, atom_groups )

        theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant=False, 
                             method='CC', SCmethod='NONE', CC_E_TYPE='CASCI' )
        
        the_energy = theDMET.selfconsistent()
        print("######  DMET(", n_carbon_units,"C , CCSD ) /", thebasis1, "/", thebasis2, " =", the_energy + ECCSD2)
