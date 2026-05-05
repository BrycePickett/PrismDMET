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
import local_integrals_hubbard, dmet
import numpy as np

HubbardU   = 1.0
Norbs      = 240
imp_size   = 2

assert ( Norbs % imp_size == 0 )

fillings = []
energies = []

for Nelectrons in range( 12, 241, 12 ):

   hopping  = np.zeros( [Norbs, Norbs], dtype=float )
   for orb in range(Norbs-1):
       hopping[ orb, orb+1 ] = -1.0
       hopping[ orb+1, orb ] = -1.0
   hopping[ 0, Norbs-1 ] = 1.0 # anti-PBC
   hopping[ Norbs-1, 0 ] = 1.0 # anti-PBC

   my_ints = local_integrals_hubbard.local_integrals_hubbard( hopping, HubbardU, Nelectrons )

   impurity_clusters = []
   for cluster in range( Norbs // imp_size ):
       impurities = np.zeros( [ my_ints.Norbs ], dtype=int )
       for orb in range( cluster*imp_size, (cluster+1)*imp_size ):
           impurities[ orb ] = 1
       impurity_clusters.append( impurities )

   totalcount = np.zeros( [ my_ints.Norbs ], dtype=int )
   for item in impurity_clusters:
       totalcount += item
   assert ( np.linalg.norm( totalcount - np.ones( [ my_ints.Norbs ], dtype=float ) ) < 1e-12 )

   isTranslationInvariant = True
   method = 'ED'
   SCmethod = 'LSTSQ' # 'LSTSQ'
   thedmet = dmet.dmet( my_ints, impurity_clusters, isTranslationInvariant, 
                        method=method, SCmethod=SCmethod )
   #oldUMAT = 0.33 * ( 2 * np.random.rand( Norbs, Norbs ) - 1 )
   #if ( oldUMAT != None ):
   #    thedmet.umat = thedmet.flat2square( thedmet.square2flat( oldUMAT ) )
   theEnergy = thedmet.selfconsistent()
   
   fillings.append( (1.0 * Nelectrons) / Norbs )
   energies.append( theEnergy / Norbs )

np.set_printoptions(precision=8, linewidth=160)
print("For U =", HubbardU,"and Norbs =", Norbs)
print("Fillings =")
print(np.array( fillings ))
print("E / site =")
print(np.array( energies ))


