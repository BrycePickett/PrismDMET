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
import local_integrals, dmet
from dmet import make_fragments
from pyscf import gto, scf, mp
from pyscf.cc import ccsd
import numpy as np

# Parameter alpha from JCP 141, 054113 (2014)
for alpha in np.arange(0.88, 1.13, 0.02):

    b1 = 1.263 * alpha
    b2 = 1.320 * alpha
    nat = 12

    sine  = np.sin( 2 * np.pi / nat )
    cosin = np.cos( 2 * np.pi / nat )
    R     = np.sqrt( 0.25 * b1 * b1 + ( ( b2 + b1 * cosin ) / ( 2 * sine ) ) ** 2 )
    shift = 2 * np.arcsin( 0.5 * b1 / R )

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = []

    angle = 0.0
    for i in range( nat // 2 ):
        mol.atom.append(('C', (R * np.cos(angle        ), R * np.sin(angle        ), 0.0)))
        mol.atom.append(('C', (R * np.cos(angle + shift), R * np.sin(angle + shift), 0.0)))
        angle += 4.0 * np.pi / nat
    
    #mol.basis = 'cc-pvdz'
    mol.basis = '6-31g'
    mol.build()

    mf = scf.RHF( mol )
    mf.verbose = 3
    mf.scf()

    if ( False ):   
        ccsolver = ccsd.CCSD( mf )
        ccsolver.verbose = 5
        e_corr, t1, t2 = ccsolver.ccsd()
        e_ccsd = mf.e_tot + e_corr
        print("e_ccsd for alpha ",alpha," =", e_ccsd)
        
    if ( False ):
        mp2solver = mp.MP2( mf )
        e_corr, t_mp2 = mp2solver.kernel()
        EMP2 = mf.e_tot + e_corr
        print("EMP2 for alpha ",alpha," =", EMP2)
    
    my_ints = local_integrals.LocalIntegrals( mf, list(range( mol.nao_nr())), 'meta_lowdin' )
    my_ints.molden( 'polyyne-loc.molden' )

    # Build fragments with the helper: 4 atoms per impurity
    atoms_per_imp = 4
    atom_groups = [ list(range(i, i+atoms_per_imp)) for i in range(0, nat, atoms_per_imp) ]
    impurity_clusters = make_fragments( mol, my_ints, atom_groups )

    is_translation_invariant = False # Both in meta_lowdin (due to px, py) and Boys TI is not OK
    method = 'CC'
    sc_method = 'NONE' #Don't do it self-consistently
    the_dmet = dmet.DMET( my_ints, impurity_clusters, is_translation_invariant,
                         method=method, sc_method=sc_method )
    the_dmet.selfconsistent()
    #the_dmet.dump_bath_orbs( 'polyyne-bath.molden' )

