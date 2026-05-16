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
from pyscf import gto, scf, ao2mo
import numpy as np

dm_guess  = None
#old_umat = None

bondlengths = np.array([0.7, 1.0])
#bondlengths = np.arange(0.6, 3.05, 0.1)[::-1]
energies = []

for bondlength in bondlengths:

    nat = 10
    mol = gto.Mole()
    mol.atom = []
    r = 0.5 * bondlength / np.sin(np.pi/nat)
    for i in range(nat):
        theta = i * (2*np.pi/nat)
        mol.atom.append(('H', (r*np.cos(theta), r*np.sin(theta), 0)))

    mol.basis = 'sto-3g'
    mol.build(verbose=0)

    mf = scf.RHF(mol)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=dm_guess)

    if ( False ):   
        ENUCL = mf.mol.energy_nuc()
        OEI   = np.dot(np.dot(mf.mo_coeff.T, mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')), mf.mo_coeff)
        TEI   = ao2mo.outcore.full_iofree(mol, mf.mo_coeff, compact=False).reshape(mol.nao_nr(), mol.nao_nr(), mol.nao_nr(), mol.nao_nr())
        import chemps2
        Energy, OneDM = chemps2.solve( ENUCL, OEI, OEI, TEI, mol.nao_nr(), mol.nelectron, mol.nao_nr(), 0.0, False )
        print("bl =", bondlength," and energy =", Energy)
        
    else:
        my_ints = local_integrals.LocalIntegrals( mf, list(range( mol.nao_nr())), 'meta_lowdin' )
        my_ints.molden( 'hydrogen-loc.molden' )
        my_ints.TI_OK = True # Only s functions

        # Build fragments with the helper: 2 consecutive atoms per impurity
        atoms_per_imp = 2
        atom_groups = [ list(range(i, i+atoms_per_imp)) for i in range(0, nat, atoms_per_imp) ]
        impurity_clusters = make_fragments( mol, my_ints, atom_groups )
        is_translation_invariant = True # OK because only s-functions and meta-lowdin

        sc_method = 'LSTSQ'
        print("Start FCI dmet")
        dmet_fci = dmet.DMET( my_ints, impurity_clusters, is_translation_invariant,
                             method='FCI', sc_method=sc_method, do_det=False )
        e_fci = dmet_fci.selfconsistent()
        print("FCI dmet Energy =", e_fci)

        print("\nStart DMRG dmet")
        dmet_dmrg = dmet.DMET( my_ints, impurity_clusters, is_translation_invariant,
                              method='DMRG', sc_method=sc_method, do_det=False )
        e_dmrg = dmet_dmrg.selfconsistent()
        print("DMRG dmet Energy =", e_dmrg)

        print( "\nDifference between FCI and DMRG: %e" % abs(e_fci - e_dmrg) )
        Energy = e_fci
        print("bl =", bondlength," and energy =", Energy)
        energies.append(Energy)

print("Bondlengths =", bondlengths)
print("Energies =", energies)
