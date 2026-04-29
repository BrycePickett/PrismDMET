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
from pyscf import gto, scf, ao2mo
import numpy as np

DMguess  = None
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
    mf.scf(dm0=DMguess)

    if ( False ):   
        ENUCL = mf.mol.energy_nuc()
        OEI   = np.dot(np.dot(mf.mo_coeff.T, mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')), mf.mo_coeff)
        TEI   = ao2mo.outcore.full_iofree(mol, mf.mo_coeff, compact=False).reshape(mol.nao_nr(), mol.nao_nr(), mol.nao_nr(), mol.nao_nr())
        import chemps2
        Energy, OneDM = chemps2.solve( ENUCL, OEI, OEI, TEI, mol.nao_nr(), mol.nelectron, mol.nao_nr(), 0.0, False )
        print("bl =", bondlength," and energy =", Energy)
        
    else:
        myInts = local_integrals.localintegrals( mf, list(range( mol.nao_nr())), 'meta_lowdin' )
        myInts.molden( 'hydrogen-loc.molden' )
        myInts.TI_OK = True # Only s functions

        # Build fragments with the helper: 2 consecutive atoms per impurity
        atoms_per_imp = 2
        atom_groups = [ list(range(i, i+atoms_per_imp)) for i in range(0, nat, atoms_per_imp) ]
        impurityClusters = make_fragments( mol, myInts, atom_groups )
        isTranslationInvariant = True # OK because only s-functions and meta-lowdin

        SCmethod = 'LSTSQ'
        print("Start FCI DMET")
        dmetFCI = dmet.dmet( myInts, impurityClusters, isTranslationInvariant,
                             method='FCI', SCmethod=SCmethod, doDET=False )
        e_fci = dmetFCI.selfconsistent()
        print("FCI DMET Energy =", e_fci)

        print("\nStart DMRG DMET")
        dmetDMRG = dmet.dmet( myInts, impurityClusters, isTranslationInvariant,
                              method='DMRG', SCmethod=SCmethod, doDET=False )
        e_dmrg = dmetDMRG.selfconsistent()
        print("DMRG DMET Energy =", e_dmrg)

        print( "\nDifference between FCI and DMRG: %e" % abs(e_fci - e_dmrg) )
        Energy = e_fci
        print("bl =", bondlength," and energy =", Energy)
        energies.append(Energy)

print("Bondlengths =", bondlengths)
print("Energies =", energies)
