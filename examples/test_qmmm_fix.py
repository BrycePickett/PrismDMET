import numpy as np
from pyscf import gto, scf, qmmm
import sys
sys.path.insert(0, '/users/PAS1583/bpickett/programs/PrismDMET/src')
import local_integrals

# 1. QM/MM test
mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='sto3g')
mf = scf.RHF(mol)
mf = qmmm.mm_charge(mf, np.array([[0,0,2.0]]), np.array([-1.0]))
mf.scf()

li = local_integrals.localintegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
print('QM/MM activeCONST:', li.activeCONST)
print('QM/MM the_mf.energy_nuc:', mf.energy_nuc())
assert abs(li.activeCONST - mf.energy_nuc()) < 1e-8, "CONST did not match mf.energy_nuc()"

# 2. ECP test
mol2 = gto.M(atom='H 0 0 0; Na 0 0 1.5', basis='lanl2dz', ecp={'Na':'lanl2dz'})
mf2 = scf.RHF(mol2).run()
li2 = local_integrals.localintegrals(mf2, list(range(mol2.nao_nr())), 'meta_lowdin')
print('ECP fullFOCKao norm:', np.linalg.norm(li2.fullFOCKao))
print('Tests passed.')
