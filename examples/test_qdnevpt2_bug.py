import sys
sys.path.insert(0, '/users/PAS1583/bpickett/programs/PrismDMET/src')
import numpy as np
from pyscf import gto, scf, mcscf
import dmet

# Setup simple chain
mol = gto.M(atom='H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3', basis='sto3g')
mf = scf.RHF(mol).run()

# The canonical active space (2e, 2o)
mc = mcscf.CASSCF(mf, 2, 2)
mc.kernel()
canonical_active_mo = mc.mo_coeff[:, mc.ncore : mc.ncore+mc.ncas]

# DMET active space (impurity on H0, bath on rest)
# If we run DMET, does it pick the same space? No, DMET localizes to H0.
print("Canonical active MOs (HOMO/LUMO) weight on H0:")
print(np.sum(canonical_active_mo[0]**2)) # H0 s-orbital contribution

# What if we run DMET with QD-NEVPT2?
# Since QD-NEVPT2 runs mcscf.CASSCF(mf, 2, 2) without an mo_guess, it will just reproduce the canonical CASSCF.
# So the defect (H0) is not isolated!
