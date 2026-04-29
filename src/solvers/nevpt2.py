'''
    PySCF NEVPT2 solver for QC-DMET.

    Implements strongly-contracted (SC) NEVPT2 using PySCF's mrpt.NEVPT class,
    which is the modern API replacing the deprecated mrpt.NEVPT2 function.

    Supports two modes:
        Single-state (ss)  — CASSCF(root=0) reference, single state NEVPT2.
        Multi-state  (ms)  — SA-CASSCF orbitals + multi-root CASCI re-diagonalisation,
                             then per-state NEVPT2. This is the correct procedure
                             for excited states (see PySCF example mrpt/41-...).

    Like QD-NEVPT2, this solver operates on the REAL physical molecule (mf_real)
    because PySCF's NEVPT2 needs the actual mol/mf/mc objects. It is therefore
    oneshot-DMET only.
'''

import numpy as np
from pyscf import gto, scf, mcscf, mrpt
from utils import silent_stdout, nullcontext

_eV = 27.21138602


def solve(mf_real, ncas, nelecas,
          nstates=1, sa_weights=None,
          root=0,
          casscf_kwargs=None,
          nevpt2_kwargs=None,
          printoutput=True):
    '''
    Run CASSCF + NEVPT2 on the real physical molecule.

    For a single ground-state calculation (nstates=1), the standard
    CASSCF → NEVPT2 pipeline is used.

    For excited states (nstates > 1), the correct multi-step procedure is:
        1. SA-CASSCF with nstates states to get optimised MOs.
        2. Multi-root CASCI with those MOs (nroots = nstates).
        3. Per-state SC-NEVPT2 via mrpt.NEVPT(mc_casci, root=i).

    Parameters
    ----------
    mf_real       : pyscf.scf.hf.RHF  – converged RHF on the real molecule
    ncas          : int  – number of active orbitals
    nelecas       : int  – number of active electrons
    nstates       : int  – number of states (1 = single-state; >1 = multi-state)
    sa_weights    : list of float or None  – SA weights (uniform if None)
    root          : int  – which state to return as ImpurityEnergy (0 = GS)
    casscf_kwargs : dict – extra attributes set on the CASSCF object
    nevpt2_kwargs : dict – extra attributes set on each mrpt.NEVPT object
    printoutput   : bool

    Returns
    -------
    e_tot      : ndarray  – total NEVPT2 energy per state (Ha)
    e_corr     : ndarray  – NEVPT2 correlation energy per state (Ha)
    mc         : mcscf object  – the CASCI/CASSCF object used for NEVPT2
    nevpt_objs : list of mrpt.NEVPT  – one per state
    '''
    casscf_kwargs = casscf_kwargs or {}
    nevpt2_kwargs = nevpt2_kwargs or {}

    ctx = silent_stdout() if not printoutput else nullcontext()

    with ctx:
        if nstates == 1:
            # ---------------------------------------------------------
            # Single-state: CASSCF -> NEVPT2 directly
            # ---------------------------------------------------------
            mc = mcscf.CASSCF(mf_real, ncas, nelecas)
            mc.verbose = 5 if printoutput else 0
            for key, val in casscf_kwargs.items():
                setattr(mc, key, val)
            mc.kernel()

            print(f"\nnevpt2::solve : CASSCF energy = {mc.e_tot:.10f} Ha")

            nevpt_obj = mrpt.NEVPT(mc, root=0)
            nevpt_obj.verbose = 5 if printoutput else 0
            for key, val in nevpt2_kwargs.items():
                setattr(nevpt_obj, key, val)
            e_c = nevpt_obj.kernel()

            e_tot  = np.array([nevpt_obj.e_tot])
            e_corr = np.array([e_c])
            nevpt_objs = [nevpt_obj]

        else:
            # ---------------------------------------------------------
            # Multi-state: SA-CASSCF -> multi-root CASCI -> per-state NEVPT2
            # (following PySCF example mrpt/41-for_state_average.py)
            # ---------------------------------------------------------
            if sa_weights is None:
                sa_weights = [1.0 / nstates] * nstates
            sa_weights = np.array(sa_weights, dtype=float)
            sa_weights /= sa_weights.sum()

            # Step 1: SA-CASSCF for optimised orbitals
            mc_sa = mcscf.CASSCF(mf_real, ncas, nelecas)
            mc_sa = mcscf.state_average_(mc_sa, weights=sa_weights.tolist())
            mc_sa.verbose = 5 if printoutput else 0
            for key, val in casscf_kwargs.items():
                setattr(mc_sa, key, val)
            mc_sa.kernel()
            sa_mo = mc_sa.mo_coeff

            print(f"\nnevpt2::solve : SA-CASSCF ({nstates} states) done.")
            for i, e in enumerate(mc_sa.e_states):
                print(f"  State {i}: {e:.10f} Ha  (weight={sa_weights[i]:.4f})")

            # Step 2: Multi-root CASCI with SA-CASSCF MOs
            mc = mcscf.CASCI(mf_real, ncas, nelecas)
            mc.verbose = 5 if printoutput else 0
            mc.fcisolver.nroots = nstates
            mc.kernel(sa_mo)

            print(f"\nnevpt2::solve : Multi-root CASCI energies:")
            for i, e in enumerate(mc.e_tot):
                print(f"  State {i}: {e:.10f} Ha")

            # Step 3: Per-state NEVPT2
            e_tot  = np.zeros(nstates)
            e_corr = np.zeros(nstates)
            nevpt_objs = []
            for i in range(nstates):
                nevpt_i = mrpt.NEVPT(mc, root=i)
                nevpt_i.verbose = 5 if printoutput else 0
                for key, val in nevpt2_kwargs.items():
                    setattr(nevpt_i, key, val)
                e_c_i = nevpt_i.kernel()
                e_tot[i]  = nevpt_i.e_tot
                e_corr[i] = e_c_i
                nevpt_objs.append(nevpt_i)

        print(f"\nnevpt2::solve : NEVPT2 results:")
        print(f"  {'State':>5}  {'E_tot (Ha)':>16}  {'E_corr (Ha)':>14}  {'ΔE from GS (eV)':>16}")
        print("  " + "-"*56)
        for i, (et, ec) in enumerate(zip(e_tot, e_corr)):
            de_ev = (et - e_tot[0]) * _eV
            print(f"  {i:>5d}  {et:>16.10f}  {ec:>14.10f}  {de_ev:>+16.4f}")

    return e_tot, e_corr, mc, nevpt_objs
