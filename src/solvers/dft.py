"""
solvers/dft.py
==============
PrismDMET impurity solver: DFT (RKS / UKS / ROKS).

Supported method keys
---------------------
  'RKS'   — Restricted Kohn-Sham (closed-shell, even nel)
  'UKS'   — Unrestricted Kohn-Sham (spin-polarized)
  'ROKS'  — Restricted Open-Shell Kohn-Sham (high-spin, odd nel)

The exchange-correlation functional is taken from task['xc'] (default 'pbe').

Spin convention
---------------
* RKS : all even-nel systems; returns RDM shape (N, N).
* UKS  : any nel; returns total RDM rdm_a + rdm_b of shape (N, N)
         for the spin-summed DMET loop, OR a tuple (rdm_a, rdm_b)
         of shape (2, N, N) when task['spin_polarized'] is True.
* ROKS : odd-nel high-spin systems; returns RDM shape (N, N).

Energy formula
--------------
Same half-projector formula as rhf.py, using the KS total-energy functional.
"""

import numpy as np
from pyscf import ao2mo, gto, dft as pyscf_dft, scf


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_embedding_mol(nel, spin=0):
    """Construct a dummy PySCF Mole for the embedding space."""
    mol = gto.Mole()
    mol.build(verbose=0)
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = nel
    mol.spin = spin
    mol.incore_anyway = True
    return mol


def _impurity_energy(const, oei, fock, tei, rdm1, rdm2, nimp):
    """Half-projector energy partitioning (same as rhf.py / casscf.py)."""
    if rdm2 is not None:
        # 2-RDM path (for future extension; unused by mean-field solvers)
        return (
            const
            + 0.25 * np.einsum('ij,ij->', rdm1[:, :nimp], fock[:nimp, :] + oei[:nimp, :])
            + 0.25 * np.einsum('ij,ij->', rdm1[:nimp, :], fock[:, :nimp] + oei[:, :nimp])
            + 0.125 * np.einsum('ijkl,ijkl->', rdm2[:nimp, :, :, :], tei[:nimp, :, :, :])
            + 0.125 * np.einsum('ijkl,ijkl->', rdm2[:, :nimp, :, :], tei[:, :nimp, :, :])
            + 0.125 * np.einsum('ijkl,ijkl->', rdm2[:, :, :nimp, :], tei[:, :, :nimp, :])
            + 0.125 * np.einsum('ijkl,ijkl->', rdm2[:, :, :, :nimp], tei[:, :, :, :nimp])
        )
    else:
        # 1-RDM path (standard mean-field, JK evaluated from rdm1)
        from pyscf import ao2mo as _ao2mo
        vj = np.einsum('ijkl,kl->ij', tei, rdm1)
        vk = np.einsum('ijkl,jl->ik', tei, rdm1)
        JK = vj - 0.5 * vk
        return (
            const
            + 0.25 * np.einsum('ij,ij->', rdm1[:, :nimp], fock[:nimp, :] + oei[:nimp, :])
            + 0.25 * np.einsum('ij,ij->', rdm1[:nimp, :], fock[:, :nimp] + oei[:, :nimp])
            + 0.25 * np.einsum('ij,ij->', rdm1[:, :nimp], JK[:nimp, :])
            + 0.25 * np.einsum('ij,ij->', rdm1[:nimp, :], JK[:, :nimp])
        )


# ---------------------------------------------------------------------------
# Closed-shell DFT: RKS
# ---------------------------------------------------------------------------

def solve_rks(const, oei, fock, tei, norb, nel, nimp, dm_guess,
              xc='pbe', chempot_imp=0.0):
    """Solve the embedding problem at the RKS level."""
    assert nel % 2 == 0, "RKS requires an even number of electrons."

    fock_copy = fock.copy()
    if chempot_imp != 0.0:
        for orb in range(nimp):
            fock_copy[orb, orb] -= chempot_imp

    mol = _build_embedding_mol(nel, spin=0)
    mf = pyscf_dft.RKS(mol)
    mf.xc = xc
    mf.get_hcore = lambda *args: fock_copy
    mf.get_ovlp  = lambda *args: np.eye(norb)
    mf._eri      = ao2mo.restore(8, tei, norb)
    mf.scf(dm_guess)
    if not mf.converged:
        mf = mf.newton()
        mf.scf(mf.make_rdm1())

    rdm1 = mf.make_rdm1()          # shape (N, N), factor-2 occupied
    JK   = mf.get_veff(None, dm=rdm1)

    energy = (
        const
        + 0.25 * np.einsum('ji,ij->', rdm1[:, :nimp], fock[:nimp, :] + oei[:nimp, :])
        + 0.25 * np.einsum('ji,ij->', rdm1[:nimp, :], fock[:, :nimp] + oei[:, :nimp])
        + 0.25 * np.einsum('ji,ij->', rdm1[:, :nimp], JK[:nimp, :])
        + 0.25 * np.einsum('ji,ij->', rdm1[:nimp, :], JK[:, :nimp])
    )

    dft_res = {
        'mo_energy': mf.mo_energy,
        'mo_occ'   : mf.mo_occ,
        'mo_coeff' : mf.mo_coeff,
    }
    return energy, rdm1, dft_res


# ---------------------------------------------------------------------------
# Unrestricted DFT: UKS
# ---------------------------------------------------------------------------

def solve_uks(const, oei, fock, tei, norb, nel, nimp, dm_guess,
              xc='pbe', chempot_imp=0.0, spin_polarized=False,
              chempot_imp_beta=None):
    """
    Solve the embedding problem at the UKS level.

    Parameters
    ----------
    spin_polarized : bool
        If True, return (energy, rdm_alpha, rdm_beta) so that the caller can
        perform independent alpha/beta chemical-potential optimization.
        If False (default), return (energy, rdm_total) where
        rdm_total = rdm_alpha + rdm_beta.
    chempot_imp_beta : float or None
        Independent beta chemical potential.  Only used when spin_polarized=True.
        If None, defaults to chempot_imp (spin-symmetric).
    """
    spin = nel % 2   # 0 for even, 1 for odd

    fock_copy = fock.copy()
    if chempot_imp != 0.0:
        for orb in range(nimp):
            fock_copy[orb, orb] -= chempot_imp

    # Build a 2-component fock if spin_polarized with independent beta mu
    fock_copy_beta = fock_copy.copy()
    if spin_polarized and chempot_imp_beta is not None:
        fock_copy_beta = fock.copy()
        for orb in range(nimp):
            fock_copy_beta[orb, orb] -= chempot_imp_beta

    mol = _build_embedding_mol(nel, spin=spin)
    mf = pyscf_dft.UKS(mol)
    mf.xc = xc
    mf.get_hcore = lambda *args: np.array([fock_copy, fock_copy_beta])
    mf.get_ovlp  = lambda *args: np.eye(norb)
    mf._eri      = ao2mo.restore(8, tei, norb)
    mf.scf(dm_guess)
    if not mf.converged:
        mf = mf.newton()
        mf.scf(mf.make_rdm1())

    rdm1_a, rdm1_b = mf.make_rdm1()   # each shape (N, N)
    JK_a, JK_b = mf.get_veff(None, dm=mf.make_rdm1())

    rdm1_tot = rdm1_a + rdm1_b
    JK_tot   = JK_a  + JK_b

    energy = (
        const
        + 0.25 * np.einsum('ji,ij->', rdm1_tot[:, :nimp], fock[:nimp, :] + oei[:nimp, :])
        + 0.25 * np.einsum('ji,ij->', rdm1_tot[:nimp, :], fock[:, :nimp] + oei[:, :nimp])
        + 0.25 * np.einsum('ji,ij->', rdm1_tot[:, :nimp], JK_tot[:nimp, :])
        + 0.25 * np.einsum('ji,ij->', rdm1_tot[:nimp, :], JK_tot[:, :nimp])
    )

    dft_res = {
        'mo_energy': mf.mo_energy,
        'mo_occ'   : mf.mo_occ,
        'mo_coeff' : mf.mo_coeff,
    }

    if spin_polarized:
        return energy, rdm1_a, rdm1_b, dft_res
    else:
        return energy, rdm1_tot, dft_res


# ---------------------------------------------------------------------------
# Restricted Open-Shell DFT: ROKS
# ---------------------------------------------------------------------------

def solve_roks(const, oei, fock, tei, norb, nel, nimp, dm_guess,
               xc='pbe', chempot_imp=0.0):
    """Solve the embedding problem at the ROKS level."""
    spin = nel % 2

    fock_copy = fock.copy()
    if chempot_imp != 0.0:
        for orb in range(nimp):
            fock_copy[orb, orb] -= chempot_imp

    mol = _build_embedding_mol(nel, spin=spin)
    mf = pyscf_dft.ROKS(mol)
    mf.xc = xc
    mf.get_hcore = lambda *args: fock_copy
    mf.get_ovlp  = lambda *args: np.eye(norb)
    mf._eri      = ao2mo.restore(8, tei, norb)
    mf.scf(dm_guess)
    if not mf.converged:
        mf = mf.newton()
        mf.scf(mf.make_rdm1())

    # ROKS returns a 2-component RDM (alpha, beta); sum to get total
    rdm1_full = mf.make_rdm1()   # shape (2, N, N) for open-shell
    rdm1 = rdm1_full[0] + rdm1_full[1] if rdm1_full.ndim == 3 else rdm1_full
    JK   = mf.get_veff(None, dm=rdm1)

    energy = (
        const
        + 0.25 * np.einsum('ji,ij->', rdm1[:, :nimp], fock[:nimp, :] + oei[:nimp, :])
        + 0.25 * np.einsum('ji,ij->', rdm1[:nimp, :], fock[:, :nimp] + oei[:, :nimp])
        + 0.25 * np.einsum('ji,ij->', rdm1[:, :nimp], JK[:nimp, :])
        + 0.25 * np.einsum('ji,ij->', rdm1[:nimp, :], JK[:, :nimp])
    )

    dft_res = {
        'mo_energy': mf.mo_energy,
        'mo_occ'   : mf.mo_occ,
        'mo_coeff' : mf.mo_coeff,
    }
    return energy, rdm1, dft_res


# ---------------------------------------------------------------------------
# SolverDispatcher entry point
# ---------------------------------------------------------------------------

def execute(task):
    """
    SolverDispatcher-compatible wrapper for DFT solvers.

    Routes to solve_rks, solve_uks, or solve_roks based on task['method'].

    task keys consumed here
    -----------------------
    method          : 'RKS', 'UKS', or 'ROKS'
    xc              : str   — XC functional (default 'pbe')
    spin_polarized  : bool  — UKS only; return separate alpha/beta RDMs
    chempot_imp_beta: float — UKS + spin_polarized only; independent beta mu
    """
    method = task['method']
    xc     = task.get('xc', 'pbe')
    spin_polarized = task.get('spin_polarized', False)

    common = dict(
        const      = task['const'],
        oei        = task['dmet_oei'],
        fock       = task['dmet_fock'],
        tei        = task['dmet_tei'],
        norb       = task['norb'],
        nel        = task['nel'],
        nimp       = task['nimp'],
        dm_guess   = task.get('dm_guess_rhf'),
        xc         = xc,
        chempot_imp= task.get('chempot_imp', 0.0),
    )

    if method == 'RKS':
        energy, rdm1, dft_res = solve_rks(**common)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1, 'dft_res': dft_res}

    elif method == 'UKS':
        if spin_polarized:
            energy, rdm_a, rdm_b, dft_res = solve_uks(
                **common,
                spin_polarized=True,
                chempot_imp_beta=task.get('chempot_imp_beta'),
            )
            return {
                'counter'  : task['counter'],
                'energy'   : energy,
                'rdm1'     : rdm_a + rdm_b,   # total for DMET loop
                'rdm1_alpha': rdm_a,
                'rdm1_beta' : rdm_b,
                'dft_res'  : dft_res,
            }
        else:
            energy, rdm1, dft_res = solve_uks(**common, spin_polarized=False)
            return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1, 'dft_res': dft_res}

    elif method == 'ROKS':
        energy, rdm1, dft_res = solve_roks(**common)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1, 'dft_res': dft_res}

    else:
        raise ValueError(f"dft.execute: unexpected method='{method}'. "
                         f"Expected 'RKS', 'UKS', or 'ROKS'.")
