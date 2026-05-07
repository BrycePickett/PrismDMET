"""
solvers/uhf.py
==============
PrismDMET impurity solver: open-shell Hartree-Fock (UHF / ROHF).

Supported method keys
---------------------
  'UHF'   — Unrestricted Hartree-Fock
  'ROHF'  — Restricted Open-Shell Hartree-Fock

Spin convention
---------------
* UHF  : returns total RDM (rdm_a + rdm_b) for the standard DMET loop,
         OR separate (rdm_a, rdm_b) when task['spin_polarized'] is True.
* ROHF : returns the spin-summed total RDM (always).

These solvers work the same way as rhf.py but accept odd electron counts
and open-shell spin states.
"""

import numpy as np
from pyscf import ao2mo, gto, scf


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _build_embedding_mol(nel, spin=0):
    mol = gto.Mole()
    mol.build(verbose=0)
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = nel
    mol.spin = spin
    mol.incore_anyway = True
    return mol


# ---------------------------------------------------------------------------
# Unrestricted HF: UHF
# ---------------------------------------------------------------------------

def solve_uhf(const, oei, fock, tei, norb, nel, nimp, dm_guess,
              chempot_imp=0.0, spin_polarized=False,
              chempot_imp_beta=None):
    """
    Solve the embedding problem at the UHF level.

    Parameters
    ----------
    spin_polarized : bool
        If True, return (energy, rdm_alpha, rdm_beta) for independent
        alpha/beta chemical-potential optimization.
        If False (default), return (energy, rdm_total).
    chempot_imp_beta : float or None
        Independent beta chemical potential (spin_polarized=True only).
    """
    spin = nel % 2

    fock_copy_a = fock.copy()
    fock_copy_b = fock.copy()
    if chempot_imp != 0.0:
        for orb in range(nimp):
            fock_copy_a[orb, orb] -= chempot_imp
    if spin_polarized and chempot_imp_beta is not None:
        fock_copy_b = fock.copy()
        for orb in range(nimp):
            fock_copy_b[orb, orb] -= chempot_imp_beta

    mol = _build_embedding_mol(nel, spin=spin)
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: np.array([fock_copy_a, fock_copy_b])
    mf.get_ovlp  = lambda *args: np.eye(norb)
    mf._eri      = ao2mo.restore(8, tei, norb)
    mf.scf(dm_guess)
    if not mf.converged:
        mf = mf.newton()
        mf.scf(mf.make_rdm1())

    rdm1_a, rdm1_b = mf.make_rdm1()   # each (N, N)
    rdm1_tot = rdm1_a + rdm1_b

    JK_a, JK_b = mf.get_veff(None, dm=mf.make_rdm1())
    JK_tot = JK_a + JK_b

    energy = (
        const
        + 0.25 * np.einsum('ji,ij->', rdm1_tot[:, :nimp], fock[:nimp, :] + oei[:nimp, :])
        + 0.25 * np.einsum('ji,ij->', rdm1_tot[:nimp, :], fock[:, :nimp] + oei[:, :nimp])
        + 0.25 * np.einsum('ji,ij->', rdm1_tot[:, :nimp], JK_tot[:nimp, :])
        + 0.25 * np.einsum('ji,ij->', rdm1_tot[:nimp, :], JK_tot[:, :nimp])
    )

    if spin_polarized:
        return energy, rdm1_a, rdm1_b
    return energy, rdm1_tot


# ---------------------------------------------------------------------------
# Restricted Open-Shell HF: ROHF
# ---------------------------------------------------------------------------

def solve_rohf(const, oei, fock, tei, norb, nel, nimp, dm_guess,
               chempot_imp=0.0):
    """Solve the embedding problem at the ROHF level."""
    spin = nel % 2

    fock_copy = fock.copy()
    if chempot_imp != 0.0:
        for orb in range(nimp):
            fock_copy[orb, orb] -= chempot_imp

    mol = _build_embedding_mol(nel, spin=spin)
    mf = scf.ROHF(mol)
    mf.get_hcore = lambda *args: fock_copy
    mf.get_ovlp  = lambda *args: np.eye(norb)
    mf._eri      = ao2mo.restore(8, tei, norb)
    mf.scf(dm_guess)
    if not mf.converged:
        mf = mf.newton()
        mf.scf(mf.make_rdm1())

    # ROHF make_rdm1() returns shape (2, N, N) or (N, N) depending on pyscf version
    rdm1_raw = mf.make_rdm1()
    rdm1 = rdm1_raw[0] + rdm1_raw[1] if rdm1_raw.ndim == 3 else rdm1_raw
    JK   = mf.get_veff(None, dm=rdm1)

    energy = (
        const
        + 0.25 * np.einsum('ji,ij->', rdm1[:, :nimp], fock[:nimp, :] + oei[:nimp, :])
        + 0.25 * np.einsum('ji,ij->', rdm1[:nimp, :], fock[:, :nimp] + oei[:, :nimp])
        + 0.25 * np.einsum('ji,ij->', rdm1[:, :nimp], JK[:nimp, :])
        + 0.25 * np.einsum('ji,ij->', rdm1[:nimp, :], JK[:, :nimp])
    )

    return energy, rdm1


# ---------------------------------------------------------------------------
# SolverDispatcher entry point
# ---------------------------------------------------------------------------

def execute(task):
    """
    SolverDispatcher-compatible wrapper for UHF / ROHF solvers.

    task keys consumed here
    -----------------------
    method           : 'UHF' or 'ROHF'
    spin_polarized   : bool — UHF only
    chempot_imp_beta : float — UHF + spin_polarized only
    """
    method = task['method']
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
        chempot_imp= task.get('chempot_imp', 0.0),
    )

    if method == 'UHF':
        if spin_polarized:
            energy, rdm_a, rdm_b = solve_uhf(
                **common,
                spin_polarized=True,
                chempot_imp_beta=task.get('chempot_imp_beta'),
            )
            return {
                'counter'   : task['counter'],
                'energy'    : energy,
                'rdm1'      : rdm_a + rdm_b,
                'rdm1_alpha': rdm_a,
                'rdm1_beta' : rdm_b,
            }
        else:
            energy, rdm1 = solve_uhf(**common)
            return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}

    elif method == 'ROHF':
        energy, rdm1 = solve_rohf(**common)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}

    else:
        raise ValueError(f"uhf.execute: unexpected method='{method}'. "
                         f"Expected 'UHF' or 'ROHF'.")
