"""
solvers/dft.py
==============
PrismDMET impurity solver: DFT (RKS / UKS / ROKS).

Supported method keys
---------------------
  'RKS'   -- Restricted Kohn-Sham (closed-shell, even electron count)
  'UKS'   -- Unrestricted Kohn-Sham (open- or closed-shell)
  'ROKS'  -- Restricted Open-Shell Kohn-Sham (high-spin open-shell)

The XC functional is set via task['xc'] (default 'pbe').

Embedding Hamiltonian convention
---------------------------------
The DMET impurity problem is solved purely algebraically: the one-electron
Hamiltonian (fock + chemical potential shift), two-electron integrals, and
overlap are all supplied as dense matrices.  PySCF's DFT classes require a
real-space numerical grid to evaluate the XC potential.  We satisfy this by
constructing a minimal dummy molecule (a single H atom) whose sole purpose is
to anchor a grid -- the actual density is then evaluated in that grid using
the embedding MOs.  The dummy basis has exactly `norb` s-type functions so
that the AO dimension matches the embedding space throughout.

Energy partitioning
-------------------
The half-projector formula from rhf.py is used with the KS effective
potential (J + Vxc) in place of the HF exchange.
"""

import numpy as np
from pyscf import ao2mo, gto, dft as pyscf_dft


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _embedding_mol(norb, nel, spin=0):
    """Build a minimal dummy Mole for the embedding space.

    A single H atom provides the coordinate anchor for the numerical grid.
    The basis is padded to exactly `norb` s-type functions so that the AO
    dimension matches the size of the embedding Hamiltonian.
    """
    mol = gto.Mole()
    mol.atom  = [['H', (0, 0, 0)]]
    mol.basis = {'H': [[0, [1.0, 1.0]] for _ in range(norb)]}
    mol.nelectron    = nel
    mol.spin         = spin
    mol.incore_anyway = True
    mol.build(verbose=0)
    return mol


# ---------------------------------------------------------------------------
# Closed-shell: RKS
# ---------------------------------------------------------------------------

def solve_rks(const, oei, fock, tei, norb, nel, nimp, dm_guess,
              xc='pbe', chempot_imp=0.0):
    """Solve the embedding Hamiltonian at the RKS level.

    Parameters
    ----------
    const : float
        Constant energy offset (core + nuclear repulsion partition).
    oei : ndarray, shape (norb, norb)
        One-electron integrals in the embedding basis.
    fock : ndarray, shape (norb, norb)
        Fock matrix in the embedding basis.
    tei : ndarray, shape (norb, norb, norb, norb)
        Two-electron integrals in the embedding basis.
    norb : int
        Number of orbitals in the embedding space.
    nel : int
        Number of electrons (must be even for RKS).
    nimp : int
        Number of impurity orbitals (energy projection cutoff).
    dm_guess : ndarray or None
        Initial density matrix guess.
    xc : str
        XC functional string accepted by PySCF (e.g. 'pbe', 'b3lyp').
    chempot_imp : float
        Chemical potential shift applied to impurity diagonal elements.

    Returns
    -------
    energy : float
    rdm1 : ndarray, shape (norb, norb)
    dft_res : dict  -- keys: 'mo_energy', 'mo_occ', 'mo_coeff'
    """
    assert nel % 2 == 0, 'RKS requires an even number of electrons.'

    h1 = fock.copy()
    if chempot_imp != 0.0:
        for i in range(nimp):
            h1[i, i] -= chempot_imp

    mol = _embedding_mol(norb, nel, spin=0)
    mf  = pyscf_dft.RKS(mol)
    mf.xc        = xc
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp  = lambda *args: np.eye(norb)
    mf._eri      = ao2mo.restore(8, tei, norb)
    # Supply a guess if none is provided; PySCF's minao guess fails on the
    # degenerate dummy basis (all exponents identical).
    if dm_guess is None:
        dm_guess = np.diag([nel / norb] * norb)
    mf.scf(dm_guess)
    if not mf.converged:
        mf = mf.newton()
        mf.scf(mf.make_rdm1())

    rdm1 = mf.make_rdm1()
    JK   = mf.get_veff(None, dm=rdm1)

    energy = (
        const
        + 0.25 * np.einsum('ji,ij->', rdm1[:, :nimp], fock[:nimp, :] + oei[:nimp, :])
        + 0.25 * np.einsum('ji,ij->', rdm1[:nimp, :], fock[:, :nimp] + oei[:, :nimp])
        + 0.25 * np.einsum('ji,ij->', rdm1[:, :nimp], JK[:nimp, :])
        + 0.25 * np.einsum('ji,ij->', rdm1[:nimp, :], JK[:, :nimp])
    )

    dft_res = {'mo_energy': mf.mo_energy, 'mo_occ': mf.mo_occ, 'mo_coeff': mf.mo_coeff}
    return energy, rdm1, dft_res


# ---------------------------------------------------------------------------
# Unrestricted: UKS
# ---------------------------------------------------------------------------

def solve_uks(const, oei, fock, tei, norb, nel, nimp, dm_guess,
              xc='pbe', chempot_imp=0.0,
              spin_polarized=False, chempot_imp_beta=None):
    """Solve the embedding Hamiltonian at the UKS level.

    Parameters
    ----------
    spin_polarized : bool
        If True, apply independent alpha/beta chemical potentials and return
        separate alpha and beta RDMs for spin-polarized DMET.
        If False (default), return the spin-summed total RDM.
    chempot_imp_beta : float or None
        Independent beta chemical potential. Used only when spin_polarized=True.
        Defaults to chempot_imp if not provided.

    Returns
    -------
    energy : float
    rdm1 or (rdm1_alpha, rdm1_beta) : ndarray
    dft_res : dict -- keys: 'mo_energy', 'mo_occ', 'mo_coeff'
    """
    spin = nel % 2

    h1_a = fock.copy()
    if chempot_imp != 0.0:
        for i in range(nimp):
            h1_a[i, i] -= chempot_imp

    h1_b = h1_a.copy()
    if spin_polarized and chempot_imp_beta is not None:
        h1_b = fock.copy()
        for i in range(nimp):
            h1_b[i, i] -= chempot_imp_beta

    mol = _embedding_mol(norb, nel, spin=spin)
    mf  = pyscf_dft.UKS(mol)
    mf.xc       = xc
    # get_hcore must return a 2D (spin-free) matrix for UKS energy_elec.
    # The chemical potential shifts are equal for both spins here.
    mf.get_hcore = lambda *args: h1_a
    mf.get_ovlp  = lambda *args: np.eye(norb)
    mf._eri      = ao2mo.restore(8, tei, norb)
    # For spin-polarized DMET, inject a differential beta shift into the
    # Fock build so the two channels see different effective potentials.
    if spin_polarized and chempot_imp_beta is not None:
        _shift = np.diag([chempot_imp - chempot_imp_beta] * nimp +
                         [0.0] * (norb - nimp))
        _base_get_fock = mf.get_fock
        def _get_fock_spinpol(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                              diis=None, diis_start_cycle=None,
                              level_shift_factor=None, damp_factor=None):
            f_a, f_b = _base_get_fock(h1e, s1e, vhf, dm, cycle, diis,
                                      diis_start_cycle, level_shift_factor,
                                      damp_factor)
            return np.array([f_a, f_b - _shift])
        mf.get_fock = _get_fock_spinpol
    if dm_guess is None:
        occ = nel / (2 * norb)
        dm0 = np.diag([occ] * norb)
        dm_guess = np.array([dm0, dm0])
    mf.scf(dm_guess)
    if not mf.converged:
        mf = mf.newton()
        mf.scf(mf.make_rdm1())

    rdm1_a, rdm1_b = mf.make_rdm1()
    JK_a, JK_b    = mf.get_veff(None, dm=mf.make_rdm1())
    rdm1_tot = rdm1_a + rdm1_b
    JK_tot   = JK_a  + JK_b

    energy = (
        const
        + 0.25 * np.einsum('ji,ij->', rdm1_tot[:, :nimp], fock[:nimp, :] + oei[:nimp, :])
        + 0.25 * np.einsum('ji,ij->', rdm1_tot[:nimp, :], fock[:, :nimp] + oei[:, :nimp])
        + 0.25 * np.einsum('ji,ij->', rdm1_tot[:, :nimp], JK_tot[:nimp, :])
        + 0.25 * np.einsum('ji,ij->', rdm1_tot[:nimp, :], JK_tot[:, :nimp])
    )

    dft_res = {'mo_energy': mf.mo_energy, 'mo_occ': mf.mo_occ, 'mo_coeff': mf.mo_coeff}

    if spin_polarized:
        return energy, rdm1_a, rdm1_b, dft_res
    return energy, rdm1_tot, dft_res


# ---------------------------------------------------------------------------
# Restricted open-shell: ROKS
# ---------------------------------------------------------------------------

def solve_roks(const, oei, fock, tei, norb, nel, nimp, dm_guess,
               xc='pbe', chempot_imp=0.0):
    """Solve the embedding Hamiltonian at the ROKS level.

    Returns
    -------
    energy : float
    rdm1 : ndarray, shape (norb, norb)  -- spin-summed total RDM
    dft_res : dict -- keys: 'mo_energy', 'mo_occ', 'mo_coeff'
    """
    spin = nel % 2

    h1 = fock.copy()
    if chempot_imp != 0.0:
        for i in range(nimp):
            h1[i, i] -= chempot_imp

    mol = _embedding_mol(norb, nel, spin=spin)
    mf  = pyscf_dft.ROKS(mol)
    mf.xc        = xc
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp  = lambda *args: np.eye(norb)
    mf._eri      = ao2mo.restore(8, tei, norb)
    if dm_guess is None:
        occ = nel / (2 * norb)
        dm0 = np.diag([occ] * norb)
        dm_guess = np.array([dm0, dm0])
    mf.scf(dm_guess)
    if not mf.converged:
        mf = mf.newton()
        mf.scf(mf.make_rdm1())

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

    dft_res = {'mo_energy': mf.mo_energy, 'mo_occ': mf.mo_occ, 'mo_coeff': mf.mo_coeff}
    return energy, rdm1, dft_res


# ---------------------------------------------------------------------------
# SolverDispatcher entry point
# ---------------------------------------------------------------------------

def execute(task):
    """SolverDispatcher-compatible entry point for DFT solvers.

    Routes to solve_rks, solve_uks, or solve_roks based on task['method'].

    Relevant task keys
    ------------------
    method           : 'RKS', 'UKS', or 'ROKS'
    xc               : XC functional string (default 'pbe')
    spin_polarized   : bool -- UKS only; enables independent alpha/beta mu
    chempot_imp_beta : float -- independent beta chemical potential (UKS + spin_polarized)
    """
    method         = task['method']
    xc             = task.get('xc', 'pbe')
    spin_polarized = task.get('spin_polarized', False)

    common = dict(
        const       = task['const'],
        oei         = task['dmet_oei'],
        fock        = task['dmet_fock'],
        tei         = task['dmet_tei'],
        norb        = task['norb'],
        nel         = task['nel'],
        nimp        = task['nimp'],
        dm_guess    = task.get('dm_guess_rhf'),
        xc          = xc,
        chempot_imp = task.get('chempot_imp', 0.0),
    )

    if method == 'RKS':
        energy, rdm1, dft_res = solve_rks(**common)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1, 'dft_res': dft_res}

    elif method == 'UKS':
        if spin_polarized:
            energy, rdm_a, rdm_b, dft_res = solve_uks(
                **common,
                spin_polarized   = True,
                chempot_imp_beta = task.get('chempot_imp_beta'),
            )
            return {
                'counter'   : task['counter'],
                'energy'    : energy,
                'rdm1'      : rdm_a + rdm_b,
                'rdm1_alpha': rdm_a,
                'rdm1_beta' : rdm_b,
                'dft_res'   : dft_res,
            }
        else:
            energy, rdm1, dft_res = solve_uks(**common, spin_polarized=False)
            return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1, 'dft_res': dft_res}

    elif method == 'ROKS':
        energy, rdm1, dft_res = solve_roks(**common)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1, 'dft_res': dft_res}

    else:
        raise ValueError(
            f"dft.execute: unexpected method='{method}'. "
            f"Expected 'RKS', 'UKS', or 'ROKS'."
        )
