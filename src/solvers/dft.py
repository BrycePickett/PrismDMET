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

Energy partitioning
-------------------
Unlike Hartree-Fock, the Kohn-Sham effective potential V_KS = V_J + V_xc
cannot be partitioned algebraically: V_xc is the functional derivative of
E_xc[rho], not the energy density. The half-projector formula used in RHF
(where 0.5 * Tr(D * V_HF) = E_J + E_x) does not apply to DFT.

The correct approach is to evaluate the KS total energy on the real molecule
using the embedding density. The solver:

  1. Deserializes the real physical molecule from task['dft_mol_dumps'].
  2. Reconstructs the DMET one-electron Hamiltonian in AO space using the
     ao2loc and loc_2_dmet transformation matrices.
  3. Runs a KS calculation on the real molecule with the DMET-modified
     one-electron Hamiltonian. This ensures that V_xc is evaluated with the
     correct AO basis and real-space grid.
  4. Returns the fragment energy as w_imp * (mf.e_tot - E_nuc), where
     w_imp = Tr(D_imp) / N_el is the fraction of electrons on impurity orbitals.
     This reduces to mf.e_tot - E_nuc exactly when nimp == norb.

The localization correction term accounts for the substitution of the DMET
Fock matrix in place of the bare one-electron integrals:

  E_imp = w_imp * (mf.e_tot - E_nuc)
        + 0.5 * Tr_imp(D_loc * (oei - fock))

For one-shot DMET (oei == fock), the correction vanishes.

Grid and AO integrals
---------------------
By running on the real molecule, the TEI used for J are the AO integrals of
the physical system (not the localized embedding TEI). This is appropriate
because the physical J interaction is what the XC functional is calibrated
against. The embedding TEI enter implicitly through the converged density
from which J and V_xc are derived.
"""

import numpy as np
from pyscf import gto, dft as pyscf_dft


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _reconstruct_mol(mol_dumps):
    """Deserialize a Mole object from its JSON string dump."""
    return gto.loads(mol_dumps)


def _build_hcore_ao(mol, ao2loc, loc_2_dmet, fock_emb, oei_emb, chempot_imp, nimp_emb,
                    mm_coords=None, mm_charges=None):
    """Reconstruct the DMET one-electron Hamiltonian in AO space.

    The AO hcore is:
        h_AO = h_AO_real + C_emb @ (fock_emb - oei_emb) @ C_emb^T

    where C_emb = ao2loc @ loc_2_dmet and h_AO_real includes MM point charges
    when mm_coords and mm_charges are provided.
    """
    from pyscf.scf import hf as pyscf_hf
    h_real = pyscf_hf.get_hcore(mol)
    if mm_coords is not None and mm_charges is not None and len(mm_coords) > 0:
        import pyscf.scf
        import pyscf.qmmm
        h_real = pyscf.qmmm.mm_charge(pyscf.scf.RHF(mol), mm_coords, mm_charges).get_hcore()

    C_emb = ao2loc @ loc_2_dmet  # (nao, norb)
    delta = fock_emb - oei_emb   # zero for one-shot DMET
    h_ao  = h_real + C_emb @ delta @ C_emb.T
    return h_ao


def _impurity_weight(rdm1_emb, ao2loc, loc_2_dmet, mol, nimp, nel_total):
    """Compute the electron fraction residing on impurity orbitals.

    For a single-fragment DMET where nimp == norb, this is 1.0 exactly.
    For multi-fragment DMET, nel_total must be the TOTAL system electron count
    so that the weights across all fragments sum to 1.

    Parameters
    ----------
    rdm1_emb : ndarray, shape (norb, norb) or (2, norb, norb)
        Density matrix in the embedding basis.
    nimp : int
        Number of impurity orbitals.
    nel_total : int
        Total number of electrons in the full system.

    Returns
    -------
    float
        w_imp = Tr(D_imp) / N_el_total
    """
    rdm1 = rdm1_emb[0] + rdm1_emb[1] if rdm1_emb.ndim == 3 else rdm1_emb
    return float(np.trace(rdm1[:nimp, :nimp])) / nel_total


def _fragment_local_mos(fock_canonical_ao, C_emb, nocc, mo_occ_value):
    """Diagonalize the canonical Fock projected into the embedding subspace.

    The embedded SCF runs on the real molecule with a modified hcore and its
    eigenvalues do not represent fragment-local orbital energies. To recover
    the local frontier orbitals (e.g. the defect trap state), project the
    canonical Fock into the impurity+bath subspace and diagonalize.

    Returns ``(mo_energy, mo_coeff_ao, mo_occ)`` with shapes ``(norb_emb,)``,
    ``(nao, norb_emb)``, and ``(norb_emb,)`` respectively. The first ``nocc``
    eigenvalues are marked occupied with ``mo_occ_value`` (2.0 for RKS,
    1.0 for one spin channel of UKS/ROKS).
    """
    fock_emb = C_emb.T @ fock_canonical_ao @ C_emb
    fock_emb = 0.5 * (fock_emb + fock_emb.T)
    mo_energy, eigvecs = np.linalg.eigh(fock_emb)
    mo_coeff_ao = C_emb @ eigvecs
    mo_occ = np.zeros(mo_energy.shape[0])
    mo_occ[:nocc] = mo_occ_value
    return mo_energy, mo_coeff_ao, mo_occ


def _oei_correction(rdm1_emb, oei, fock, ao2loc, loc_2_dmet, nimp):
    """Half-projector correction for the fock-vs-oei substitution.

    For one-shot DMET (oei == fock), this is zero. For self-consistent DMET,
    this term accounts for the correlation potential shift.

    Returns
    -------
    float
    """
    diff = oei - fock
    if np.max(np.abs(diff)) < 1e-14:
        return 0.0
    rdm1 = rdm1_emb[0] + rdm1_emb[1] if rdm1_emb.ndim == 3 else rdm1_emb
    return 0.5 * (
        np.einsum('ij,ji->', rdm1[:, :nimp], diff[:nimp, :]) +
        np.einsum('ij,ji->', rdm1[:nimp, :], diff[:, :nimp])
    )


# ---------------------------------------------------------------------------
# Closed-shell: RKS
# ---------------------------------------------------------------------------

def solve_rks(const, oei, fock, tei, norb, nel, nimp, dm_guess,
              xc='pbe', chempot_imp=0.0,
              mol=None, ao2loc=None, loc_2_dmet=None,
              mm_coords=None, mm_charges=None, nel_total=None,
              global_spin=0, level_shift=0.0,
              use_density_fit=False, df_auxbasis=None,
              fock_canonical_alpha=None, fock_canonical_beta=None):
    """Solve the embedding Hamiltonian at the RKS level.

    Parameters
    ----------
    const : float
        Constant energy offset (nuclear repulsion partition from dmet.py).
    oei : ndarray, shape (norb, norb)
        One-electron integrals in the embedding basis.
    fock : ndarray, shape (norb, norb)
        Fock matrix in the embedding basis.
    tei : ndarray, shape (norb, norb, norb, norb)
        Two-electron integrals in the embedding basis (not used for J/K here).
    norb : int
        Number of embedding orbitals.
    nel : int
        Number of electrons (must be even for RKS).
    nimp : int
        Number of impurity orbitals.
    dm_guess : ndarray or None
        Initial AO density matrix guess.
    xc : str
        XC functional string accepted by PySCF.
    chempot_imp : float
        Chemical potential shift applied to impurity diagonal elements.
    mol : Mole or None
        Real physical molecule. Required for correct XC grid evaluation.
    ao2loc : ndarray or None
        AO-to-localized-orbital matrix.
    loc_2_dmet : ndarray or None
        Localized-to-embedding orbital matrix.

    Returns
    -------
    energy : float
    rdm1 : ndarray, shape (norb, norb)  -- density matrix in embedding basis
    dft_res : dict  -- keys: 'mo_energy', 'mo_occ', 'mo_coeff'
    """
    assert nel % 2 == 0, 'RKS requires an even number of electrons.'

    h_emb = fock.copy()
    if chempot_imp != 0.0:
        for i in range(nimp):
            h_emb[i, i] -= chempot_imp

    h_ao = _build_hcore_ao(mol, ao2loc, loc_2_dmet, h_emb, oei,
                           chempot_imp, nimp,
                           mm_coords=mm_coords, mm_charges=mm_charges)

    mf = pyscf_dft.RKS(mol)
    mf.xc        = xc
    if use_density_fit:
        mf = mf.density_fit(auxbasis=df_auxbasis)
    mf.get_hcore = lambda *args: h_ao
    mf.scf(dm_guess)
    if not mf.converged:
        mf = mf.newton()
        mf.scf(mf.make_rdm1())

    # Transform AO density matrix to embedding basis to recover rdm1_emb.
    S       = mol.intor_symmetric('int1e_ovlp')
    C_emb   = ao2loc @ loc_2_dmet    # (nao, norb)
    C_inv   = C_emb.T @ S            # (norb, nao)  -- left inverse when C_emb^T S C_emb = I
    rdm1_ao = mf.make_rdm1()
    rdm1    = C_inv @ rdm1_ao @ C_inv.T

    w_imp  = _impurity_weight(rdm1, ao2loc, loc_2_dmet, mol, nimp, nel_total or nel)
    e_elec = mf.e_tot - mol.energy_nuc()
    energy = const + w_imp * e_elec + _oei_correction(rdm1, oei, fock,
                                                       ao2loc, loc_2_dmet, nimp)

    if fock_canonical_alpha is not None:
        mo_energy, mo_coeff_ao, mo_occ = _fragment_local_mos(
            fock_canonical_alpha, C_emb, nel // 2, 2.0)
        dft_res = {'mo_energy': mo_energy, 'mo_occ': mo_occ, 'mo_coeff': mo_coeff_ao}
    else:
        dft_res = {'mo_energy': mf.mo_energy, 'mo_occ': mf.mo_occ, 'mo_coeff': mf.mo_coeff}
    return energy, rdm1, dft_res


# ---------------------------------------------------------------------------
# Unrestricted: UKS
# ---------------------------------------------------------------------------

def solve_uks(const, oei, fock, tei, norb, nel, nimp, dm_guess,
              xc='pbe', chempot_imp=0.0,
              spin_polarized=False, chempot_imp_beta=None,
              mol=None, ao2loc=None, loc_2_dmet=None,
              mm_coords=None, mm_charges=None, nel_total=None,
              global_spin=0, level_shift=0.0,
              use_density_fit=False, df_auxbasis=None,
              fock_canonical_alpha=None, fock_canonical_beta=None):
    """Solve the embedding Hamiltonian at the UKS level.

    Parameters
    ----------
    spin_polarized : bool
        If True, apply independent alpha/beta chemical potentials and return
        separate alpha and beta RDMs for spin-polarized DMET.
        If False (default), return the spin-summed total RDM.
    chempot_imp_beta : float or None
        Independent beta chemical potential. Used only when spin_polarized=True.

    Returns
    -------
    energy : float
    rdm1 or (rdm1_alpha, rdm1_beta) : ndarray
    dft_res : dict -- keys: 'mo_energy', 'mo_occ', 'mo_coeff'
    """
    spin = global_spin if (nel % 2 == global_spin % 2) else max(0, global_spin - 1)

    h_emb_a = fock.copy()
    if chempot_imp != 0.0:
        for i in range(nimp):
            h_emb_a[i, i] -= chempot_imp

    h_emb_b = h_emb_a.copy()
    if spin_polarized and chempot_imp_beta is not None:
        h_emb_b = fock.copy()
        for i in range(nimp):
            h_emb_b[i, i] -= chempot_imp_beta

    h_ao_a = _build_hcore_ao(mol, ao2loc, loc_2_dmet, h_emb_a, oei,
                              chempot_imp, nimp,
                              mm_coords=mm_coords, mm_charges=mm_charges)
    h_ao_b = _build_hcore_ao(mol, ao2loc, loc_2_dmet, h_emb_b, oei,
                              chempot_imp_beta if spin_polarized and chempot_imp_beta is not None
                              else chempot_imp, nimp,
                              mm_coords=mm_coords, mm_charges=mm_charges)

    mol_spin = gto.loads(mol.dumps())
    mol_spin.spin = spin
    mol_spin.build(verbose=0)

    mf = pyscf_dft.UKS(mol_spin)
    mf.xc = xc
    # When a canonical DM warm-start is available, apply at least a 0.2 Eh level
    # shift. The DMET embedding Hamiltonian modifies orbital energies enough to
    # flip HOMO/LUMO occupations in early SCF cycles; the shift pins the canonical
    # occupation pattern long enough for convergence to the correct basin.
    _has_canonical_guess = isinstance(dm_guess, np.ndarray) and dm_guess.ndim == 3
    _effective_shift = max(level_shift, 0.2) if _has_canonical_guess else level_shift
    if _effective_shift != 0.0:
        mf.level_shift = _effective_shift
    if use_density_fit:
        mf = mf.density_fit(auxbasis=df_auxbasis)
    # For UKS, get_hcore must return a spin-averaged (2D) hcore for energy_elec.
    # Spin-dependent shifts are applied via get_fock override.
    mf.get_hcore = lambda *args: h_ao_a
    if spin_polarized and chempot_imp_beta is not None and not np.allclose(h_ao_a, h_ao_b):
        _shift_ao = h_ao_a - h_ao_b
        _base_get_fock = mf.get_fock
        def _get_fock_spinpol(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                              diis=None, diis_start_cycle=None,
                              level_shift_factor=None, damp_factor=None):
            f_a, f_b = _base_get_fock(h1e, s1e, vhf, dm, cycle, diis,
                                      diis_start_cycle, level_shift_factor,
                                      damp_factor)
            return np.array([f_a, f_b - _shift_ao])
        mf.get_fock = _get_fock_spinpol
    mf.scf(dm_guess)
    if not mf.converged:
        mf = mf.newton()
        mf.scf(mf.make_rdm1())

    S       = mol_spin.intor_symmetric('int1e_ovlp')
    C_emb   = ao2loc @ loc_2_dmet
    C_inv   = C_emb.T @ S
    rdm1_ao_a, rdm1_ao_b = mf.make_rdm1()
    rdm1_a  = C_inv @ rdm1_ao_a @ C_inv.T
    rdm1_b  = C_inv @ rdm1_ao_b @ C_inv.T
    rdm1    = rdm1_a + rdm1_b

    w_imp  = _impurity_weight(rdm1, ao2loc, loc_2_dmet, mol_spin, nimp, nel_total or nel)
    e_elec = mf.e_tot - mol_spin.energy_nuc()
    energy = const + w_imp * e_elec + _oei_correction(rdm1, oei, fock,
                                                       ao2loc, loc_2_dmet, nimp)

    if fock_canonical_alpha is not None and fock_canonical_beta is not None:
        nalpha = (nel + spin) // 2
        nbeta  = (nel - spin) // 2
        mo_e_a, mo_c_a, mo_o_a = _fragment_local_mos(
            fock_canonical_alpha, C_emb, nalpha, 1.0)
        mo_e_b, mo_c_b, mo_o_b = _fragment_local_mos(
            fock_canonical_beta, C_emb, nbeta, 1.0)
        dft_res = {
            'mo_energy': np.array([mo_e_a, mo_e_b]),
            'mo_occ'   : np.array([mo_o_a, mo_o_b]),
            'mo_coeff' : np.array([mo_c_a, mo_c_b]),
        }
    else:
        dft_res = {'mo_energy': mf.mo_energy, 'mo_occ': mf.mo_occ, 'mo_coeff': mf.mo_coeff}

    if spin_polarized:
        return energy, rdm1_a, rdm1_b, dft_res
    return energy, rdm1, dft_res


# ---------------------------------------------------------------------------
# Restricted open-shell: ROKS
# ---------------------------------------------------------------------------

def solve_roks(const, oei, fock, tei, norb, nel, nimp, dm_guess,
               xc='pbe', chempot_imp=0.0,
               mol=None, ao2loc=None, loc_2_dmet=None,
               mm_coords=None, mm_charges=None, nel_total=None,
               global_spin=0, level_shift=0.0,
               use_density_fit=False, df_auxbasis=None,
               fock_canonical_alpha=None, fock_canonical_beta=None):
    """Solve the embedding Hamiltonian at the ROKS level.

    Returns
    -------
    energy : float
    rdm1 : ndarray, shape (norb, norb)  -- spin-summed density matrix in embedding basis
    dft_res : dict -- keys: 'mo_energy', 'mo_occ', 'mo_coeff'
    """
    spin = global_spin if (nel % 2 == global_spin % 2) else max(0, global_spin - 1)

    h_emb = fock.copy()
    if chempot_imp != 0.0:
        for i in range(nimp):
            h_emb[i, i] -= chempot_imp

    h_ao = _build_hcore_ao(mol, ao2loc, loc_2_dmet, h_emb, oei,
                           chempot_imp, nimp,
                           mm_coords=mm_coords, mm_charges=mm_charges)

    mol_spin = gto.loads(mol.dumps())
    mol_spin.spin = spin
    mol_spin.build(verbose=0)

    mf = pyscf_dft.ROKS(mol_spin)
    mf.xc        = xc
    if level_shift != 0.0:
        mf.level_shift = level_shift
    if use_density_fit:
        mf = mf.density_fit(auxbasis=df_auxbasis)
    mf.get_hcore = lambda *args: h_ao
    mf.scf(dm_guess)
    if not mf.converged:
        mf = mf.newton()
        mf.scf(mf.make_rdm1())

    S       = mol_spin.intor_symmetric('int1e_ovlp')
    C_emb   = ao2loc @ loc_2_dmet
    C_inv   = C_emb.T @ S
    rdm1_ao_raw = mf.make_rdm1()
    if rdm1_ao_raw.ndim == 3:
        rdm1_ao = rdm1_ao_raw[0] + rdm1_ao_raw[1]
    else:
        rdm1_ao = rdm1_ao_raw
    rdm1 = C_inv @ rdm1_ao @ C_inv.T

    w_imp  = _impurity_weight(rdm1, ao2loc, loc_2_dmet, mol_spin, nimp, nel_total or nel)
    e_elec = mf.e_tot - mol_spin.energy_nuc()
    energy = const + w_imp * e_elec + _oei_correction(rdm1, oei, fock,
                                                       ao2loc, loc_2_dmet, nimp)

    if fock_canonical_alpha is not None:
        nalpha = (nel + spin) // 2
        nbeta  = (nel - spin) // 2
        fock_use = fock_canonical_alpha if fock_canonical_beta is None else \
                   0.5 * (fock_canonical_alpha + fock_canonical_beta)
        mo_energy, mo_coeff_ao, _ = _fragment_local_mos(fock_use, C_emb, nalpha, 1.0)
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[:nbeta]         = 2.0
        mo_occ[nbeta:nalpha]   = 1.0
        dft_res = {'mo_energy': mo_energy, 'mo_occ': mo_occ, 'mo_coeff': mo_coeff_ao}
    else:
        dft_res = {'mo_energy': mf.mo_energy, 'mo_occ': mf.mo_occ, 'mo_coeff': mf.mo_coeff}
    return energy, rdm1, dft_res


# ---------------------------------------------------------------------------
# SolverDispatcher entry point
# ---------------------------------------------------------------------------

def execute(task):
    """SolverDispatcher-compatible entry point for DFT solvers.

    Routes to solve_rks, solve_uks, or solve_roks based on task['method'].

    Required task keys
    ------------------
    method           : 'RKS', 'UKS', or 'ROKS'
    dft_mol_dumps    : JSON string from mol.dumps() for the real physical molecule
    ao2loc           : ndarray -- AO-to-localized-orbital transformation
    loc_2_dmet       : ndarray -- localized-to-embedding-orbital transformation

    Optional task keys
    ------------------
    xc               : XC functional string (default 'pbe')
    spin_polarized   : bool -- UKS only; enables independent alpha/beta mu
    chempot_imp_beta : float -- independent beta chemical potential (UKS + spin_polarized)
    """
    method         = task['method']
    xc             = task.get('xc', 'pbe')
    spin_polarized = task.get('spin_polarized', False)

    mol = _reconstruct_mol(task['dft_mol_dumps'])

    common = dict(
        const           = task['const'],
        oei             = task['dmet_oei'],
        fock            = task['dmet_fock'],
        tei             = task['dmet_tei'],
        norb            = task['norb'],
        nel             = task['nel'],
        nimp            = task['nimp'],
        dm_guess        = task.get('dm_guess_rhf'),
        xc              = xc,
        chempot_imp     = task.get('chempot_imp', 0.0),
        mol             = mol,
        ao2loc          = task['ao2loc'],
        loc_2_dmet      = task['loc_2_dmet'],
        mm_coords       = task.get('mm_coords'),
        mm_charges      = task.get('mm_charges'),
        nel_total       = task.get('nel_total'),
        global_spin     = task.get('spin', 0),
        level_shift     = task.get('level_shift', 0.0),
        use_density_fit = task.get('use_density_fit', False),
        df_auxbasis     = task.get('df_auxbasis', None),
        fock_canonical_alpha = task.get('fock_canonical_alpha'),
        fock_canonical_beta  = task.get('fock_canonical_beta'),
    )

    if method == 'RKS':
        energy, rdm1, dft_res = solve_rks(**common)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1, 'dft_res': dft_res}

    elif method == 'UKS':
        _dm_a = task.get('dm_canonical_alpha')
        _dm_b = task.get('dm_canonical_beta')
        if _dm_a is not None and _dm_b is not None:
            common['dm_guess'] = np.array([_dm_a, _dm_b])
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
