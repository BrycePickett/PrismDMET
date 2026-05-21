"""PrismDMET impurity solver: DFT (RKS / UKS / ROKS)."""

import numpy as np
from pyscf import gto, dft as pyscf_dft


def _reconstruct_mol(mol_dumps):
    return gto.loads(mol_dumps)


def _build_hcore_ao(mol, ao2loc, loc_2_dmet, fock_emb, oei_emb, chempot_imp, nimp_emb,
                    mm_coords=None, mm_charges=None):
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
    rdm1 = rdm1_emb[0] + rdm1_emb[1] if rdm1_emb.ndim == 3 else rdm1_emb
    return float(np.trace(rdm1[:nimp, :nimp])) / nel_total


def _fragment_local_mos(fock_canonical_ao, C_emb, nocc, mo_occ_value):
    fock_emb = C_emb.T @ fock_canonical_ao @ C_emb
    fock_emb = 0.5 * (fock_emb + fock_emb.T)
    mo_energy, eigvecs = np.linalg.eigh(fock_emb)
    mo_coeff_ao = C_emb @ eigvecs
    mo_occ = np.zeros(mo_energy.shape[0])
    mo_occ[:nocc] = mo_occ_value
    return mo_energy, mo_coeff_ao, mo_occ


def _oei_correction(rdm1_emb, oei, fock, ao2loc, loc_2_dmet, nimp):
    diff = oei - fock
    if np.max(np.abs(diff)) < 1e-14:
        return 0.0
    rdm1 = rdm1_emb[0] + rdm1_emb[1] if rdm1_emb.ndim == 3 else rdm1_emb
    return 0.5 * (
        np.einsum('ij,ji->', rdm1[:, :nimp], diff[:nimp, :]) +
        np.einsum('ij,ji->', rdm1[:nimp, :], diff[:, :nimp])
    )


def solve_rks(const, oei, fock, tei, norb, nel, nimp, dm_guess,
              xc='pbe', chempot_imp=0.0,
              mol=None, ao2loc=None, loc_2_dmet=None,
              mm_coords=None, mm_charges=None, nel_total=None,
              global_spin=0, level_shift=0.0,
              use_density_fit=False, df_auxbasis=None,
              fock_canonical_alpha=None, fock_canonical_beta=None):
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


def solve_uks(const, oei, fock, tei, norb, nel, nimp, dm_guess,
              xc='pbe', chempot_imp=0.0,
              spin_polarized=False, chempot_imp_beta=None,
              mol=None, ao2loc=None, loc_2_dmet=None,
              mm_coords=None, mm_charges=None, nel_total=None,
              global_spin=0, level_shift=0.0,
              use_density_fit=False, df_auxbasis=None,
              fock_canonical_alpha=None, fock_canonical_beta=None):
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
    _has_canonical_guess = isinstance(dm_guess, np.ndarray) and dm_guess.ndim == 3
    _effective_shift = max(level_shift, 0.2) if _has_canonical_guess else level_shift
    if _effective_shift != 0.0:
        mf.level_shift = _effective_shift
    if use_density_fit:
        mf = mf.density_fit(auxbasis=df_auxbasis)
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


def solve_roks(const, oei, fock, tei, norb, nel, nimp, dm_guess,
               xc='pbe', chempot_imp=0.0,
               mol=None, ao2loc=None, loc_2_dmet=None,
               mm_coords=None, mm_charges=None, nel_total=None,
               global_spin=0, level_shift=0.0,
               use_density_fit=False, df_auxbasis=None,
               fock_canonical_alpha=None, fock_canonical_beta=None):
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


def execute(task):
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
