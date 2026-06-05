'''Utility functions for QC-dmet active-space solvers. Adapted from mrh.my_dmet.pyscf_casscf.'''

import numpy as np


def select_cas_orbitals(mc, cas_select, ncas, nimp, norb,
                        ao2eo=None, ao_mol_dumps=None, ao_labels=None):
    '''Pick the CAS active orbitals and reorder mc.mo_coeff so they occupy the active window.

    cas_select:
      'energy'       - no reordering (default PySCF energy ordering).
      'impurity'     - rank frontier MOs by impurity localization weight and take the ncas largest.
      'ao_character' - mrh-style: rank MOs by projection onto ao_labels and take the ncas largest.

    Returns the selected orbital index list (1-based PySCF convention via base=0 sort_mo),
    or None when cas_select == 'energy'.
    '''
    if cas_select == 'energy':
        return None

    C = mc.mo_coeff
    if C.ndim == 3:
        C = C[0] + C[1]

    if cas_select == 'impurity':
        weights = np.sum(np.abs(C[:nimp, :]) ** 2, axis=0)
        frontier = range(mc.ncore, norb)
    elif cas_select == 'ao_character':
        weights = _ao_character_weights(C, ao2eo, ao_mol_dumps, ao_labels)
        frontier = range(norb)
    else:
        raise ValueError(f"select_cas_orbitals: unknown cas_select='{cas_select}'")

    selected = sorted(sorted(frontier, key=lambda i: -weights[i])[:ncas])
    mc.mo_coeff = mc.sort_mo(selected, base=0)
    return selected


def _ao_character_weights(emb_mo, ao2eo, ao_mol_dumps, ao_labels):
    '''Projection of each embedded MO onto the named AO labels (mrh getorbindex / mo_comps style).

    Uses plain Lowdin orthogonalization (pre_orth_ao=None) instead of pyscf's mo_comps, whose
    default meta-lowdin ANO reference fails on GTH/ECP/ghost-atom systems such as Cu2O.
    '''
    from pyscf import gto
    from pyscf.lo.orth import lowdin

    ao_mol = gto.loads(ao_mol_dumps)
    ao_mo  = ao2eo @ emb_mo                       # embedded MOs in AO basis
    s      = ao_mol.intor_symmetric('int1e_ovlp')
    idx    = ao_mol.search_ao_label(ao_labels)
    if len(idx) == 0:
        raise ValueError(f"ao_character selection: no AOs match labels {ao_labels}")
    c_orth = lowdin(s)
    mo1    = c_orth[:, idx].T @ s @ ao_mo
    return np.einsum('ki,ki->i', mo1, mo1)


def project_amo_manually(old_mo_coeff, ncas, ncore, new_fock, norb):
    '''Project old CASSCF active MOs onto the current embedding basis. Returns (new_mo, fidelity); fidelity near 1 means active space survived intact.'''
    assert old_mo_coeff.shape == (norb, norb), \
        f"project_amo_manually: expected old_mo_coeff shape ({norb},{norb}), got {old_mo_coeff.shape}"
    nocc = ncore + ncas

    old_amo = old_mo_coeff[:, ncore:nocc]

    proj = old_amo @ old_amo.T
    evals, evecs = np.linalg.eigh(proj)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    new_amo = evecs[:, :ncas].copy()
    new_imo = evecs[:, ncas:].copy()
    fidelity = evals[:ncas].copy()

    # Align sign of each new AMO with the old to avoid CI vector phase flips.
    overlap = new_amo.T @ old_amo
    for i in range(ncas):
        if overlap[i, i] < 0:
            new_amo[:, i] *= -1

    fock_imo = new_imo.T @ new_fock @ new_imo
    imo_evals, imo_evecs = np.linalg.eigh(fock_imo)
    new_imo = new_imo @ imo_evecs
    new_cmo = new_imo[:, :ncore]
    new_vmo = new_imo[:, ncore:]

    new_mo = np.concatenate([new_cmo, new_amo, new_vmo], axis=1)

    print(f"qcsolver_utils::project_amo : fidelity = {fidelity}")
    return new_mo, fidelity


def fix_casscf_for_nonsinglet_env(mc, h1e_s):
    '''Wrap a CASSCF object to inject spin-dependent 1e potential h1e_s = 0.5*(h_alpha - h_beta) (mrh-style).'''
    from pyscf import lib

    if h1e_s is None or np.all(np.abs(h1e_s) < 1e-8):
        return mc

    # Initial projection into the active space
    amo = mc.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
    amoH = amo.conj().T
    h1e_s_amo  = amoH @ h1e_s @ amo
    h1e_s_amou = h1e_s_amo.copy()
    last_cached_sdm = np.zeros_like(h1e_s_amo)

    class FixedFCI(mc.fcisolver.__class__):

        def __init__(self, base_fci):
            self.__dict__.update(base_fci.__dict__)

        def kernel(self, h1e, eri, norb, nelec, ci0=None, **kwargs):
            if np.asarray(h1e).ndim == 2:
                h1e = [h1e + h1e_s_amo, h1e - h1e_s_amo]
            return super().kernel(h1e, eri, norb, nelec, ci0=ci0, **kwargs)

        def make_rdm12(self, ci, ncas, nelecas, link_index=None):
            dm1, dm2 = super().make_rdm12(ci, ncas, nelecas, link_index=link_index)
            dm1a, dm1b = self.make_rdm1s(ci, ncas, nelecas, link_index=link_index)
            dm1 = lib.tag_array(dm1, sdm=dm1a - dm1b)
            last_cached_sdm[:, :] = dm1.sdm[:, :]
            return dm1, dm2

    class FixedCASSCF(mc.__class__):

        def __init__(self, base_mc):
            self.__dict__.update(base_mc.__dict__)
            self.fcisolver = FixedFCI(base_mc.fcisolver)

        def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
            _amo  = mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
            _amoH = _amo.conj().T
            h1e_s_amo[:, :] = _amoH @ h1e_s @ _amo
            return super().casci(mo_coeff, ci0=ci0, eris=eris,
                                 verbose=verbose, envs=envs)

        def update_casdm(self, mo, u, fcivec, e_cas, eris, envs={}):
            amou  = mo @ u[:, self.ncore:self.ncore + self.ncas]
            amouH = amou.conj().T
            h1e_s_amou[:, :] = amouH @ h1e_s @ amou
            return super().update_casdm(mo, u, fcivec, e_cas, eris, envs=envs)

        def solve_approx_ci(self, h1, h2, ci0, ecore, e_cas, envs):
            h1 = np.stack([h1, h1e_s_amou], axis=0)
            return super().solve_approx_ci(h1, h2, ci0, ecore, e_cas, envs)

        def gen_g_hop(self, mo, u, casdm1, casdm2, eris):
            g_orb, gorb_update, h_op, h_diag = super().gen_g_hop(
                mo, u, casdm1, casdm2, eris
            )

            ncore = self.ncore
            ncas  = self.ncas
            nocc  = ncore + ncas
            h1e_s_mo = mo.conj().T @ h1e_s @ mo
            sdm_mo = np.zeros_like(h1e_s)
            sdm_mo[ncore:nocc, ncore:nocc] = casdm1.sdm
            sdm_u  = np.copy(sdm_mo)
            sdm_au = sdm_u[ncore:nocc, ncore:nocc]

            # Macrocycle gradient correction
            gen_k = h1e_s_mo @ sdm_mo
            g_orb += self.pack_uniq_var(gen_k - gen_k.T)

            # Microcycle gradient correction
            def my_gorb_update(u, fcivec):
                g_orb_u = gorb_update(u, fcivec)
                sdm_au[:, :] = last_cached_sdm
                uH = u.conj().T
                h1e_s_u = uH @ h1e_s_mo @ u
                gen_k_u = h1e_s_u @ sdm_u
                return g_orb_u + self.pack_uniq_var(gen_k_u - gen_k_u.T)

            # Hessian diagonal correction
            h_diag_s  = np.outer(np.diag(h1e_s_mo), np.diag(sdm_mo))
            h_diag_s -= h1e_s_mo * sdm_mo
            h_diag_s -= np.diag(gen_k)[:, None]
            idx = np.diag_indices_from(h_diag_s)
            h_diag_s[idx] = 0
            h_diag += self.pack_uniq_var(h_diag_s + h_diag_s.T)

            # Hessian-vector product correction
            def my_h_op(x):
                x1 = self.unpack_uniq_var(x)
                hx = h1e_s_mo @ x1 @ sdm_mo
                hx -= (gen_k + gen_k.T) @ x1 / 2
                return h_op(x) + self.pack_uniq_var(hx - hx.T)

            return g_orb, my_gorb_update, my_h_op, h_diag

    return FixedCASSCF(mc)
