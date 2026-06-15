'''Utility functions for QC-dmet active-space solvers. Adapted from mrh.my_dmet.pyscf_casscf.'''

import numpy as np


def select_cas_orbitals(mc, cas_select, ncas, nimp, norb,
                        ao2eo=None, ao_mol_dumps=None, ao_labels=None):
    '''Pick the CAS active orbitals and reorder mc.mo_coeff so they occupy the active window.

    cas_select:
      'energy'       - no reordering (default PySCF energy ordering).
      'impurity'     - rank frontier MOs by impurity localization weight and take the ncas largest.
      'ao_character' - rank MOs by projection onto ao_labels; falls back to 'impurity' if
                       ao_labels or required parameters are missing.

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
        # Validate all required parameters for ao_character mode
        if ao_labels is None or ao2eo is None or ao_mol_dumps is None:
            print(f"WARNING: cas_select='ao_character' but ao_labels/ao2eo/ao_mol_dumps not fully provided. "
                  f"Falling back to 'impurity' localization selection.")
            weights = np.sum(np.abs(C[:nimp, :]) ** 2, axis=0)
            frontier = range(mc.ncore, norb)
        else:
            try:
                weights = _ao_character_weights(C, ao2eo, ao_mol_dumps, ao_labels)
                frontier = range(norb)
            except (ValueError, KeyError) as e:
                print(f"WARNING: ao_character selection failed ({e}). "
                      f"Falling back to 'impurity' localization selection.")
                weights = np.sum(np.abs(C[:nimp, :]) ** 2, axis=0)
                frontier = range(mc.ncore, norb)
    else:
        raise ValueError(f"select_cas_orbitals: unknown cas_select='{cas_select}'. "
                        f"Valid options: 'energy', 'impurity', 'ao_character'")

    ranked = sorted(frontier, key=lambda i: -weights[i])
    _warn_if_ambiguous_boundary(ranked, weights, ncas, cas_select)
    selected = sorted(ranked[:ncas])
    _scf = getattr(mc, '_scf', None)
    _warn_if_degenerate_boundary(selected, getattr(_scf, 'mo_energy', None))
    mc.mo_coeff = mc.sort_mo(selected, base=0)
    return selected


def _warn_if_ambiguous_boundary(ranked, weights, ncas, cas_select):
    if len(ranked) <= ncas:
        return
    score_in  = weights[ranked[ncas - 1]]
    score_out = weights[ranked[ncas]]
    if score_in < 1e-10:
        return
    gap_rel = (score_in - score_out) / score_in
    if gap_rel < 0.05:
        print(f"WARNING: cas_select='{cas_select}' selection boundary is ambiguous — "
              f"orbital {ranked[ncas-1]} (score={score_in:.4f}) vs orbital {ranked[ncas]} "
              f"(score={score_out:.4f}), gap={gap_rel:.1%}. "
              f"Consider increasing ncas by 1 or verifying the active space manually.")


def _warn_if_degenerate_boundary(selected, mo_energy, tol=1e-3):
    '''Warn when a selected orbital is near-degenerate in energy with an excluded one.

    Splitting a degenerate manifold across the active/inactive boundary leaves the
    CASSCF seed span rotation-ambiguous: an infinitesimal mean-field perturbation
    rotates the manifold and CASSCF converges to a different solution. Prints the
    offending pairs so the active space can be widened to close the manifold.
    '''
    if mo_energy is None:
        return
    e = np.asarray(mo_energy)
    if e.ndim == 2:
        e = e.mean(axis=0)
    sel = set(selected)
    for i in selected:
        for j in range(len(e)):
            if j in sel:
                continue
            if abs(e[i] - e[j]) < tol:
                print(f"WARNING: active orbital {i} (E={e[i]:.6f}) is near-degenerate with "
                      f"excluded orbital {j} (E={e[j]:.6f}, dE={abs(e[i]-e[j]):.2e} Ha); the "
                      f"active-space boundary splits a degenerate manifold and the CASSCF seed "
                      f"is rotation-ambiguous. Widen the active space to include orbital {j}.")


def close_degenerate_manifolds(selected, mo_energy, tol=1e-3):
    '''Expand the selected active orbitals to whole near-degenerate manifolds.

    Orbitals are grouped into manifolds by energy: a new manifold starts wherever
    consecutive sorted energies differ by at least tol. Any manifold containing a
    selected orbital is included in full, so the active span is invariant to
    rotation within the manifold (which is what removes the seed ambiguity).
    Returns the expanded sorted index list.
    '''
    e = np.asarray(mo_energy)
    if e.ndim == 2:
        e = e.mean(axis=0)
    order = np.argsort(e, kind='stable')
    manifolds = []
    current = [int(order[0])]
    for k in range(1, len(order)):
        if e[order[k]] - e[order[k - 1]] < tol:
            current.append(int(order[k]))
        else:
            manifolds.append(current)
            current = [int(order[k])]
    manifolds.append(current)

    sel = set(int(i) for i in selected)
    closed = set(sel)
    for manifold in manifolds:
        if sel.intersection(manifold):
            closed.update(manifold)
    return sorted(closed)


def cas_electron_delta(added_orbs, mo_occ):
    '''Electrons brought in by added active orbitals (rounded total occupation).'''
    occ = np.asarray(mo_occ)
    if occ.ndim == 2:
        occ = occ[0] + occ[1]
    return int(round(sum(float(occ[a]) for a in added_orbs)))


def plan_cas_active_space(mf, cas_select, ncas, nelecas, nimp, norb,
                          ao2eo=None, ao_mol_dumps=None, ao_labels=None,
                          close_degeneracy=True):
    '''Rank the active orbitals, then close any split near-degenerate manifold.

    Returns (selected, ncas, nelecas) to build the CASSCF object with, then sort_mo.
    Ranking reuses select_cas_orbitals on a throwaway CASSCF (no kernel run); closure
    grows ncas/nelecas so the active span is invariant to within-manifold rotation.
    selected is None for cas_select == 'energy'.
    '''
    from pyscf import mcscf
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    selected = select_cas_orbitals(mc, cas_select, ncas, nimp, norb,
                                   ao2eo=ao2eo, ao_mol_dumps=ao_mol_dumps, ao_labels=ao_labels)
    if selected is None or not close_degeneracy:
        return selected, ncas, nelecas

    closed = close_degenerate_manifolds(selected, mf.mo_energy)
    added = sorted(set(closed) - set(selected))
    if added:
        d = cas_electron_delta(added, mf.mo_occ)
        if (mf.mol.nelectron - (nelecas + d)) % 2 != 0:
            raise ValueError(
                f"plan_cas_active_space: closing the degenerate manifold adds an odd "
                f"electron count (+{d} e- from orbitals {added}), leaving (nel - nelecas) "
                f"odd so the frozen core cannot be closed-shell — a singly-occupied orbital "
                f"was pulled into the active space. Set ncas/nelecas manually for this system."
            )
        print(f"qcsolver_utils: active boundary split a degenerate manifold; added orbitals "
              f"{added} (+{d} e-) to close it. ncas {ncas}->{len(closed)}, "
              f"nelecas {nelecas}->{nelecas + d}.")
        # Characterize added orbitals so the active-space change can be verified, not assumed.
        if cas_select == 'ao_character' and ao2eo is not None and ao_mol_dumps is not None and ao_labels:
            try:
                C = mf.mo_coeff if np.asarray(mf.mo_coeff).ndim == 2 else mf.mo_coeff[0] + mf.mo_coeff[1]
                w = _ao_character_weights(C, ao2eo, ao_mol_dumps, ao_labels)
                w_added = {i: round(float(w[i]), 4) for i in added}
                w_sel_min = min(float(w[i]) for i in selected)
                print(f"qcsolver_utils: added-orbital target-AO character {w_added} vs min "
                      f"selected {w_sel_min:.4f} — verify the added orbitals carry the intended character.")
            except (ValueError, KeyError):
                pass
        ncas, nelecas, selected = len(closed), nelecas + d, closed
    return selected, ncas, nelecas


def _ao_character_weights(emb_mo, ao2eo, ao_mol_dumps, ao_labels):
    '''Projection of each embedded MO onto the named AO labels (mrh getorbindex / mo_comps style).

    Uses plain Lowdin orthogonalization (pre_orth_ao=None) instead of pyscf's mo_comps, whose
    default meta-lowdin ANO reference fails on GTH/ECP/ghost-atom systems.

    Raises ValueError if ao_labels do not match any AO or if ao2eo/ao_mol_dumps are invalid.
    '''
    from pyscf import gto
    from pyscf.lo.orth import lowdin

    # Validate inputs
    if ao_mol_dumps is None:
        raise ValueError("ao_character selection: ao_mol_dumps is None")
    if ao2eo is None:
        raise ValueError("ao_character selection: ao2eo is None")
    if ao_labels is None or len(ao_labels) == 0:
        raise ValueError("ao_character selection: ao_labels is None or empty")

    ao_mol = gto.loads(ao_mol_dumps)
    ao_mo  = ao2eo @ emb_mo                       # embedded MOs in AO basis
    s      = ao_mol.intor_symmetric('int1e_ovlp')
    idx    = ao_mol.search_ao_label(ao_labels)
    if len(idx) == 0:
        raise ValueError(f"ao_character selection: no AOs match labels {ao_labels}. "
                        f"Available AO labels in system: {ao_mol.ao_labels()}")
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
