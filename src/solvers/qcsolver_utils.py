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
                          close_degeneracy=True, natorb_occ_thresh=0.02,
                          natorb_max_superset=None, sa_nstates=1, avas_threshold=0.2,
                          spade_gap_tol=0.3, spade_n_fallback=16):
    '''Decide the CAS active space. Returns (selected, ncas, nelecas, mo_coeff).

    Index modes ('impurity'/'ao_character'): mo_coeff is None and the caller does
    mc.sort_mo(selected); near-degenerate manifolds touching the selection are closed.
    'natorb': selected is None and the caller sets mc.mo_coeff = mo_coeff directly; the
    active space is the fractionally occupied CASCI natural orbitals (character-free,
    rotation-invariant; ncas/nelecas are derived). 'energy': all three extras are None.
    '''
    if cas_select == 'natorb':
        mo_coeff, ncas, nelecas = natorb_active_space(
            mf, ncas, occ_thresh=natorb_occ_thresh, max_superset=natorb_max_superset,
            sa_nstates=sa_nstates)
        return None, ncas, nelecas, mo_coeff

    if cas_select == 'avas':
        mo_coeff, ncas, nelecas = avas_active_space(
            mf, ao2eo, ao_mol_dumps, ao_labels, threshold=avas_threshold)
        return None, ncas, nelecas, mo_coeff

    if cas_select == 'spade':
        mo_coeff, ncas, nelecas = spade_active_space(
            mf, nimp,
            gap_tol=spade_gap_tol, n_fallback=spade_n_fallback,
            occ_thresh=natorb_occ_thresh, max_superset=natorb_max_superset,
            sa_nstates=sa_nstates)
        return None, ncas, nelecas, mo_coeff

    from pyscf import mcscf
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    selected = select_cas_orbitals(mc, cas_select, ncas, nimp, norb,
                                   ao2eo=ao2eo, ao_mol_dumps=ao_mol_dumps, ao_labels=ao_labels)
    if selected is None or not close_degeneracy:
        return selected, ncas, nelecas, None

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
    return selected, ncas, nelecas, None


def natorb_active_space(mf, n_superset, occ_thresh=0.02, deg_tol=1e-3, max_superset=None,
                        sa_nstates=1):
    '''Character-free active space from CASCI natural-orbital occupations.

    Builds a deterministic superset (n_superset orbitals around the Fermi level,
    snapped out to complete degenerate manifolds), runs a CASCI on it, and keeps the
    natural orbitals with fractional occupation (occ_thresh < n < 2 - occ_thresh) as
    the active space. Character-free and invariant to within-manifold rotation;
    n_superset is a methodological parameter (converge it). Returns (mo_coeff, ncas,
    nelecas) with the active NOs in the [ncore, ncore+ncas) window.

    sa_nstates > 1 selects on the equal-weight state-averaged density over that many
    CASCI roots (CISNO-style): a ground-state density is blind to orbitals that are
    integer-occupied in the GS but define an excitation, so a state-averaged target
    needs a state-averaged selection density.

    max_superset, if set, raises if manifold snapping inflates the superset past it
    (a dense Fermi-level manifold can push the exact CASCI past the tractable limit).
    '''
    from pyscf import mcscf
    from math import comb
    C = np.asarray(mf.mo_coeff)
    if C.ndim == 3:
        raise NotImplementedError("natorb selection expects a single set of orbitals (RHF/ROHF reference).")
    mo_e = np.asarray(mf.mo_energy)
    occ = np.asarray(mf.mo_occ)
    norb = C.shape[1]
    nocc = int(np.sum(occ > 0))

    half = max(1, int(n_superset) // 2)
    lo, hi = max(0, nocc - half), min(norb, nocc + half)
    while lo > 0 and (mo_e[lo] - mo_e[lo - 1]) < deg_tol:
        lo -= 1
    while hi < norb and (mo_e[hi] - mo_e[hi - 1]) < deg_tol:
        hi += 1
    ncas_s = hi - lo
    if max_superset is not None and ncas_s > max_superset:
        raise ValueError(
            f"natorb_active_space: manifold snapping inflated the superset to {ncas_s} "
            f"orbitals (window [{lo}, {hi})), exceeding max_superset={max_superset}. A dense "
            f"near-degenerate manifold at the Fermi level pushed the exact CASCI past the "
            f"tractable/QD-NEVPT2 limit. Lower n_superset, raise max_superset (exact-CASCI "
            f"cost grows ~factorially), or use a cheaper selector (locality_window / MP2 pre-screen)."
        )
    win = occ[lo:hi]
    na = int(np.sum(np.rint(win) >= 1))   # alpha occupied in window
    nb = int(np.sum(np.rint(win) >= 2))   # beta (doubly) occupied in window

    # Report superset size and FCI cost before the (possibly multi-hour) CASCI; flush
    # so the line reaches a buffered SLURM log even if the CASCI then hangs.
    fci_dim = comb(ncas_s, na) * comb(ncas_s, nb)
    print(f"qcsolver_utils: natorb superset window [{lo}, {hi}) = {ncas_s} orbitals, "
          f"CASCI({na + nb},{ncas_s}) x {sa_nstates} root(s), FCI dim ~{fci_dim:.2e} dets "
          f"- starting CASCI...", flush=True)

    mc = mcscf.CASCI(mf, ncas_s, (na, nb))
    mc.fcisolver.conv_tol = 1e-10
    mc.verbose = 0
    if sa_nstates > 1:
        mc.fcisolver.nroots = sa_nstates
    mc.kernel()

    # State-averaged selection density when targeting >1 state (CISNO-style); equal
    # weights so every target state's orbitals register in the selection.
    if sa_nstates > 1:
        dm1 = sum(mc.fcisolver.make_rdm1(civec, ncas_s, mc.nelecas)
                  for civec in mc.ci) / sa_nstates
    else:
        dm1 = mc.fcisolver.make_rdm1(mc.ci, ncas_s, mc.nelecas)
    no_occ, u = np.linalg.eigh(dm1)
    order = np.argsort(no_occ)[::-1]
    no_occ, u = no_occ[order], u[:, order]
    cas_no = mc.mo_coeff[:, mc.ncore:mc.ncore + ncas_s] @ u

    is_core = no_occ >= 2 - occ_thresh
    is_act  = (no_occ > occ_thresh) & (no_occ < 2 - occ_thresh)
    if not np.any(is_act):
        raise ValueError(
            f"natorb_active_space: no fractionally occupied NOs in the CAS({na+nb},{ncas_s}) "
            f"superset (occupations {np.round(no_occ, 3).tolist()}). Increase n_superset.")

    # Canonicalize within each degenerate group of active NOs (Part A determinism fix).
    # eigh of the 1-RDM returns eigenvectors defined up to an arbitrary rotation within each
    # degenerate eigenspace; BLAS noise (~1e-9) can rotate an exactly degenerate pair by O(100°).
    # Diagonalizing the embedded Fock restricted to each degenerate group pins a unique, physically
    # motivated orientation that is stable to float64-level perturbations.
    F_emb = (mf.mo_coeff * mf.mo_energy) @ mf.mo_coeff.T
    act_idx = np.where(is_act)[0]
    if len(act_idx) > 1:
        act_cols = cas_no[:, act_idx].copy()
        act_occ_vals = no_occ[act_idx]
        i = 0
        while i < len(act_idx):
            j = i + 1
            while j < len(act_idx) and abs(act_occ_vals[j] - act_occ_vals[i]) < deg_tol:
                j += 1
            if j - i > 1:
                blk = act_cols[:, i:j]
                _, U = np.linalg.eigh(blk.T @ F_emb @ blk)
                act_cols[:, i:j] = blk @ U
            i = j
        for c in range(act_cols.shape[1]):
            if act_cols[np.argmax(np.abs(act_cols[:, c])), c] < 0:
                act_cols[:, c] *= -1
        cas_no[:, act_idx] = act_cols

    mo = C.copy()
    mo[:, lo:hi] = np.hstack([cas_no[:, is_core], cas_no[:, is_act], cas_no[:, ~is_core & ~is_act]])
    ncore = lo + int(np.sum(is_core))
    ncas = int(np.sum(is_act))
    nelecas = int(round(float(np.sum(no_occ[is_act]))))

    if sa_nstates > 1:
        spin = int(mf.mol.spin)
        na_sel, nb_sel = (nelecas + spin) // 2, (nelecas - spin) // 2
        if comb(ncas, na_sel) * comb(ncas, nb_sel) < sa_nstates:
            raise ValueError(
                f"natorb_active_space: selected CAS({nelecas},{ncas}) supports fewer than "
                f"sa_nstates={sa_nstates} states, so a state-averaged solver cannot build that "
                f"many roots (this is the CAS(1,1) IndexError failure mode). The natorb density "
                f"found little multireference character in this window; widen n_superset, loosen "
                f"occ_thresh, or use a selector that targets the trap orbitals directly "
                f"(locality_window / ao_character)."
            )

    print(f"qcsolver_utils: natorb CAS from CASCI({na + nb},{ncas_s}) superset -> "
          f"CAS({nelecas},{ncas}); active NO occupations "
          f"{np.round(no_occ[is_act], 4).tolist()}.")
    return mo, ncas, nelecas


def spade_active_space(mf, nimp, gap_tol=0.3, n_fallback=16, occ_thresh=0.02,
                       deg_tol=1e-3, sa_nstates=1, max_superset=None):
    '''SPADE-like active space from impurity-projection weight gap detection.

    Ranks all embedded MOs by their impurity projection weight w[j]=||C[:nimp,j]||^2,
    finds the largest relative gap in the sorted weight spectrum to determine the
    superset size automatically, runs CASCI on that superset, and selects active NOs
    by fractional occupation. Applies Part A degenerate-NO Fock canonicalization.

    Adapted from: Kolodzeiski & Stein, J. Chem. Theory Comput. 19, 6643 (2023).
    In DMET the impurity AO block (first nimp rows of C) is the natural projection
    target; no SVD or user-specified AO labels are needed.

    gap_tol:    minimum relative gap (w[k]-w[k+1])/w[k] to call it a partition.
                Lower this if the spectrum is smooth (no clean jump).
    n_fallback: superset size when gap detection fails (like natorb's n_superset).
    '''
    from pyscf import mcscf
    from math import comb

    C    = np.asarray(mf.mo_coeff)
    if C.ndim == 3:
        raise NotImplementedError("spade selection expects a single set of orbitals (RHF/ROHF reference).")
    occ  = np.asarray(mf.mo_occ)
    mo_e = np.asarray(mf.mo_energy)
    norb = C.shape[1]
    n_total_el = int(round(float(np.sum(occ))))
    spin = int(mf.mol.spin)

    # Stage 1: impurity-projection weight and gap detection.
    w     = np.sum(C[:nimp, :] ** 2, axis=0)       # (norb,) fraction on impurity AOs
    order = np.argsort(w)[::-1]                    # most impurity-like first
    w_s   = w[order]

    sig  = np.where(w_s > occ_thresh)[0]
    n_sig = len(sig)

    if n_sig > 1:
        g     = np.zeros(n_sig - 1)
        for k in range(n_sig - 1):
            if w_s[k] > 1e-10:
                g[k] = (w_s[k] - w_s[k + 1]) / w_s[k]
        k_best = int(np.argmax(g))
        print(f"qcsolver_utils: spade weight spectrum "
              f"(top {min(n_sig, 12)}/{norb} MOs): "
              f"{np.round(w_s[:min(n_sig, 12)], 3).tolist()}  "
              f"largest gap @ k={k_best}: rel={g[k_best]:.3f} "
              f"(w: {w_s[k_best]:.3f} -> {w_s[k_best+1]:.3f})", flush=True)
        if g[k_best] >= gap_tol:
            n_spade = k_best + 1
            print(f"qcsolver_utils: spade gap accepted: n_spade={n_spade}")
        else:
            print(f"qcsolver_utils: spade no gap >= {gap_tol:.2f}; "
                  f"falling back to n_fallback={n_fallback}")
            n_spade = min(n_fallback, norb)
    else:
        print(f"qcsolver_utils: spade: fewer than 2 MOs with w > {occ_thresh}; "
              f"falling back to n_fallback={n_fallback}")
        n_spade = min(n_fallback, norb)

    # Stage 2: build superset, ensuring any SOMO is included.
    spade_set = set(int(order[k]) for k in range(n_spade))
    somo_forced = [j for j in range(norb) if 0 < occ[j] < 2 and j not in spade_set]
    if somo_forced:
        print(f"qcsolver_utils: spade: SOMO(s) {somo_forced} "
              f"(w={[round(float(w[j]), 3) for j in somo_forced]}) "
              f"not in superset; forcing inclusion.")
        spade_set.update(somo_forced)
        n_spade = len(spade_set)

    if max_superset is not None and n_spade > max_superset:
        raise ValueError(
            f"spade_active_space: superset size {n_spade} exceeds "
            f"max_superset={max_superset}. Raise gap_tol to shrink it.")

    core_idx  = sorted([j for j in range(norb)
                        if j not in spade_set and occ[j] >= 2 - 1e-6],
                       key=lambda j: mo_e[j])
    virt_idx  = sorted([j for j in range(norb)
                        if j not in spade_set and occ[j] <  1e-6],
                       key=lambda j: mo_e[j])
    # Keep superset MOs in descending weight order for reproducibility.
    spade_idx = sorted(spade_set, key=lambda j: -w[j])

    n_core_el  = 2 * len(core_idx)
    nelecas_s  = n_total_el - n_core_el
    ncore_s    = len(core_idx)
    na = (nelecas_s + spin) // 2
    nb = (nelecas_s - spin) // 2

    # Stage 3: CASCI on the superset.
    fci_dim = comb(n_spade, na) * comb(n_spade, nb)
    print(f"qcsolver_utils: spade CASCI({nelecas_s},{n_spade}) "
          f"x {sa_nstates} root(s), FCI dim ~{fci_dim:.2e} - starting CASCI...",
          flush=True)

    C_spade = np.hstack([C[:, core_idx], C[:, spade_idx], C[:, virt_idx]])
    mc = mcscf.CASCI(mf, n_spade, (na, nb))
    mc.mo_coeff = C_spade
    mc.fcisolver.conv_tol = 1e-10
    mc.verbose = 0
    if sa_nstates > 1:
        mc.fcisolver.nroots = sa_nstates
    mc.kernel()

    if sa_nstates > 1:
        dm1 = sum(mc.fcisolver.make_rdm1(ci, n_spade, mc.nelecas)
                  for ci in mc.ci) / sa_nstates
    else:
        dm1 = mc.fcisolver.make_rdm1(mc.ci, n_spade, mc.nelecas)

    no_occ, u  = np.linalg.eigh(dm1)
    order_no   = np.argsort(no_occ)[::-1]
    no_occ, u  = no_occ[order_no], u[:, order_no]
    cas_no     = mc.mo_coeff[:, ncore_s:ncore_s + n_spade] @ u

    is_core = no_occ >= 2 - occ_thresh
    is_act  = (no_occ > occ_thresh) & (no_occ < 2 - occ_thresh)
    if not np.any(is_act):
        raise ValueError(
            f"spade_active_space: no fractionally occupied NOs in the "
            f"SPADE CAS({nelecas_s},{n_spade}) superset "
            f"(occupations {np.round(no_occ, 3).tolist()}). "
            f"Lower gap_tol or raise n_fallback to widen the superset.")

    # Stage 4: Part A Fock canonicalization within degenerate active-NO groups.
    F_emb        = (mf.mo_coeff * mf.mo_energy) @ mf.mo_coeff.T
    act_idx      = np.where(is_act)[0]
    act_cols     = cas_no[:, act_idx].copy()
    act_occ_vals = no_occ[act_idx]
    i = 0
    while i < len(act_idx):
        j = i + 1
        while j < len(act_idx) and abs(act_occ_vals[j] - act_occ_vals[i]) < deg_tol:
            j += 1
        if j - i > 1:
            blk = act_cols[:, i:j]
            _, U = np.linalg.eigh(blk.T @ F_emb @ blk)
            act_cols[:, i:j] = blk @ U
        i = j
    for c in range(act_cols.shape[1]):
        if act_cols[np.argmax(np.abs(act_cols[:, c])), c] < 0:
            act_cols[:, c] *= -1
    cas_no[:, act_idx] = act_cols

    # Stage 5: assemble final MO matrix.
    mo = C_spade.copy()
    mo[:, ncore_s:ncore_s + n_spade] = np.hstack(
        [cas_no[:, is_core], cas_no[:, is_act], cas_no[:, ~is_core & ~is_act]])
    ncore_out   = ncore_s + int(np.sum(is_core))
    ncas_out    = int(np.sum(is_act))
    nelecas_out = int(round(float(np.sum(no_occ[is_act]))))

    if sa_nstates > 1:
        na_sel = (nelecas_out + spin) // 2
        nb_sel = (nelecas_out - spin) // 2
        if comb(ncas_out, na_sel) * comb(ncas_out, nb_sel) < sa_nstates:
            raise ValueError(
                f"spade_active_space: selected CAS({nelecas_out},{ncas_out}) "
                f"supports fewer than sa_nstates={sa_nstates} states. "
                f"Lower gap_tol or raise n_fallback.")

    print(f"qcsolver_utils: spade CAS from CASCI({nelecas_s},{n_spade}) -> "
          f"CAS({nelecas_out},{ncas_out}); active NO occupations "
          f"{np.round(no_occ[is_act], 4).tolist()}.")
    return mo, ncas_out, nelecas_out


def multiseed_casscf(mc, mo_seed, deg_tol=1e-3, angles=(0, 30, 60, 90)):
    '''Run CASSCF from deterministic rotated seeds; return mc at the lowest-energy solution.

    Detects degenerate active-orbital pairs via a single CASCI on mo_seed, generates
    rotated seeds at the given angles (degrees), runs CASSCF from each, and keeps the
    lowest SA-energy (or total-energy for single-state) solution. Reproducible because
    seeds are deterministic angles and the argmin is deterministic.

    Targets the confirmed PySCF CASSCF basin-hopping problem (GitHub issues #912, #1033):
    the lowest-SA-energy solution is the variationally correct one. If no degenerate pairs
    are found, falls through to a single mc.kernel(mo_seed) call.
    '''
    from pyscf import mcscf as _mcscf

    ncore = mc.ncore
    ncas  = mc.ncas
    nelecas = mc.nelecas
    nroots = len(mc.weights) if hasattr(mc, 'weights') else 1

    # Single CASCI to measure active-NO occupations and detect degenerate pairs.
    ci_det = _mcscf.CASCI(mc._scf, ncas, nelecas)
    ci_det.verbose = 0
    if nroots > 1:
        ci_det.fcisolver.nroots = nroots
    ci_det.kernel(mo_seed)
    if nroots > 1 and isinstance(ci_det.ci, list):
        dm1 = sum(ci_det.fcisolver.make_rdm1(ci, ncas, nelecas)
                  for ci in ci_det.ci) / nroots
    else:
        dm1 = ci_det.fcisolver.make_rdm1(ci_det.ci, ncas, nelecas)
    occ = np.sort(np.linalg.eigvalsh(dm1))[::-1]
    deg_pairs = [(i, i + 1) for i in range(ncas - 1)
                 if abs(occ[i] - occ[i + 1]) < deg_tol]

    if not deg_pairs:
        print(f"qcsolver_utils: multiseed_casscf: no degenerate active pairs "
              f"(min occ gap={np.min(np.abs(np.diff(occ))):.2e}); single kernel call.")
        mc.kernel(mo_seed)
        return mc

    print(f"qcsolver_utils: multiseed_casscf: {len(deg_pairs)} degenerate pair(s) "
          f"{deg_pairs}; running {len(angles)} seeds.", flush=True)

    best_e  = np.inf
    best_mo = None

    for angle in angles:
        mo = mo_seed.copy()
        th = np.deg2rad(angle)
        c_r, s_r = np.cos(th), np.sin(th)
        for i, j in deg_pairs:
            col_i = mo[:, ncore + i].copy()
            col_j = mo[:, ncore + j].copy()
            mo[:, ncore + i] =  c_r * col_i + s_r * col_j
            mo[:, ncore + j] = -s_r * col_i + c_r * col_j
        mc.kernel(mo)
        print(f"  seed {angle:3d}°: e_tot={mc.e_tot:.10f}  converged={mc.converged}")
        if mc.e_tot < best_e:
            best_e  = mc.e_tot
            best_mo = mc.mo_coeff.copy()

    # Re-run from the best seed so mc is fully consistent at that solution.
    if abs(mc.e_tot - best_e) > 1e-10:
        mc.kernel(best_mo)
    print(f"qcsolver_utils: multiseed_casscf -> best SA-energy = {best_e:.10f} Ha")
    return mc


def avas_active_space(mf, ao2eo, ao_mol_dumps, ao_labels, threshold=0.2,
                      openshell_option=None, canonicalize=True, deg_tol=1e-3):
    '''AVAS active space (arXiv:1701.07862) in the DMET embedding basis.

    Projects embedded MOs onto the named real AOs (reached via ao2eo) and rotates the
    occupied/virtual blocks to span that projection; the active space is the orbitals
    whose projector eigenvalue exceeds threshold. Rotation-invariant within a
    degenerate block by construction. Mirrors pyscf.mcscf.avas using the named AOs of
    the real basis as the reference (equivalent to pyscf avas with minao set to the
    molecule's own basis), since minao fails on Cu/O/ghost+ECP. openshell_option
    defaults to 3 (SOMOs always active) for ROHF, 2 for closed-shell. Returns
    (mo_coeff, ncas, nelecas) ordered [core, active, virtual].
    '''
    from pyscf import gto
    import scipy.linalg

    C = np.asarray(mf.mo_coeff)
    if C.ndim == 3:
        raise NotImplementedError("avas selection expects a single set of orbitals (RHF/ROHF reference).")
    if ao2eo is None or ao_mol_dumps is None or not ao_labels:
        raise ValueError("avas selection requires ao2eo, ao_mol_dumps, and ao_labels.")
    occ = np.asarray(mf.mo_occ)
    mo_e = np.asarray(mf.mo_energy)
    spin = int(mf.mol.spin)
    nocc = int(np.count_nonzero(occ != 0))
    if openshell_option is None:
        openshell_option = 3 if spin != 0 else 2

    # MO-space projector onto the named AOs (PySCF avas non-IAO construction).
    ao_mol = gto.loads(ao_mol_dumps)
    s = ao_mol.intor_symmetric('int1e_ovlp')
    idx = ao_mol.search_ao_label(ao_labels)
    if len(idx) == 0:
        raise ValueError(f"avas: no AOs match labels {ao_labels}. Available: {ao_mol.ao_labels()}")
    M = ao2eo @ C
    s2, s21 = s[idx][:, idx], s[idx] @ M
    sa = s21.T @ scipy.linalg.solve(s2, s21, assume_a='pos')

    # Residual reproducibility risk: AVAS partitions occ/vir by occupation; a manifold
    # straddling the Fermi level can move an orbital between blocks if occupation flips.
    if 0 < nocc < len(mo_e) and abs(mo_e[nocc] - mo_e[nocc - 1]) < deg_tol:
        print(f"WARNING: avas: near-degenerate manifold straddles the occ/vir boundary "
              f"(E[{nocc-1}]={mo_e[nocc-1]:.6f}, E[{nocc}]={mo_e[nocc]:.6f}, "
              f"dE={abs(mo_e[nocc]-mo_e[nocc-1]):.2e} Ha); an occupation flip here can move an "
              f"orbital between AVAS blocks and break reproducibility — verify the A/A result.")

    if openshell_option == 2:
        docc = nocc
    elif openshell_option == 3:
        docc = nocc - spin
    else:
        raise ValueError(f"avas: unknown openshell_option {openshell_option} (use 2 or 3).")

    wocc, uo = np.linalg.eigh(sa[:docc, :docc])
    nelecas = mf.mol.nelectron - int((wocc < threshold).sum()) * 2
    mocore  = C[:, :docc] @ uo[:, wocc < threshold]
    mocas_o = C[:, :docc] @ uo[:, wocc >= threshold]

    wvir, uv = np.linalg.eigh(sa[nocc:, nocc:])
    mocas_v = C[:, nocc:] @ uv[:, wvir >= threshold]
    movir   = C[:, nocc:] @ uv[:, wvir < threshold]

    if openshell_option == 3:
        mocas = np.hstack([mocas_o, C[:, docc:nocc], mocas_v])
    else:
        mocas = np.hstack([mocas_o, mocas_v])

    if canonicalize:
        fock_emb = (C * mo_e) @ C.T   # embedding overlap is identity
        def canon(block):
            if block.shape[1] == 0:
                return block
            _, u = np.linalg.eigh(block.T @ fock_emb @ block)
            return block @ u
        mocore, mocas, movir = canon(mocore), canon(mocas), canon(movir)

    mo = np.hstack([mocore, mocas, movir])
    ncas = mocas.shape[1]
    print(f"qcsolver_utils: avas (option {openshell_option}, threshold {threshold}) -> "
          f"CAS({nelecas},{ncas}); occ eig {np.round(np.sort(wocc)[::-1], 3).tolist()}, "
          f"vir eig {np.round(np.sort(wvir)[::-1], 3).tolist()}.")
    return mo, ncas, nelecas


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
