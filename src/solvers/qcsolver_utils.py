'''
    Utility functions for QC-DMET active-space solvers.

    Provides orbital projection tools for warm-restarting CASSCF across
    DMET self-consistency iterations, preventing root collapse when the
    u-matrix changes the embedding Hamiltonian.

    Adapted from mrh.my_dmet.pyscf_casscf (Hung Pham / Matthew Hermes)
    for the QC-DMET framework.

    Key function
    ------------
    project_amo_manually : project active-space MOs from the previous
        iteration onto the current embedding basis, producing a stable
        initial guess for CASSCF that tracks the same electronic state.
'''

import numpy as np
from scipy import linalg


def project_amo_manually(old_mo_coeff, ncas, ncore, new_fock, Norb):
    '''
    Re-order and project old CASSCF MOs onto a new embedding basis so that
    the active space is preserved across u-matrix updates.

    Given the MO coefficient matrix from a *previous* CASSCF solve, this
    function:
      1. Extracts the old active MOs (columns ncore : ncore+ncas).
      2. Builds a projector onto that active subspace and diagonalises it
         to find the "most active-like" directions in the current basis.
      3. Sorts the inactive (core + virtual) orbitals by the Fock matrix
         eigenvalues so the CASSCF macro-iterations start from a
         physically sensible ordering.
      4. Applies a sign convention so that CI vector overlaps remain
         positive (avoids spurious phase flips).

    Parameters
    ----------
    old_mo_coeff : ndarray (Norb, Norb)
        MO coefficients from the previous CASSCF solve, in the local
        DMET orbital basis.
    ncas : int
        Number of active orbitals.
    ncore : int
        Number of core (doubly-occupied, frozen) orbitals.
    new_fock : ndarray (Norb, Norb)
        Current Fock matrix in the local DMET orbital basis (with the
        new u-matrix applied).  Used only to order inactive orbitals.
    Norb : int
        Total number of embedding orbitals (impurity + bath).

    Returns
    -------
    new_mo : ndarray (Norb, Norb)
        Re-ordered MO coefficients suitable as an initial guess for
        mcscf.CASSCF.kernel(new_mo).  The columns are ordered:
        [core | active | virtual].
    fidelity : ndarray (ncas,)
        Eigenvalues of the projector restricted to the active subspace.
        Values close to 1.0 mean the old active orbital survived the
        basis change intact; values << 1 indicate that the active space
        has shifted significantly.
    '''
    assert old_mo_coeff.shape == (Norb, Norb), \
        f"project_amo_manually: expected old_mo_coeff shape ({Norb},{Norb}), got {old_mo_coeff.shape}"
    nocc = ncore + ncas

    # --- Extract old active MOs ---
    old_amo = old_mo_coeff[:, ncore:nocc]          # (Norb, ncas)

    # --- Build projector onto old active space and diagonalise ---
    # P = old_amo @ old_amo^T is the projector.
    # Diagonalise in the full space to get the ncas directions with
    # eigenvalue closest to 1.
    proj = old_amo @ old_amo.T                     # (Norb, Norb)
    evals, evecs = np.linalg.eigh(proj)
    # eigh returns ascending; we want the *largest* eigenvalues first
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    new_amo = evecs[:, :ncas].copy()               # most "active-like"
    new_imo = evecs[:, ncas:].copy()               # inactive complement
    fidelity = evals[:ncas].copy()

    # --- Sign convention: align each new AMO with the old AMO ---
    overlap = new_amo.T @ old_amo                  # (ncas, ncas)
    for i in range(ncas):
        if overlap[i, i] < 0:
            new_amo[:, i] *= -1

    # --- Sort inactive orbitals by Fock eigenvalues ---
    fock_imo = new_imo.T @ new_fock @ new_imo
    imo_evals, imo_evecs = np.linalg.eigh(fock_imo)
    new_imo = new_imo @ imo_evecs
    # Split into core (lowest) and virtual (highest)
    new_cmo = new_imo[:, :ncore]
    new_vmo = new_imo[:, ncore:]

    # --- Assemble [core | active | virtual] ---
    new_mo = np.concatenate([new_cmo, new_amo, new_vmo], axis=1)

    print(f"qcsolver_utils::project_amo : fidelity = {fidelity}")
    return new_mo, fidelity


def fix_casscf_for_nonsinglet_env(mc, h1e_s):
    '''
    Wrap a PySCF CASSCF object so that it correctly minimises in the
    presence of a spin-dependent one-electron potential (open-shell
    environment).

    This is the QC-DMET adaptation of mrh.my_dmet.pyscf_casscf.
    fix_my_CASSCF_for_nonsinglet_env.  It intercepts:

        * fcisolver.kernel  — splits h1e into [h1e+h1e_s, h1e-h1e_s]
        * fcisolver.make_rdm12 — caches the spin-density matrix
        * mc.casci — re-projects h1e_s into the current active space
        * mc.update_casdm — re-projects h1e_s after micro-rotation u
        * mc.solve_approx_ci — stacks the two-component h1
        * mc.gen_g_hop — adds the spin-potential orbital gradient and
          Hessian contributions

    Parameters
    ----------
    mc : mcscf.CASSCF
        A fully initialised (but not yet solved) PySCF CASSCF object.
    h1e_s : ndarray (Norb, Norb) or None
        The spin-dependent one-electron potential in the local embedding
        basis:  h1e_s = (h_alpha - h_beta) / 2.
        If None or all zeros, ``mc`` is returned unchanged.

    Returns
    -------
    mc_fixed : CASSCF-like
        A modified CASSCF object whose .kernel() will correctly account
        for the open-shell environment.
    '''
    from pyscf import lib

    if h1e_s is None or np.all(np.abs(h1e_s) < 1e-8):
        return mc

    # Initial projection into the active space
    amo = mc.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
    amoH = amo.conj().T
    h1e_s_amo  = amoH @ h1e_s @ amo
    h1e_s_amou = h1e_s_amo.copy()
    last_cached_sdm = np.zeros_like(h1e_s_amo)

    # ------------------------------------------------------------------
    # Wrapped FCI solver
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Wrapped CASSCF driver
    # ------------------------------------------------------------------
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
