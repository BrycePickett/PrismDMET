"""PrismDMET impurity solver: UHF and ROHF."""

import numpy as np
from pyscf import ao2mo, gto, scf


def _embedding_mol(nel, spin=0):
    """Build a minimal dummy Mole for the embedding space."""
    mol = gto.Mole()
    mol.build(verbose=0)
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron    = nel
    mol.spin         = spin
    mol.incore_anyway = True
    return mol


def solve_uhf(const, oei, fock, tei, norb, nel, nimp, dm_guess,
              chempot_imp=0.0, spin_polarized=False, chempot_imp_beta=None):
    """Solve the embedding Hamiltonian at the UHF level. Returns (energy, rdm1) or (energy, rdm_a, rdm_b) when spin_polarized."""
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

    mol = _embedding_mol(nel, spin=spin)
    mf  = scf.UHF(mol)
    mf.get_hcore = lambda *args: np.array([h1_a, h1_b])
    mf.get_ovlp  = lambda *args: np.eye(norb)
    mf._eri      = ao2mo.restore(8, tei, norb)
    mf.scf(dm_guess)
    if not mf.converged:
        mf.max_cycle = 300
        mf.diis_space = 12
        mf.scf(mf.make_rdm1())

    rdm1_a, rdm1_b = mf.make_rdm1()
    rdm1_tot = rdm1_a + rdm1_b

    JK_a, JK_b = mf.get_veff(None, dm=mf.make_rdm1())

    e1_a = 0.5 * (np.einsum('ji,ij->', rdm1_a[:, :nimp], fock[:nimp, :] + oei[:nimp, :]) +
                  np.einsum('ji,ij->', rdm1_a[:nimp, :], fock[:, :nimp] + oei[:, :nimp]))
    e1_b = 0.5 * (np.einsum('ji,ij->', rdm1_b[:, :nimp], fock[:nimp, :] + oei[:nimp, :]) +
                  np.einsum('ji,ij->', rdm1_b[:nimp, :], fock[:, :nimp] + oei[:, :nimp]))

    e2_a = 0.5 * (np.einsum('ji,ij->', rdm1_a[:, :nimp], JK_a[:nimp, :]) +
                  np.einsum('ji,ij->', rdm1_a[:nimp, :], JK_a[:, :nimp]))
    e2_b = 0.5 * (np.einsum('ji,ij->', rdm1_b[:, :nimp], JK_b[:nimp, :]) +
                  np.einsum('ji,ij->', rdm1_b[:nimp, :], JK_b[:, :nimp]))

    energy = const + 0.5 * (e1_a + e1_b + e2_a + e2_b)

    if spin_polarized:
        return energy, rdm1_a, rdm1_b
    return energy, rdm1_tot


def solve_rohf(const, oei, fock, tei, norb, nel, nimp, dm_guess,
               chempot_imp=0.0):
    """Solve the embedding Hamiltonian at the ROHF level. Returns (energy, rdm1)."""
    spin = nel % 2

    h1 = fock.copy()
    if chempot_imp != 0.0:
        for i in range(nimp):
            h1[i, i] -= chempot_imp

    mol = _embedding_mol(nel, spin=spin)
    mf  = scf.ROHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp  = lambda *args: np.eye(norb)
    mf._eri      = ao2mo.restore(8, tei, norb)
    mf.scf(dm_guess)
    if not mf.converged:
        mf.max_cycle = 300
        mf.diis_space = 12
        mf.scf(mf.make_rdm1())

    rdm1_a, rdm1_b = mf.make_rdm1()
    rdm1_tot = rdm1_a + rdm1_b

    JK_a, JK_b = mf.get_veff(None, dm=mf.make_rdm1())

    e1_a = 0.5 * (np.einsum('ji,ij->', rdm1_a[:, :nimp], fock[:nimp, :] + oei[:nimp, :]) +
                  np.einsum('ji,ij->', rdm1_a[:nimp, :], fock[:, :nimp] + oei[:, :nimp]))
    e1_b = 0.5 * (np.einsum('ji,ij->', rdm1_b[:, :nimp], fock[:nimp, :] + oei[:nimp, :]) +
                  np.einsum('ji,ij->', rdm1_b[:nimp, :], fock[:, :nimp] + oei[:, :nimp]))

    e2_a = 0.5 * (np.einsum('ji,ij->', rdm1_a[:, :nimp], JK_a[:nimp, :]) +
                  np.einsum('ji,ij->', rdm1_a[:nimp, :], JK_a[:, :nimp]))
    e2_b = 0.5 * (np.einsum('ji,ij->', rdm1_b[:, :nimp], JK_b[:nimp, :]) +
                  np.einsum('ji,ij->', rdm1_b[:nimp, :], JK_b[:, :nimp]))

    energy = const + 0.5 * (e1_a + e1_b + e2_a + e2_b)

    return energy, rdm1_tot


def execute(task):
    """SolverDispatcher entry point for UHF and ROHF solvers."""
    method         = task['method']
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
        chempot_imp = task.get('chempot_imp', 0.0),
    )

    if method == 'UHF':
        if spin_polarized:
            energy, rdm_a, rdm_b = solve_uhf(
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
            }
        else:
            energy, rdm1 = solve_uhf(**common)
            return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}

    elif method == 'ROHF':
        energy, rdm1 = solve_rohf(**common)
        return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}

    else:
        raise ValueError(
            f"uhf.execute: unexpected method='{method}'. "
            f"Expected 'UHF' or 'ROHF'."
        )
