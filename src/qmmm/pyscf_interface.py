"""Translate a QMMMCluster into a PySCF Mole and mean-field object."""

from __future__ import annotations

from typing import Optional, Dict, Tuple

from .cluster import QMMMCluster
from .ecp_library import ECP_LIBRARY


def build_mol(
    cluster:   QMMMCluster,
    qm_basis:  str  = 'def2-svp',
    verbose:   int  = 4,
    max_memory: int = 200_000,
    ecp_lib:   Optional[Dict[str, str]] = None,
):
    """Build and return a PySCF gto.Mole from a QMMMCluster (built, ready for SCF)."""
    try:
        import pyscf.gto as gto
    except ImportError as e:
        raise ImportError(
            "PySCF is required for build_mol().  "
            "Install it with: pip install pyscf"
        ) from e

    if ecp_lib is None:
        ecp_lib = ECP_LIBRARY

    mol             = gto.Mole()
    mol.verbose     = verbose
    mol.max_memory  = max_memory
    mol.symmetry    = False
    mol.spin        = cluster.qm_spin
    mol.charge      = int(round(cluster.qm_charge))
    mol.atom        = cluster.to_pyscf_mol_atom()
    mol.basis       = cluster.basis_dict(qm_basis=qm_basis)
    mol.ecp         = cluster.ecp_dict(ecp_lib)
    mol.build()

    return mol


def build_meanfield(
    cluster:    QMMMCluster,
    qm_basis:   str   = 'def2-svp',
    xc:         str   = 'b3lyp',
    conv_tol:   float = 1e-8,
    max_cycle:  int   = 200,
    density_fit: bool = True,
    verbose:    int   = 4,
    max_memory: int   = 200_000,
    ecp_lib:    Optional[Dict[str, str]] = None,
) -> Tuple:
    """Build a PySCF QM/MM mean-field (RKS or UKS) with mm_charge embedding. Returns (mf, mol); not yet converged."""
    try:
        import pyscf.scf
        import pyscf.dft
        import pyscf.qmmm
    except ImportError as e:
        raise ImportError("PySCF is required.") from e

    mol = build_mol(
        cluster,
        qm_basis=qm_basis,
        verbose=verbose,
        max_memory=max_memory,
        ecp_lib=ecp_lib,
    )

    # Choose RKS or UKS based on spin
    if cluster.qm_spin == 0:
        scf_obj = pyscf.dft.RKS(mol)
    else:
        scf_obj = pyscf.dft.UKS(mol)

    scf_obj.xc        = xc
    scf_obj.conv_tol  = conv_tol
    scf_obj.max_cycle = max_cycle

    # Wrap with QM/MM point charge embedding
    # mm_coords and mm_charges include both ECP and MM atoms — exactly
    # matching the 'coords' / 'charges' arrays in Michael's PySCF scripts.
    mf = pyscf.qmmm.mm_charge(
        scf_obj,
        cluster.mm_coords,
        cluster.mm_charges,
    )

    if density_fit:
        mf = mf.density_fit()

    return mf, mol
