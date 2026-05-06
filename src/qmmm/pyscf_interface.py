"""
Functions to translate a QMMMCluster into a PySCF mean-field object.

Atom labeling:
  QM atoms   : 'Cu0', 'O0'       - Standard labels.
  ECP atoms  : 'X-Cu1', 'X-O1'  - 'X-' prefix for custom ECPs.
  Ghost atoms: 'ghost-Cu'        - PySCF ghost prefix.

Usage:
>>> from qmmm.pyscf_interface import build_meanfield
>>> mf, mol = build_meanfield(cluster, qm_basis='def2-svp', xc='b3lyp')
>>> mf.kernel()
"""

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
    """
    Build and return a PySCF ``gto.Mole`` from a ``QMMMCluster``.

    Parameters
    ----------
    cluster : QMMMCluster
        Output of QMMMBuilder.build().
    qm_basis : str
        Basis set name for QM and ghost atoms (e.g. 'def2-svp').
    verbose : int
        PySCF verbosity level (4 = INFO).
    max_memory : int
        Maximum memory in MB for PySCF.
    ecp_lib : dict, optional
        Mapping {element: ecp_string}.  Defaults to ECP_LIBRARY from
        ecp_library.py (Cu and O Pauli repulsion ECPs).

    Returns
    -------
    mol : pyscf.gto.Mole  (built, ready for SCF)
    """
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
    """
    Build a PySCF QM/MM mean-field object and return ``(mf, mol)``.

    The mean field is NOT yet converged — call ``mf.kernel()`` or
    ``mf.scf()`` to run the SCF.

    Automatically selects:
      - RKS  for closed-shell systems (cluster.qm_spin == 0).
      - UKS  for open-shell systems   (cluster.qm_spin != 0).

    Point charges (ECP + MM combined) are embedded via
    ``pyscf.qmmm.mm_charge`` using ``cluster.mm_coords`` and
    ``cluster.mm_charges``, matching Michael's PySCF script convention.

    Parameters
    ----------
    cluster : QMMMCluster
    qm_basis : str
        Basis set for QM and ghost atoms.
    xc : str
        DFT exchange-correlation functional.
    conv_tol : float
        SCF convergence threshold.
    max_cycle : int
        Maximum number of SCF cycles.
    density_fit : bool
        Use density fitting (RI) to accelerate the SCF.
    verbose : int
    max_memory : int  (MB)
    ecp_lib : dict, optional

    Returns
    -------
    mf  : pyscf RKS or UKS object wrapped with mm_charge
    mol : pyscf.gto.Mole (already built)
    """
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
        scf_obj = pyscf.scf.RKS(mol)
    else:
        scf_obj = pyscf.scf.UKS(mol)

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
