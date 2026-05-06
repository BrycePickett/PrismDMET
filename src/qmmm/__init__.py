"""Public API for the PrismDMET QM/MM embedding module."""

from .builder          import QMMMBuilder
from .cluster          import (
    QMMMCluster, AtomRecord,
    REGION_QM, REGION_ECP, REGION_MM, REGION_GHOST,
)
from .material_config  import MaterialConfig, CU2O_CONFIG
from .ecp_library      import ECP_LIBRARY, get_ecp_string, has_ecp
from .pyscf_interface  import build_mol, build_meanfield

__all__ = [
    # Main builder
    'QMMMBuilder',
    # Data objects
    'QMMMCluster',
    'AtomRecord',
    # Region constants
    'REGION_QM',
    'REGION_ECP',
    'REGION_MM',
    'REGION_GHOST',
    # Material configs
    'MaterialConfig',
    'CU2O_CONFIG',
    # ECP library
    'ECP_LIBRARY',
    'get_ecp_string',
    'has_ecp',
    # PySCF integration
    'build_mol',
    'build_meanfield',
]
