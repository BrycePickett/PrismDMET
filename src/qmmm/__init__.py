"""Public API for the PrismDMET QM/MM embedding module."""

from .builder          import QMMMBuilder
from .defect_builder    import DefectBuilder
from .cluster          import (
    QMMMCluster, AtomRecord,
    REGION_QM, REGION_ECP, REGION_MM, REGION_GHOST,
)
from .material_config  import MaterialConfig, cu2o_matproj_config, cu2o_hse06_opt_config
from .ecp_library      import ECP_LIBRARY, get_ecp_string, has_ecp
from .pyscf_interface  import build_mol, build_meanfield

__all__ = [
    # Main builder
    'QMMMBuilder',
    # Lightweight defect-region selector from XYZ pairs
    'DefectBuilder',
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
    'cu2o_matproj_config',
    'cu2o_hse06_opt_config',
    # ECP library
    'ECP_LIBRARY',
    'get_ecp_string',
    'has_ecp',
    # PySCF integration
    'build_mol',
    'build_meanfield',
]
