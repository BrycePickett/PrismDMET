"""
Auxiliary basis sets for density-fitted QM/MM calculations.

Stores raw Gaussian basis strings (def2-universal-jkfit, valence-only)
compatible with GTH pseudopotentials. Use QMMMCluster.auxbasis_dict() to
obtain a mol.auxbasis-ready dict keyed by PySCF atom labels.
"""

_CU2O_JKFIT_O = '''
O    S
    11.807759100           1.0000000
O    S
    6.1827814000          1.0000000
O    S
    3.3709061000          1.0000000
O    S
    1.9042805000          1.0000000
O    S
    1.1085447000          1.0000000
O    S
    0.66098860000         1.0000000
O    S
    0.40108140000         1.0000000
O    S
    0.24597690000         1.0000000
O    S
    0.15139390000         1.0000000
O    P
    5.4848863000          1.0000000
O    P
    2.9732983000          1.0000000
O    P
    1.4735260000          1.0000000
O    P
    0.73603410000         1.0000000
O    P
    0.36974140000         1.0000000
O    P
    0.18637210000         1.0000000
O    P
    0.09499060000         1.0000000
O    D
    2.3304365000          1.0000000
O    D
    0.93282670000         1.0000000
O    D
    0.37392850000         1.0000000
O    F
    3.0293422000          0.76154791140
O    F
    0.92484900000         0.64810861640
O    G
    1.6934809000          1.0000000
'''

_CU2O_JKFIT_CU = '''
Cu   S
    47.735237700           1.0000000
Cu   S
    24.738302000           1.0000000
Cu   S
    13.222158000           1.0000000
Cu   S
    7.2708750000          1.0000000
Cu   S
    4.1026961000          1.0000000
Cu   S
    2.3686256000          1.0000000
Cu   S
    1.3948246000          1.0000000
Cu   S
    0.83503510000         1.0000000
Cu   S
    0.50645760000         1.0000000
Cu   S
    0.31007380000         1.0000000
Cu   S
    0.19092190000         1.0000000
Cu   S
    0.11778020000         1.0000000
Cu   P
    25.058705500           1.0000000
Cu   P
    11.673631800           1.0000000
Cu   P
    5.5521750000          1.0000000
Cu   P
    2.6883929000          1.0000000
Cu   P
    1.3212553000          1.0000000
Cu   P
    0.65701600000         1.0000000
Cu   P
    0.32948950000         1.0000000
Cu   P
    0.16608410000         1.0000000
Cu   P
    0.08386050000         1.0000000
Cu   D
    29.881915100           1.0000000
Cu   D
    14.804201500           1.0000000
Cu   D
    6.4439904000          1.0000000
Cu   D
    3.0570706000          1.0000000
Cu   D
    1.4825524000          1.0000000
Cu   D
    0.73163360000         1.0000000
Cu   D
    0.36565620000         1.0000000
Cu   D
    0.18415490000         1.0000000
Cu   D
    0.09298430000         1.0000000
Cu   F
    21.434763900           1.0000000
Cu   F
    9.4252553000          1.0000000
Cu   F
    4.3346328000          1.0000000
Cu   F
    1.9439700000          1.0000000
Cu   F
    0.87198350000         1.0000000
Cu   F
    0.45251650000         1.0000000
Cu   F
    0.22225260000         1.0000000
Cu   F
    0.11179000000         1.0000000
Cu   G
    4.8310946000          1.0000000
Cu   G
    2.2187485000          1.0000000
Cu   G
    1.0317571000          1.0000000
Cu   G
    0.48179710000         1.0000000
'''

# Parsed basis objects (cached on first access)
_parsed = {}


def _get_parsed():
    if not _parsed:
        import pyscf.gto
        _parsed['Cu'] = pyscf.gto.basis.parse(_CU2O_JKFIT_CU)
        _parsed['O']  = pyscf.gto.basis.parse(_CU2O_JKFIT_O)
    return _parsed


def cu2o_jkfit(element: str):
    """Return the parsed def2-universal-jkfit (valence) auxbasis for Cu or O."""
    parsed = _get_parsed()
    if element not in parsed:
        raise KeyError(
            f"No cu2o_jkfit auxbasis for element '{element}'. "
            f"Available: {list(parsed.keys())}"
        )
    return parsed[element]
