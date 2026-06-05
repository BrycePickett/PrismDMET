"""Nelec=0 Pauli repulsion ECPs for QM/MM boundary atoms in PySCF parse_ecp string format."""

_CU_ECP_STR = """\
Cu nelec 0
Cu ul
2       1.000000000            0.000000000
Cu S
2      30.220000000          355.770158000
2      13.190000000           70.865357000
Cu P
2      33.130000000          233.891976000
2      13.220000000           53.947299000
Cu D
2      38.420000000          -31.272165000
2      13.260000000           -2.741104000
"""

_O_ECP_STR = """\
O nelec 0
O ul
2       1.000000000            0.000000000
O S
2      10.44567000            50.77106900
O P
2      18.04517400            -4.903551000
O D
2      8.164798000            -3.312124000
"""

ECP_LIBRARY: dict = {
    'Cu': _CU_ECP_STR,
    'O':  _O_ECP_STR,
}


def get_ecp_string(element: str) -> str:
    """Return the raw ECP string for element; raises KeyError if not in the library."""
    if element not in ECP_LIBRARY:
        raise KeyError(
            f"No ECP parameters found for element '{element}'. "
            f"Available: {list(ECP_LIBRARY.keys())}"
        )
    return ECP_LIBRARY[element]


def has_ecp(element: str) -> bool:
    """Return True if ECP parameters exist for *element*."""
    return element in ECP_LIBRARY
