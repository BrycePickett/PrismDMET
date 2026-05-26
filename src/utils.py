'''
    QC-dmet: a python implementation of density matrix embedding theory for ab initio quantum chemistry
    Copyright (C) 2015 Sebastian Wouters

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

import os
import sys
import numpy as np
from contextlib import contextmanager, nullcontext


def scf_rdm1_total(mf):
    # Returns 2D (nao, nao) total density regardless of RHF/ROHF/UHF spin shape.
    dm = np.asarray(mf.make_rdm1())
    if dm.ndim == 3:
        return dm[0] + dm[1]
    return dm


def scf_veff_total(mf, dm_raw):
    # Returns 2D (nao, nao) total Veff. Pass raw make_rdm1() output so PySCF
    # applies correct per-spin exchange before summing over spins.
    veff = np.asarray(mf.get_veff(dm=dm_raw))
    if veff.ndim == 3:
        return veff[0] + veff[1]
    return veff


@contextmanager
def silent_stdout():
    '''
    Context manager that safely suppresses stdout (C-level and Python-level).
    Restores stdout correctly even if an exception is raised inside the block.

    Usage:
        with silent_stdout():
            noisy_library_call()
    '''
    sys.stdout.flush()
    old_fd = os.dup(sys.stdout.fileno())
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, sys.stdout.fileno())
        yield
    finally:
        sys.stdout.flush()
        os.dup2(old_fd, sys.stdout.fileno())
        os.close(old_fd)
        os.close(devnull)
