"""Routines from reading and writing LEGEND Data Objects in HDF5 files.
Currently the primary on-disk format for LGDO object is LEGEND HDF5 (LH5) files. IO
is done via the class :class:`.store.LH5Store`. LH5 files can also be
browsed easily in python like any `HDF5 <https://www.hdfgroup.org>`_ file using
`h5py <https://www.h5py.org>`_.
"""

from .iterator import Iterator
from .store import Store, load_dfs, load_nda, ls, show

__all__ = [
    "Iterator",
    "Store",
    "load_dfs",
    "load_nda",
    "ls",
    "show",
]
