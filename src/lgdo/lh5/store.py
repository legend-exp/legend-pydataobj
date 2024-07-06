"""
This module implements routines from reading and writing LEGEND Data Objects in
HDF5 files.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import h5py
from numpy.typing import ArrayLike

from .. import types
from . import _serializers, utils

log = logging.getLogger(__name__)


class LH5Store:
    """
    Class to represent a store of LEGEND HDF5 files. The two main methods
    implemented by the class are :meth:`read` and :meth:`write`.

    Examples
    --------
    >>> from lgdo import LH5Store
    >>> store = LH5Store()
    >>> obj, _ = store.read("/geds/waveform", "file.lh5")
    >>> type(obj)
    lgdo.waveformtable.WaveformTable
    """

    def __init__(self, base_path: str = "", keep_open: bool = False) -> None:
        """
        Parameters
        ----------
        base_path
            directory path to prepend to LH5 files.
        keep_open
            whether to keep files open by storing the :mod:`h5py` objects as
            class attributes.
        """
        self.base_path = "" if base_path == "" else utils.expand_path(base_path)
        self.keep_open = keep_open
        self.files = {}

    def gimme_file(self, lh5_file: str | h5py.File, mode: str = "r") -> h5py.File:
        """Returns a :mod:`h5py` file object from the store or creates a new one.

        Parameters
        ----------
        lh5_file
            LH5 file name.
        mode
            mode in which to open file. See :class:`h5py.File` documentation.
        """
        if isinstance(lh5_file, h5py.File):
            return lh5_file

        if mode == "r":
            lh5_file = utils.expand_path(lh5_file, base_path=self.base_path)

        if lh5_file in self.files:
            return self.files[lh5_file]

        if self.base_path != "":
            full_path = os.path.join(self.base_path, lh5_file)
        else:
            full_path = lh5_file

        if mode != "r":
            directory = os.path.dirname(full_path)
            if directory != "" and not os.path.exists(directory):
                log.debug(f"making path {directory}")
                os.makedirs(directory)

        if mode == "r" and not os.path.exists(full_path):
            msg = f"file {full_path} not found"
            raise FileNotFoundError(msg)

        if mode != "r" and os.path.exists(full_path):
            log.debug(f"opening existing file {full_path} in mode '{mode}'")

        h5f = h5py.File(full_path, mode)

        if self.keep_open:
            self.files[lh5_file] = h5f

        return h5f

    def gimme_group(
        self,
        group: str | h5py.Group,
        base_group: h5py.Group,
        grp_attrs: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> h5py.Group:
        """
        Returns an existing :class:`h5py` group from a base group or creates a new one.

        See Also
        --------
        .lh5.utils.get_h5_group
        """
        return utils.get_h5_group(group, base_group, grp_attrs, overwrite)

    def get_buffer(
        self,
        name: str,
        lh5_file: str | h5py.File | Sequence[str | h5py.File],
        size: int | None = None,
        field_mask: Mapping[str, bool] | Sequence[str] | None = None,
    ) -> types.LGDO:
        """Returns an LH5 object appropriate for use as a pre-allocated buffer
        in a read loop. Sets size to `size` if object has a size.
        """
        obj, n_rows = self.read(name, lh5_file, n_rows=0, field_mask=field_mask)
        if hasattr(obj, "resize") and size is not None:
            obj.resize(new_size=size)
        return obj

    def read(
        self,
        name: str,
        lh5_file: str | h5py.File | Sequence[str | h5py.File],
        start_row: int = 0,
        n_rows: int = sys.maxsize,
        idx: ArrayLike = None,
        use_h5idx: bool = False,
        field_mask: Mapping[str, bool] | Sequence[str] | None = None,
        obj_buf: types.LGDO = None,
        obj_buf_start: int = 0,
        decompress: bool = True,
    ) -> tuple[types.LGDO, int]:
        """Read LH5 object data from a file in the store.

        See Also
        --------
        .lh5.core.read
        """
        # grab files from store
        if not isinstance(lh5_file, (str, h5py.File)):
            lh5_obj = [self.gimme_file(f, "r")[name] for f in list(lh5_file)]
        else:
            lh5_obj = self.gimme_file(lh5_file, "r")[name]

        return _serializers._h5_read_lgdo(
            lh5_obj,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            field_mask=field_mask,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
            decompress=decompress,
        )

    def write(
        self,
        obj: types.LGDO,
        name: str,
        lh5_file: str | h5py.File,
        group: str | h5py.Group = "/",
        start_row: int = 0,
        n_rows: int | None = None,
        wo_mode: str = "append",
        write_start: int = 0,
        **h5py_kwargs,
    ) -> None:
        """Write an LGDO into an LH5 file.

        See Also
        --------
        .lh5.core.write
        """
        if wo_mode == "write_safe":
            wo_mode = "w"
        if wo_mode == "append":
            wo_mode = "a"
        if wo_mode == "overwrite":
            wo_mode = "o"
        if wo_mode == "overwrite_file":
            wo_mode = "of"
            write_start = 0
        if wo_mode == "append_column":
            wo_mode = "ac"
        if wo_mode not in ["w", "a", "o", "of", "ac"]:
            msg = f"unknown wo_mode '{wo_mode}'"
            raise ValueError(msg)

        # "mode" is for the h5df.File and wo_mode is for this function
        # In hdf5, 'a' is really "modify" -- in addition to appending, you can
        # change any object in the file. So we use file:append for
        # write_object:overwrite.
        mode = "w" if wo_mode == "of" else "a"

        return _serializers._h5_write_lgdo(
            obj,
            name,
            self.gimme_file(lh5_file, mode=mode),
            group=group,
            start_row=start_row,
            n_rows=n_rows,
            wo_mode=wo_mode,
            write_start=write_start,
            **h5py_kwargs,
        )

    def read_n_rows(self, name: str, lh5_file: str | h5py.File) -> int | None:
        """Look up the number of rows in an Array-like object called `name` in `lh5_file`.

        Return ``None`` if it is a :class:`.Scalar` or a :class:`.Struct`.
        """
        return utils.read_n_rows(name, self.gimme_file(lh5_file, "r"))
