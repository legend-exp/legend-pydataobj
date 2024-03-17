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

DEFAULT_HDF5_SETTINGS: dict[str, ...] = {"shuffle": True, "compression": "gzip"}
DEFAULT_HDF5_COMPRESSION = None


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
        Returns an existing :class:`h5py` group from a base group or creates a
        new one. Can also set (or replace) group attributes.

        Parameters
        ----------
        group
            name of the HDF5 group.
        base_group
            HDF5 group to be used as a base.
        grp_attrs
            HDF5 group attributes.
        overwrite
            whether overwrite group attributes, ignored if `grp_attrs` is
            ``None``.
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
        return utils.get_buffer(name, lh5_file, size, field_mask)

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
            lh5_file = [self.gimme_file(f, "r") for f in list(lh5_file)]

        return _serializers._h5_read_lgdo(
            name,
            lh5_file,
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

        If the `obj` :class:`.LGDO` has a `compression` attribute, its value is
        interpreted as the algorithm to be used to compress `obj` before
        writing to disk. The type of `compression` can be:

        string, kwargs dictionary, hdf5plugin filter
          interpreted as the name of a built-in or custom `HDF5 compression
          filter <https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline>`_
          (``"gzip"``, ``"lzf"``, :mod:`hdf5plugin` filter object etc.) and
          passed directly to :meth:`h5py.Group.create_dataset`.

        :class:`.WaveformCodec` object
          If `obj` is a :class:`.WaveformTable` and ``obj.values`` holds the
          attribute, compress ``values`` using this algorithm. More
          documentation about the supported waveform compression algorithms at
          :mod:`.lgdo.compression`.

        If the `obj` :class:`.LGDO` has a `hdf5_settings` attribute holding a
        dictionary, it is interpreted as a list of keyword arguments to be
        forwarded directly to :meth:`h5py.Group.create_dataset` (exactly like
        the first format of `compression` above). This is the preferred way to
        specify HDF5 dataset options such as chunking etc. If compression
        options are specified, they take precedence over those set with the
        `compression` attribute.

        Note
        ----------
        The `compression` LGDO attribute takes precedence over the default HDF5
        compression settings. The `hdf5_settings` attribute takes precedence
        over `compression`. These attributes are not written to disk.

        Note
        ----------
        HDF5 compression is skipped for the `encoded_data.flattened_data`
        dataset of :class:`.VectorOfEncodedVectors` and
        :class:`.ArrayOfEncodedEqualSizedArrays`.

        Parameters
        ----------
        obj
            LH5 object. if object is array-like, writes `n_rows` starting from
            `start_row` in `obj`.
        name
            name of the object in the output HDF5 file.
        lh5_file
            HDF5 file name or :class:`h5py.File` object.
        group
            HDF5 group name or :class:`h5py.Group` object in which `obj` should
            be written.
        start_row
            first row in `obj` to be written.
        n_rows
            number of rows in `obj` to be written.
        wo_mode
            - ``write_safe`` or ``w``: only proceed with writing if the
              object does not already exist in the file.
            - ``append`` or ``a``: append along axis 0 (the first dimension)
              of array-like objects and array-like subfields of structs.
              :class:`~.lgdo.scalar.Scalar` objects get overwritten.
            - ``overwrite`` or ``o``: replace data in the file if present,
              starting from `write_start`. Note: overwriting with `write_start` =
              end of array is the same as ``append``.
            - ``overwrite_file`` or ``of``: delete file if present prior to
              writing to it. `write_start` should be 0 (its ignored).
            - ``append_column`` or ``ac``: append columns from an
              :class:`~.lgdo.table.Table` `obj` only if there is an existing
              :class:`~.lgdo.table.Table` in the `lh5_file` with the same
              `name` and :class:`~.lgdo.table.Table.size`. If the sizes don't
              match, or if there are matching fields, it errors out.
        write_start
            row in the output file (if already existing) to start overwriting
            from.
        **h5py_kwargs
            additional keyword arguments forwarded to
            :meth:`h5py.Group.create_dataset` to specify, for example, an HDF5
            compression filter to be applied before writing non-scalar
            datasets. **Note: `compression` Ignored if compression is specified
            as an `obj` attribute.**
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

        Return ``None`` if it is a :class:`.Scalar` or a :class:`.Struct`."""
        return utils.read_n_rows(name, self.gimme_file(lh5_file, "r"))
