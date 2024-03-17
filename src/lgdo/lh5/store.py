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
import numpy as np
from numpy.typing import ArrayLike

from .. import compression as compress
from ..compression import WaveformCodec
from ..types import (
    LGDO,
    Array,
    ArrayOfEncodedEqualSizedArrays,
    Scalar,
    Struct,
    VectorOfEncodedVectors,
    VectorOfVectors,
    WaveformTable,
)
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
    ) -> LGDO:
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
        obj_buf: LGDO = None,
        obj_buf_start: int = 0,
        decompress: bool = True,
    ) -> tuple[LGDO, int]:
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
        obj: LGDO,
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
        log.debug(
            f"writing {obj!r}[{start_row}:{n_rows}] as "
            f"{lh5_file}:{group}/{name}[{write_start}:], "
            f"mode = {wo_mode}, h5py_kwargs = {h5py_kwargs}"
        )

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
        lh5_file = self.gimme_file(lh5_file, mode=mode)
        group = self.gimme_group(group, lh5_file)
        if wo_mode == "w" and name in group:
            msg = f"can't overwrite '{name}' in wo_mode 'write_safe'"
            raise RuntimeError(msg)

        # struct or table or waveform table
        if isinstance(obj, Struct):
            # In order to append a column, we need to update the
            # `table{old_fields}` value in `group.attrs['datatype"]` to include
            # the new fields.  One way to do this is to override
            # `obj.attrs["datatype"]` to include old and new fields. Then we
            # can write the fields to the table as normal.
            if wo_mode == "ac":
                old_group = self.gimme_group(name, group)
                datatype, shape, fields = utils.parse_datatype(
                    old_group.attrs["datatype"]
                )
                if datatype not in ["table", "struct"]:
                    msg = f"Trying to append columns to an object of type {datatype}"
                    raise RuntimeError(msg)

                # If the mode is `append_column`, make sure we aren't appending
                # a table that has a column of the same name as in the existing
                # table. Also make sure that the field we are adding has the
                # same size
                if len(list(set(fields).intersection(set(obj.keys())))) != 0:
                    msg = f"Can't append {list(set(fields).intersection(set(obj.keys())))} column(s) to a table with the same field(s)"
                    raise ValueError(msg)
                # It doesn't matter what key we access, as all fields in the old table have the same size
                if old_group[next(iter(old_group.keys()))].size != obj.size:
                    msg = f"Table sizes don't match. Trying to append column of size {obj.size} to a table of size {old_group[next(iter(old_group.keys()))].size}."
                    raise ValueError(msg)

                # Now we can append the obj.keys() to the old fields, and then update obj.attrs.
                fields.extend(list(obj.keys()))
                obj.attrs.pop("datatype")
                obj.attrs["datatype"] = "table" + "{" + ",".join(fields) + "}"

            group = self.gimme_group(
                name,
                group,
                grp_attrs=obj.attrs,
                overwrite=(wo_mode in ["o", "ac"]),
            )
            # If the mode is overwrite, then we need to peek into the file's
            # table's existing fields.  If we are writing a new table to the
            # group that does not contain an old field, we should delete that
            # old field from the file
            if wo_mode == "o":
                # Find the old keys in the group that are not present in the
                # new table's keys, then delete them
                for key in list(set(group.keys()) - set(obj.keys())):
                    log.debug(f"{key} is not present in new table, deleting field")
                    del group[key]

            for field in obj:
                # eventually compress waveform table values with LGDO's
                # custom codecs before writing
                # if waveformtable.values.attrs["compression"] is NOT a
                # WaveformCodec, just leave it there
                obj_fld = None
                if (
                    isinstance(obj, WaveformTable)
                    and field == "values"
                    and not isinstance(obj.values, VectorOfEncodedVectors)
                    and not isinstance(obj.values, ArrayOfEncodedEqualSizedArrays)
                    and "compression" in obj.values.attrs
                    and isinstance(obj.values.attrs["compression"], WaveformCodec)
                ):
                    codec = obj.values.attrs["compression"]
                    obj_fld = compress.encode(obj.values, codec=codec)
                else:
                    obj_fld = obj[field]

                # Convert keys to string for dataset names
                f = str(field)
                self.write(
                    obj_fld,
                    f,
                    lh5_file,
                    group=group,
                    start_row=start_row,
                    n_rows=n_rows,
                    wo_mode=wo_mode,
                    write_start=write_start,
                    **h5py_kwargs,
                )
            return

        # scalars
        if isinstance(obj, Scalar):
            if name in group:
                if wo_mode in ["o", "a"]:
                    log.debug(f"overwriting {name} in {group}")
                    del group[name]
                else:
                    msg = f"tried to overwrite {name} in {group} for wo_mode {wo_mode}"
                    raise RuntimeError(msg)
            ds = group.create_dataset(name, shape=(), data=obj.value)
            ds.attrs.update(obj.attrs)

            return

        # vector of encoded vectors
        if isinstance(obj, (VectorOfEncodedVectors, ArrayOfEncodedEqualSizedArrays)):
            group = self.gimme_group(
                name, group, grp_attrs=obj.attrs, overwrite=(wo_mode == "o")
            )

            # ask not to further compress flattened_data, it is already compressed!
            obj.encoded_data.flattened_data.attrs["compression"] = None

            self.write(
                obj.encoded_data,
                "encoded_data",
                lh5_file,
                group=group,
                start_row=start_row,
                n_rows=n_rows,
                wo_mode=wo_mode,
                write_start=write_start,
                **h5py_kwargs,
            )

            self.write(
                obj.decoded_size,
                "decoded_size",
                lh5_file,
                group=group,
                start_row=start_row,
                n_rows=n_rows,
                wo_mode=wo_mode,
                write_start=write_start,
                **h5py_kwargs,
            )

        # vector of vectors
        elif isinstance(obj, VectorOfVectors):
            group = self.gimme_group(
                name, group, grp_attrs=obj.attrs, overwrite=(wo_mode == "o")
            )
            if (
                n_rows is None
                or n_rows > obj.cumulative_length.nda.shape[0] - start_row
            ):
                n_rows = obj.cumulative_length.nda.shape[0] - start_row

            # if appending we need to add an appropriate offset to the
            # cumulative lengths as appropriate for the in-file object
            offset = 0  # declare here because we have to subtract it off at the end
            if (wo_mode in ("a", "o")) and "cumulative_length" in group:
                len_cl = len(group["cumulative_length"])
                if wo_mode == "a":
                    write_start = len_cl
                if len_cl > 0:
                    offset = group["cumulative_length"][write_start - 1]

            # First write flattened_data array. Only write rows with data.
            fd_start = 0 if start_row == 0 else obj.cumulative_length.nda[start_row - 1]
            fd_n_rows = obj.cumulative_length.nda[start_row + n_rows - 1] - fd_start
            self.write(
                obj.flattened_data,
                "flattened_data",
                lh5_file,
                group=group,
                start_row=fd_start,
                n_rows=fd_n_rows,
                wo_mode=wo_mode,
                write_start=offset,
                **h5py_kwargs,
            )

            # now offset is used to give appropriate in-file values for
            # cumulative_length. Need to adjust it for start_row
            if start_row > 0:
                offset -= obj.cumulative_length.nda[start_row - 1]

            # Add offset to obj.cumulative_length itself to avoid memory allocation.
            # Then subtract it off after writing! (otherwise it will be changed
            # upon return)
            cl_dtype = obj.cumulative_length.nda.dtype.type
            obj.cumulative_length.nda += cl_dtype(offset)

            self.write(
                obj.cumulative_length,
                "cumulative_length",
                lh5_file,
                group=group,
                start_row=start_row,
                n_rows=n_rows,
                wo_mode=wo_mode,
                write_start=write_start,
                **h5py_kwargs,
            )
            obj.cumulative_length.nda -= cl_dtype(offset)

            return

        # if we get this far, must be one of the Array types
        elif isinstance(obj, Array):
            if n_rows is None or n_rows > obj.nda.shape[0] - start_row:
                n_rows = obj.nda.shape[0] - start_row

            nda = obj.nda[start_row : start_row + n_rows]

            # hack to store bools as uint8 for c / Julia compliance
            if nda.dtype.name == "bool":
                nda = nda.astype(np.uint8)

            # need to create dataset from ndarray the first time for speed
            # creating an empty dataset and appending to that is super slow!
            if (wo_mode != "a" and write_start == 0) or name not in group:
                # this is needed in order to have a resizable (in the first
                # axis) data set, i.e. rows can be appended later
                # NOTE: this automatically turns chunking on!
                maxshape = (None,) + nda.shape[1:]
                h5py_kwargs.setdefault("maxshape", maxshape)

                if wo_mode == "o" and name in group:
                    log.debug(f"overwriting {name} in {group}")
                    del group[name]

                # set default compression options
                for k, v in DEFAULT_HDF5_SETTINGS.items():
                    h5py_kwargs.setdefault(k, v)

                # compress using the 'compression' LGDO attribute, if available
                if "compression" in obj.attrs:
                    comp_algo = obj.attrs["compression"]
                    if isinstance(comp_algo, dict):
                        h5py_kwargs |= obj.attrs["compression"]
                    else:
                        h5py_kwargs["compression"] = obj.attrs["compression"]

                # and even the 'hdf5_settings' one, preferred
                if "hdf5_settings" in obj.attrs:
                    h5py_kwargs |= obj.attrs["hdf5_settings"]

                # create HDF5 dataset
                ds = group.create_dataset(name, data=nda, **h5py_kwargs)

                # attach HDF5 dataset attributes, but not "compression"!
                _attrs = obj.getattrs(datatype=True)
                _attrs.pop("compression", None)
                _attrs.pop("hdf5_settings", None)
                ds.attrs.update(_attrs)
                return

            # Now append or overwrite
            ds = group[name]
            if not isinstance(ds, h5py.Dataset):
                msg = (
                    f"existing HDF5 object '{name}' in group '{group}'"
                    " is not a dataset! Cannot overwrite or append"
                )
                raise RuntimeError(msg)

            old_len = ds.shape[0]
            if wo_mode == "a":
                write_start = old_len
            add_len = write_start + nda.shape[0] - old_len
            ds.resize(old_len + add_len, axis=0)
            ds[write_start:] = nda
            return

        else:
            msg = f"do not know how to write '{name}' of type '{type(obj).__name__}'"
            raise RuntimeError(msg)

    def read_n_rows(self, name: str, lh5_file: str | h5py.File) -> int | None:
        """Look up the number of rows in an Array-like object called `name` in `lh5_file`.

        Return ``None`` if it is a :class:`.Scalar` or a :class:`.Struct`."""
        # this is basically a stripped down version of read_object
        return utils.read_n_rows(name, self.gimme_file(lh5_file, "r"))
