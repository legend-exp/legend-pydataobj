"""
This module implements routines from reading and writing LEGEND Data Objects in
HDF5 files.
"""
from __future__ import annotations

import logging
import os
import sys
from bisect import bisect_left
from collections import defaultdict
from typing import Any, Union

import h5py
import numba as nb
import numpy as np

from .. import compression as compress
from ..compression import WaveformCodec
from ..types import (
    Array,
    ArrayOfEncodedEqualSizedArrays,
    ArrayOfEqualSizedArrays,
    FixedSizeArray,
    Scalar,
    Struct,
    Table,
    VectorOfEncodedVectors,
    VectorOfVectors,
    WaveformTable,
)
from .utils import expand_path, parse_datatype

LGDO = Union[Array, Scalar, Struct, VectorOfVectors]

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
        self.base_path = "" if base_path == "" else expand_path(base_path)
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
            lh5_file = expand_path(lh5_file, base_path=self.base_path)
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
        if not isinstance(group, h5py.Group):
            if group in base_group:
                group = base_group[group]
            else:
                group = base_group.create_group(group)
                if grp_attrs is not None:
                    group.attrs.update(grp_attrs)
                return group
        if (
            grp_attrs is not None
            and len(set(grp_attrs.items()) ^ set(group.attrs.items())) > 0
        ):
            if not overwrite:
                msg = "grp_attrs != group.attrs but overwrite not set"
                raise RuntimeError(msg)

            log.debug(f"overwriting {group}.attrs...")
            for key in group.attrs:
                group.attrs.pop(key)
            group.attrs.update(grp_attrs)

        return group

    def get_buffer(
        self,
        name: str,
        lh5_file: str | h5py.File | list[str | h5py.File],
        size: int | None = None,
        field_mask: dict[str, bool] | list[str] | tuple[str] | None = None,
    ) -> LGDO:
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
        lh5_file: str | h5py.File | list[str | h5py.File],
        start_row: int = 0,
        n_rows: int = sys.maxsize,
        idx: np.ndarray | list | tuple | list[np.ndarray | list | tuple] = None,
        use_h5idx: bool = False,
        field_mask: dict[str, bool] | list[str] | tuple[str] | None = None,
        obj_buf: LGDO = None,
        obj_buf_start: int = 0,
        decompress: bool = True,
    ) -> tuple[LGDO, int]:
        """Read LH5 object data from a file.

        Use the ``idx`` parameter to read out particular rows of the data. The ``use_h5idx`` flag
        controls whether *only* those rows are read from disk or if the rows are indexed after reading
        the entire object. Reading individual rows can be orders of magnitude slower than reading
        the whole object and then indexing the desired rows. The default behavior (``use_h5idx=False``)
        is to use slightly more memory for a much faster read. See
        `legend-pydataobj #29 <https://github.com/legend-exp/legend-pydataobj/issues/29>`_
        for additional information.

        Parameters
        ----------
        name
            Name of the LH5 object to be read (including its group path).
        lh5_file
            The file(s) containing the object to be read out. If a list of
            files, array-like object data will be concatenated into the output
            object.
        start_row
            Starting entry for the object read (for array-like objects). For a
            list of files, only applies to the first file.
        n_rows
            The maximum number of rows to read (for array-like objects). The
            actual number of rows read will be returned as one of the return
            values (see below).
        idx
            For NumPy-style "fancying indexing" for the read to select only some
            rows, e.g. after applying some cuts to particular columns.
            Only selection along the first axis is supported, so tuple arguments
            must be one-tuples.  If `n_rows` is not false, `idx` will be truncated to
            `n_rows` before reading. To use with a list of files, can pass in a list of
            `idx`'s (one for each file) or use a long contiguous list (e.g. built from a previous
            identical read). If used in conjunction with `start_row` and `n_rows`,
            will be sliced to obey those constraints, where `n_rows` is
            interpreted as the (max) number of *selected* values (in `idx`) to be
            read out. Note that the ``use_h5idx`` parameter controls some behaviour of the
            read and that the default behavior (``use_h5idx=False``) prioritizes speed over
            a small memory penalty.
        use_h5idx
            ``True`` will directly pass the ``idx`` parameter to the underlying
            ``h5py`` call such that only the selected rows are read directly into memory,
            which conserves memory at the cost of speed. There can be a significant penalty
            to speed for larger files (1 - 2 orders of magnitude longer time).
            ``False`` (default) will read the entire object into memory before
            performing the indexing. The default is much faster but requires additional memory,
            though a relatively small amount in the typical use case. It is recommended to
            leave this parameter as its default.
        field_mask
            For tables and structs, determines which fields get written out.
            Only applies to immediate fields of the requested objects. If a dict
            is used, a default dict will be made with the default set to the
            opposite of the first element in the dict. This way if one specifies
            a few fields at ``False``, all but those fields will be read out,
            while if one specifies just a few fields as ``True``, only those
            fields will be read out. If a list is provided, the listed fields
            will be set to ``True``, while the rest will default to ``False``.
        obj_buf
            Read directly into memory provided in `obj_buf`. Note: the buffer
            will be expanded to accommodate the data requested. To maintain the
            buffer length, send in ``n_rows = len(obj_buf)``.
        obj_buf_start
            Start location in ``obj_buf`` for read. For concatenating data to
            array-like objects.
        decompress
            Decompress data encoded with LGDO's compression routines right
            after reading. The option has no effect on data encoded with HDF5
            built-in filters, which is always decompressed upstream by HDF5.


        Returns
        -------
        (object, n_rows_read)
            `object` is the read-out object `n_rows_read` is the number of rows
            successfully read out. Essential for arrays when the amount of data
            is smaller than the object buffer.  For scalars and structs
            `n_rows_read` will be``1``. For tables it is redundant with
            ``table.loc``.
        """
        # Handle list-of-files recursively
        if not isinstance(lh5_file, (str, h5py.File)):
            lh5_file = list(lh5_file)
            n_rows_read = 0

            # to know whether we are reading in a list of files.
            # this is part of the fix for reading data by idx
            # (see https://github.com/legend-exp/legend-pydataobj/issues/29)
            # so that we only make a copy of the data if absolutely necessary
            # or if we can read the data from file without having to make a copy
            self.in_file_loop = True

            for i, h5f in enumerate(lh5_file):
                if isinstance(idx, list) and len(idx) > 0 and not np.isscalar(idx[0]):
                    # a list of lists: must be one per file
                    idx_i = idx[i]
                elif idx is not None:
                    # make idx a proper tuple if it's not one already
                    if not (isinstance(idx, tuple) and len(idx) == 1):
                        idx = (idx,)
                    # idx is a long continuous array
                    n_rows_i = self.read_n_rows(name, h5f)
                    # find the length of the subset of idx that contains indices
                    # that are less than n_rows_i
                    n_rows_to_read_i = bisect_left(idx[0], n_rows_i)
                    # now split idx into idx_i and the remainder
                    idx_i = (idx[0][:n_rows_to_read_i],)
                    idx = (idx[0][n_rows_to_read_i:] - n_rows_i,)
                else:
                    idx_i = None
                n_rows_i = n_rows - n_rows_read

                # maybe someone passed in a list of len==1?
                if i == (len(lh5_file) - 1):
                    self.in_file_loop = False

                obj_buf, n_rows_read_i = self.read(
                    name,
                    lh5_file[i],
                    start_row=start_row,
                    n_rows=n_rows_i,
                    idx=idx_i,
                    use_h5idx=use_h5idx,
                    field_mask=field_mask,
                    obj_buf=obj_buf,
                    obj_buf_start=obj_buf_start,
                    decompress=decompress,
                )

                n_rows_read += n_rows_read_i
                if n_rows_read >= n_rows or obj_buf is None:
                    return obj_buf, n_rows_read
                start_row = 0
                obj_buf_start += n_rows_read_i

            self.in_file_loop = False

            return obj_buf, n_rows_read

        # get the file from the store
        h5f = self.gimme_file(lh5_file, "r")
        if not h5f or name not in h5f:
            msg = f"'{name}' not in {h5f.filename}"
            raise KeyError(msg)

        log.debug(
            f"reading {h5f.filename}:{name}[{start_row}:{n_rows}], decompress = {decompress}, "
            + (f" with field mask {field_mask}" if field_mask else "")
        )

        # make idx a proper tuple if it's not one already
        if not (isinstance(idx, tuple) and len(idx) == 1) and idx is not None:
            idx = (idx,)

        # get the object's datatype
        if "datatype" not in h5f[name].attrs:
            msg = f"'{name}' in file {lh5_file} is missing the datatype attribute"
            raise RuntimeError(msg)

        datatype = h5f[name].attrs["datatype"]
        datatype, shape, elements = parse_datatype(datatype)

        # check field_mask and make it a default dict
        if datatype in ("struct", "table"):
            if field_mask is None:
                field_mask = defaultdict(lambda: True)
            elif isinstance(field_mask, dict):
                default = True
                if len(field_mask) > 0:
                    default = not field_mask[next(iter(field_mask.keys()))]
                field_mask = defaultdict(lambda: default, field_mask)
            elif isinstance(field_mask, (list, tuple)):
                field_mask = defaultdict(bool, {field: True for field in field_mask})
            elif not isinstance(field_mask, defaultdict):
                msg = "bad field_mask of type"
                raise RuntimeError(msg, type(field_mask).__name__)
        elif field_mask is not None:
            msg = f"datatype {datatype} does not accept a field_mask"
            raise RuntimeError(msg)

        # Scalar
        # scalars are dim-0 datasets
        if datatype == "scalar":
            value = h5f[name][()]
            if elements == "bool":
                value = np.bool_(value)
            if obj_buf is not None:
                obj_buf.value = value
                obj_buf.attrs.update(h5f[name].attrs)
                return obj_buf, 1

            return Scalar(value=value, attrs=h5f[name].attrs), 1

        # Struct
        # recursively build a struct, return as a dictionary
        if datatype == "struct":
            # ignore obj_buf.
            # TODO: could append new fields or overwrite/concat to existing
            # fields. If implemented, get_buffer() above should probably also
            # (optionally?) prep buffers for each field
            if obj_buf is not None:
                msg = "obj_buf not implemented for LGOD Structs"
                raise NotImplementedError(msg)

            # loop over fields and read
            obj_dict = {}
            for field in elements:
                if not field_mask[field]:
                    continue
                # TODO: it's strange to pass start_row, n_rows, idx to struct
                # fields. If they all had shared indexing, they should be in a
                # table... Maybe should emit a warning? Or allow them to be
                # dicts keyed by field name?
                if "int_keys" in h5f[name].attrs:
                    if dict(h5f[name].attrs)["int_keys"]:
                        f = int(field)
                else:
                    f = str(field)
                obj_dict[f], _ = self.read(
                    name + "/" + field,
                    h5f,
                    start_row=start_row,
                    n_rows=n_rows,
                    idx=idx,
                    use_h5idx=use_h5idx,
                    decompress=decompress,
                )
            # modify datatype in attrs if a field_mask was used
            attrs = dict(h5f[name].attrs)
            if field_mask is not None:
                selected_fields = []
                for field in elements:
                    if field_mask[field]:
                        selected_fields.append(field)
                attrs["datatype"] = "struct" + "{" + ",".join(selected_fields) + "}"
            return Struct(obj_dict=obj_dict, attrs=attrs), 1

        # Below here is all array-like types. So trim idx if needed
        if idx is not None:
            # check if idx is just an ordered list of the integers if so can ignore
            if (idx[0] == np.arange(0, len(idx[0]), 1)).all():
                if n_rows > len(idx[0]):
                    n_rows = len(idx[0])
                idx = None
            else:
                # chop off indices < start_row
                i_first_valid = bisect_left(idx[0], start_row)
                idxa = idx[0][i_first_valid:]
                # don't readout more than n_rows indices
                idx = (idxa[:n_rows],)  # works even if n_rows > len(idxa)

        # Table or WaveformTable
        if datatype == "table":
            col_dict = {}

            # read out each of the fields
            rows_read = []
            for field in elements:
                if not field_mask[field]:
                    continue

                fld_buf = None
                if obj_buf is not None:
                    if not isinstance(obj_buf, Table) or field not in obj_buf:
                        msg = f"obj_buf for LGDO Table '{name}' not formatted correctly"
                        raise ValueError(msg)

                    fld_buf = obj_buf[field]

                col_dict[field], n_rows_read = self.read(
                    name + "/" + field,
                    h5f,
                    start_row=start_row,
                    n_rows=n_rows,
                    idx=idx,
                    use_h5idx=use_h5idx,
                    obj_buf=fld_buf,
                    obj_buf_start=obj_buf_start,
                    decompress=decompress,
                )
                if obj_buf is not None and obj_buf_start + n_rows_read > len(obj_buf):
                    obj_buf.resize(obj_buf_start + n_rows_read)

                rows_read.append(n_rows_read)

            # warn if all columns don't read in the same number of rows
            if len(rows_read) > 0:
                n_rows_read = rows_read[0]
            else:
                n_rows_read = 0
                log.warning(f"Table '{name}' has no subgroups accepted by field mask")

            for n in rows_read[1:]:
                if n != n_rows_read:
                    log.warning(
                        f"Table '{name}' got strange n_rows_read = {n}, {n_rows_read} was expected ({rows_read})"
                    )

            # modify datatype in attrs if a field_mask was used
            attrs = dict(h5f[name].attrs)
            if field_mask is not None:
                selected_fields = []
                for field in elements:
                    if field_mask[field]:
                        selected_fields.append(field)
                attrs["datatype"] = "table" + "{" + ",".join(selected_fields) + "}"

            # fields have been read out, now return a table
            if obj_buf is None:
                # if col_dict contains just 3 objects called t0, dt, and values,
                # return a WaveformTable
                if (
                    len(col_dict) == 3
                    and "t0" in col_dict
                    and "dt" in col_dict
                    and "values" in col_dict
                ):
                    table = WaveformTable(
                        t0=col_dict["t0"], dt=col_dict["dt"], values=col_dict["values"]
                    )
                else:
                    table = Table(col_dict=col_dict, attrs=attrs)

                # set (write) loc to end of tree
                table.loc = n_rows_read
                return table, n_rows_read

            # We have read all fields into the object buffer. Run
            # checks: All columns should be the same size. So update
            # table's size as necessary, warn if any mismatches are found
            obj_buf.resize(do_warn=True)
            # set (write) loc to end of tree
            obj_buf.loc = obj_buf_start + n_rows_read
            # check attributes
            if set(obj_buf.attrs.keys()) != set(attrs.keys()):
                msg = (
                    f"attrs mismatch. obj_buf.attrs: "
                    f"{obj_buf.attrs}, h5f[{name}].attrs: {attrs}"
                )
                raise RuntimeError(msg)
            return obj_buf, n_rows_read

        # ArrayOfEncodedEqualSizedArrays and VectorOfEncodedVectors
        for cond, enc_lgdo in [
            (
                datatype == "array_of_encoded_equalsized_arrays",
                ArrayOfEncodedEqualSizedArrays,
            ),
            (elements.startswith("encoded_array"), VectorOfEncodedVectors),
        ]:
            if cond:
                if (
                    not decompress
                    and obj_buf is not None
                    and not isinstance(obj_buf, enc_lgdo)
                ):
                    msg = f"obj_buf for '{name}' not a {enc_lgdo}"
                    raise ValueError(msg)

                # read out decoded_size, either a Scalar or an Array
                decoded_size_buf = encoded_data_buf = None
                if obj_buf is not None and not decompress:
                    decoded_size_buf = obj_buf.decoded_size
                    encoded_data_buf = obj_buf.encoded_data

                decoded_size, _ = self.read(
                    f"{name}/decoded_size",
                    h5f,
                    start_row=start_row,
                    n_rows=n_rows,
                    idx=idx,
                    use_h5idx=use_h5idx,
                    obj_buf=None if decompress else decoded_size_buf,
                    obj_buf_start=0 if decompress else obj_buf_start,
                )

                # read out encoded_data, a VectorOfVectors
                encoded_data, n_rows_read = self.read(
                    f"{name}/encoded_data",
                    h5f,
                    start_row=start_row,
                    n_rows=n_rows,
                    idx=idx,
                    use_h5idx=use_h5idx,
                    obj_buf=None if decompress else encoded_data_buf,
                    obj_buf_start=0 if decompress else obj_buf_start,
                )

                # return the still encoded data in the buffer object, if there
                if obj_buf is not None and not decompress:
                    return obj_buf, n_rows_read

                # otherwise re-create the encoded LGDO
                rawdata = enc_lgdo(
                    encoded_data=encoded_data,
                    decoded_size=decoded_size,
                    attrs=h5f[name].attrs,
                )

                # already return if no decompression is requested
                if not decompress:
                    return rawdata, n_rows_read

                # if no buffer, decode and return
                if obj_buf is None and decompress:
                    return compress.decode(rawdata), n_rows_read

                # eventually expand provided obj_buf, if too short
                buf_size = obj_buf_start + n_rows_read
                if len(obj_buf) < buf_size:
                    obj_buf.resize(buf_size)

                # use the (decoded object type) buffer otherwise
                if enc_lgdo == ArrayOfEncodedEqualSizedArrays:
                    if not isinstance(obj_buf, ArrayOfEqualSizedArrays):
                        msg = f"obj_buf for decoded '{name}' not an ArrayOfEqualSizedArrays"
                        raise ValueError(msg)

                    compress.decode(rawdata, obj_buf[obj_buf_start:buf_size])

                elif enc_lgdo == VectorOfEncodedVectors:
                    if not isinstance(obj_buf, VectorOfVectors):
                        msg = f"obj_buf for decoded '{name}' not a VectorOfVectors"
                        raise ValueError(msg)

                    # FIXME: not a good idea. an in place decoding version
                    # of decode would be needed to avoid extra memory
                    # allocations
                    for i, wf in enumerate(compress.decode(rawdata)):
                        obj_buf[obj_buf_start + i] = wf

                return obj_buf, n_rows_read

        # VectorOfVectors
        # read out vector of vectors of different size
        if elements.startswith("array"):
            if obj_buf is not None and not isinstance(obj_buf, VectorOfVectors):
                msg = f"obj_buf for '{name}' not a LGDO VectorOfVectors"
                raise ValueError(msg)

            # read out cumulative_length
            cumulen_buf = None if obj_buf is None else obj_buf.cumulative_length
            cumulative_length, n_rows_read = self.read(
                f"{name}/cumulative_length",
                h5f,
                start_row=start_row,
                n_rows=n_rows,
                idx=idx,
                use_h5idx=use_h5idx,
                obj_buf=cumulen_buf,
                obj_buf_start=obj_buf_start,
            )
            # get a view of just what was read out for cleaner code below
            this_cumulen_nda = cumulative_length.nda[
                obj_buf_start : obj_buf_start + n_rows_read
            ]

            if idx is not None and n_rows_read > 0:
                # get the starting indices for each array in flattended data:
                # the starting index for array[i] is cumulative_length[i-1]
                idx2 = (np.asarray(idx[0]).copy() - 1,)
                # re-read cumulative_length with these indices
                # note this will allocate memory for fd_starts!
                fd_start = None
                if idx2[0][0] == -1:
                    idx2 = (idx2[0][1:],)
                    fd_start = 0  # this variable avoids an ndarray append
                fd_starts, fds_n_rows_read = self.read(
                    f"{name}/cumulative_length",
                    h5f,
                    start_row=start_row,
                    n_rows=n_rows,
                    idx=idx2,
                    use_h5idx=use_h5idx,
                )
                fd_starts = fd_starts.nda  # we just need the nda
                if fd_start is None:
                    fd_start = fd_starts[0]

                # compute the length that flattened_data will have after the
                # fancy-indexed read
                fd_n_rows = np.sum(this_cumulen_nda[-len(fd_starts) :] - fd_starts)
                if fd_start == 0:
                    fd_n_rows += this_cumulen_nda[0]

                # now make fd_idx
                fd_idx = np.empty(fd_n_rows, dtype="uint32")
                fd_idx = _make_fd_idx(fd_starts, this_cumulen_nda, fd_idx)

                # Now clean up this_cumulen_nda, to be ready
                # to match the in-memory version of flattened_data. Note: these
                # operations on the view change the original array because they are
                # numpy arrays, not lists.
                this_cumulen_nda[-len(fd_starts) :] -= fd_starts
                np.cumsum(this_cumulen_nda, out=this_cumulen_nda)

            else:
                fd_idx = None

                # determine the start_row and n_rows for the flattened_data readout
                fd_start = 0
                if start_row > 0 and n_rows_read > 0:
                    # need to read out the cumulen sample -before- the first sample
                    # read above in order to get the starting row of the first
                    # vector to read out in flattened_data
                    fd_start = h5f[f"{name}/cumulative_length"][start_row - 1]

                    # check limits for values that will be used subsequently
                    if this_cumulen_nda[-1] < fd_start:
                        log.debug(
                            f"this_cumulen_nda[-1] = {this_cumulen_nda[-1]}, "
                            f"fd_start = {fd_start}, "
                            f"start_row = {start_row}, "
                            f"n_rows_read = {n_rows_read}"
                        )
                        msg = (
                            f"cumulative_length non-increasing between entries "
                            f"{start_row} and {start_row+n_rows_read} ??"
                        )
                        raise RuntimeError(msg)

                # determine the number of rows for the flattened_data readout
                fd_n_rows = this_cumulen_nda[-1] if n_rows_read > 0 else 0

                # Now done with this_cumulen_nda, so we can clean it up to be ready
                # to match the in-memory version of flattened_data. Note: these
                # operations on the view change the original array because they are
                # numpy arrays, not lists.
                #
                # First we need to subtract off the in-file offset for the start of
                # read for flattened_data
                this_cumulen_nda -= fd_start

            # If we started with a partially-filled buffer, add the
            # appropriate offset for the start of the in-memory flattened
            # data for this read.
            fd_buf_start = np.uint32(0)
            if obj_buf_start > 0:
                fd_buf_start = cumulative_length.nda[obj_buf_start - 1]
                this_cumulen_nda += fd_buf_start

            # Now prepare the object buffer if necessary
            fd_buf = None
            if obj_buf is not None:
                fd_buf = obj_buf.flattened_data
                # grow fd_buf if necessary to hold the data
                fdb_size = fd_buf_start + fd_n_rows
                if len(fd_buf) < fdb_size:
                    fd_buf.resize(fdb_size)

            # now read
            flattened_data, dummy_rows_read = self.read(
                f"{name}/flattened_data",
                h5f,
                start_row=fd_start,
                n_rows=fd_n_rows,
                idx=fd_idx,
                use_h5idx=use_h5idx,
                obj_buf=fd_buf,
                obj_buf_start=fd_buf_start,
            )
            if obj_buf is not None:
                return obj_buf, n_rows_read
            return (
                VectorOfVectors(
                    flattened_data=flattened_data,
                    cumulative_length=cumulative_length,
                    attrs=h5f[name].attrs,
                ),
                n_rows_read,
            )

        # Array
        # FixedSizeArray
        # ArrayOfEqualSizedArrays
        # read out all arrays by slicing
        if "array" in datatype:
            if obj_buf is not None and not isinstance(obj_buf, Array):
                msg = f"obj_buf for '{name}' not an LGDO Array"
                raise ValueError(msg)
                obj_buf = None

            # compute the number of rows to read
            # we culled idx above for start_row and n_rows, now we have to apply
            # the constraint of the length of the dataset
            ds_n_rows = h5f[name].shape[0]
            if idx is not None:
                if len(idx[0]) > 0 and idx[0][-1] >= ds_n_rows:
                    log.warning(
                        "idx indexed past the end of the array in the file. Culling..."
                    )
                    n_rows_to_read = bisect_left(idx[0], ds_n_rows)
                    idx = (idx[0][:n_rows_to_read],)
                    if len(idx[0]) == 0:
                        log.warning("idx empty after culling.")
                n_rows_to_read = len(idx[0])
            else:
                n_rows_to_read = ds_n_rows - start_row
            if n_rows_to_read > n_rows:
                n_rows_to_read = n_rows

            # if idx is passed, check if we can make it a slice instead (faster)
            change_idx_to_slice = False

            # prepare the selection for the read. Use idx if available
            if idx is not None:
                # check if idx is empty and convert to slice instead
                if len(idx[0]) == 0:
                    source_sel = np.s_[0:0]
                    change_idx_to_slice = True
                # check if idx is contiguous and increasing
                # if so, convert it to a slice instead (faster)
                elif np.all(np.diff(idx[0]) == 1):
                    source_sel = np.s_[idx[0][0] : idx[0][-1] + 1]
                    change_idx_to_slice = True
                else:
                    source_sel = idx
            else:
                source_sel = np.s_[start_row : start_row + n_rows_to_read]

            # Now read the array
            if obj_buf is not None and n_rows_to_read > 0:
                buf_size = obj_buf_start + n_rows_to_read
                if len(obj_buf) < buf_size:
                    obj_buf.resize(buf_size)
                dest_sel = np.s_[obj_buf_start:buf_size]

                # this is required to make the read of multiple files faster
                # until a better solution found.
                if change_idx_to_slice or idx is None or use_h5idx:
                    h5f[name].read_direct(obj_buf.nda, source_sel, dest_sel)
                else:
                    # it is faster to read the whole object and then do fancy indexing
                    obj_buf.nda[dest_sel] = h5f[name][...][source_sel]

                nda = obj_buf.nda
            elif n_rows == 0:
                tmp_shape = (0,) + h5f[name].shape[1:]
                nda = np.empty(tmp_shape, h5f[name].dtype)
            elif change_idx_to_slice or idx is None or use_h5idx:
                nda = h5f[name][source_sel]
            else:
                # it is faster to read the whole object and then do fancy indexing
                nda = h5f[name][...][source_sel]

                # if reading a list of files recursively, this is given to obj_buf on
                # the first file read. obj_buf needs to be resized and therefore
                # it needs to hold the data itself (not a view of the data).
                # a view is returned by the source_sel indexing, which cannot be resized
                # by ndarray.resize().
                if hasattr(self, "in_file_loop") and self.in_file_loop:
                    nda = np.copy(nda)

            # special handling for bools
            # (c and Julia store as uint8 so cast to bool)
            if elements == "bool":
                nda = nda.astype(np.bool_)

            # Finally, set attributes and return objects
            attrs = h5f[name].attrs
            if obj_buf is None:
                if datatype == "array":
                    return Array(nda=nda, attrs=attrs), n_rows_to_read
                if datatype == "fixedsize_array":
                    return FixedSizeArray(nda=nda, attrs=attrs), n_rows_to_read
                if datatype == "array_of_equalsized_arrays":
                    return (
                        ArrayOfEqualSizedArrays(nda=nda, dims=shape, attrs=attrs),
                        n_rows_to_read,
                    )
            else:
                if set(obj_buf.attrs.keys()) != set(attrs.keys()):
                    msg = (
                        f"attrs mismatch. "
                        f"obj_buf.attrs: {obj_buf.attrs}, "
                        f"h5f[{name}].attrs: {attrs}"
                    )
                    raise RuntimeError(msg)
                return obj_buf, n_rows_to_read

        msg = "don't know how to read datatype {datatype}"
        raise RuntimeError(msg)

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
            - ``append_column`` or ``ac``: append columns from an :class:`~.lgdo.table.Table`
              `obj` only if there is an existing :class:`~.lgdo.table.Table` in the `lh5_file` with
              the same `name` and :class:`~.lgdo.table.Table.size`. If the sizes don't match,
              or if there are matching fields, it errors out.
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
            # In order to append a column, we need to update the `table{old_fields}` value in `group.attrs['datatype"]` to include the new fields.
            # One way to do this is to override `obj.attrs["datatype"]` to include old and new fields. Then we can write the fields to the table as normal.
            if wo_mode == "ac":
                old_group = self.gimme_group(name, group)
                datatype, shape, fields = parse_datatype(old_group.attrs["datatype"])
                if datatype not in ["table", "struct"]:
                    msg = f"Trying to append columns to an object of type {datatype}"
                    raise RuntimeError(msg)

                # If the mode is `append_column`, make sure we aren't appending a table that has a column of the same name as in the existing table
                # Also make sure that the field we are adding has the same size
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
            # If the mode is overwrite, then we need to peek into the file's table's existing fields
            # If we are writing a new table to the group that does not contain an old field, we should delete that old field from the file
            if wo_mode == "o":
                # Find the old keys in the group that are not present in the new table's keys, then delete them
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
        """Look up the number of rows in an Array-like object called `name` in
        `lh5_file`.

        Return ``None`` if it is a :class:`.Scalar` or a :class:`.Struct`."""
        # this is basically a stripped down version of read_object
        h5f = self.gimme_file(lh5_file, "r")
        if not h5f or name not in h5f:
            msg = f"'{name}' not in {lh5_file}"
            raise KeyError(msg)

        # get the datatype
        if "datatype" not in h5f[name].attrs:
            msg = f"'{name}' in file {lh5_file} is missing the datatype attribute"
            raise RuntimeError(msg)

        datatype = h5f[name].attrs["datatype"]
        datatype, shape, elements = parse_datatype(datatype)

        # scalars are dim-0 datasets
        if datatype == "scalar":
            return None

        # structs don't have rows
        if datatype == "struct":
            return None

        # tables should have elements with all the same length
        if datatype == "table":
            # read out each of the fields
            rows_read = None
            for field in elements:
                n_rows_read = self.read_n_rows(name + "/" + field, h5f)
                if not rows_read:
                    rows_read = n_rows_read
                elif rows_read != n_rows_read:
                    log.warning(
                        f"'{field}' field in table '{name}' has {rows_read} rows, "
                        f"{n_rows_read} was expected"
                    )
            return rows_read

        # length of vector of vectors is the length of its cumulative_length
        if elements.startswith("array"):
            return self.read_n_rows(f"{name}/cumulative_length", h5f)

        # length of vector of encoded vectors is the length of its decoded_size
        if (
            elements.startswith("encoded_array")
            or datatype == "array_of_encoded_equalsized_arrays"
        ):
            return self.read_n_rows(f"{name}/encoded_data", h5f)

        # return array length (without reading the array!)
        if "array" in datatype:
            # compute the number of rows to read
            return h5f[name].shape[0]

        msg = f"don't know how to read datatype '{datatype}'"
        raise RuntimeError(msg)


@nb.njit(parallel=False, fastmath=True)
def _make_fd_idx(starts, stops, idx):
    k = 0
    if len(starts) < len(stops):
        for i in range(stops[0]):
            idx[k] = i
            k += 1
        stops = stops[1:]
    for j in range(len(starts)):
        for i in range(starts[j], stops[j]):
            idx[k] = i
            k += 1
    return (idx,)
