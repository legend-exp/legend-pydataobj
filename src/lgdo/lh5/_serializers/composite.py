from __future__ import annotations

import bisect
import logging
import sys
from collections import defaultdict

import numba
import numpy as np

from ... import compression as compress
from ...types import (
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
from . import datatype as utils
from .array import (
    _h5_read_array,
    _h5_read_array_of_equalsized_arrays,
    _h5_read_fixedsize_array,
)
from .scalar import _h5_read_scalar

log = logging.getLogger(__name__)


def _h5_read_lgdo(
    name,
    h5f,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    use_h5idx=False,
    field_mask=None,
    obj_buf=None,
    obj_buf_start=0,
    decompress=True,
):
    log.debug(
        f"reading {h5f.filename}:{name}[{start_row}:{n_rows}], decompress = {decompress}, "
        + (f" with field mask {field_mask}" if field_mask else "")
    )

    # make idx a proper tuple if it's not one already
    if not (isinstance(idx, tuple) and len(idx) == 1) and idx is not None:
        idx = (idx,)

    datatype = h5f[name].attrs["datatype"]
    lgdotype = utils.datatype(h5f[name].attrs["datatype"])

    if lgdotype is Scalar:
        return _h5_read_scalar(
            name,
            h5f,
            obj_buf=obj_buf,
        )

    # check field_mask and make it a default dict
    if lgdotype in (Struct, Table):
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

    if lgdotype is Struct:
        return _h5_read_struct(
            name,
            h5f,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            field_mask=field_mask,
            decompress=decompress,
        )

    # Below here is all array-like types. So trim idx if needed
    if idx is not None:
        # check if idx is just an ordered list of the integers if so can ignore
        if (idx[0] == np.arange(0, len(idx[0]), 1)).all():
            if n_rows > len(idx[0]):
                n_rows = len(idx[0])
            idx = None
        else:
            # chop off indices < start_row
            i_first_valid = bisect.bisect_left(idx[0], start_row)
            idxa = idx[0][i_first_valid:]
            # don't readout more than n_rows indices
            idx = (idxa[:n_rows],)  # works even if n_rows > len(idxa)

    if lgdotype is Table:
        return _h5_read_table(
            name,
            h5f,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            field_mask=field_mask,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
            decompress=decompress,
        )

    if lgdotype in (ArrayOfEncodedEqualSizedArrays, VectorOfEncodedVectors):
        return _h5_read_encoded_array(
            name,
            h5f,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
            decompress=decompress,
        )

    if lgdotype is VectorOfVectors:
        return _h5_read_vector_of_vectors(
            name,
            h5f,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
        )

    # FixedSizeArray
    if lgdotype is FixedSizeArray:
        return _h5_read_fixedsize_array(
            name,
            h5f,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
        )

    # ArrayOfEqualSizedArrays
    if lgdotype is ArrayOfEqualSizedArrays:
        return _h5_read_array_of_equalsized_arrays(
            name,
            h5f,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
        )

    # Array
    if lgdotype is Array:
        return _h5_read_array(
            name,
            h5f,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
        )

    msg = f"don't know how to read datatype {datatype}"
    raise RuntimeError(msg)


def _h5_read_vector_of_vectors(
    name,
    h5f,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    use_h5idx=False,
    obj_buf=None,
    obj_buf_start=0,
):
    if obj_buf is not None and not isinstance(obj_buf, VectorOfVectors):
        msg = f"obj_buf for '{name}' not a LGDO VectorOfVectors"
        raise ValueError(msg)

    # read out cumulative_length
    cumulen_buf = None if obj_buf is None else obj_buf.cumulative_length
    cumulative_length, n_rows_read = _h5_read_lgdo(
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
        # get the starting indices for each array in flattened data:
        # the starting index for array[i] is cumulative_length[i-1]
        idx2 = (np.asarray(idx[0]).copy() - 1,)

        # re-read cumulative_length with these indices
        # note this will allocate memory for fd_starts!
        fd_start = None
        if idx2[0][0] == -1:
            idx2 = (idx2[0][1:],)
            fd_start = 0  # this variable avoids an ndarray append

        fd_starts, fds_n_rows_read = _h5_read_lgdo(
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
        fd_idx = np.empty(fd_n_rows, dtype="int32")
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
    flattened_data, dummy_rows_read = _h5_read_lgdo(
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


@numba.njit(parallel=False, fastmath=True)
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


def _h5_read_struct(
    name,
    h5f,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    use_h5idx=False,
    field_mask=None,
    decompress=True,
):
    # modify datatype in attrs if a field_mask was used
    attrs = dict(h5f[name].attrs)
    if field_mask is not None:
        selected_fields = []
        for field in utils.get_struct_fields(attrs["datatype"]):
            if field_mask[field]:
                selected_fields.append(field)
        attrs["datatype"] = "struct{" + ",".join(selected_fields) + "}"
    else:
        selected_fields = utils.get_struct_fields(attrs["datatype"])

    # loop over fields and read
    obj_dict = {}
    for field in selected_fields:
        # TODO: it's strange to pass start_row, n_rows, idx to struct
        # fields. If they all had shared indexing, they should be in a
        # table... Maybe should emit a warning? Or allow them to be
        # dicts keyed by field name?
        if "int_keys" in h5f[name].attrs:
            if dict(h5f[name].attrs)["int_keys"]:
                f = int(field)
        else:
            f = str(field)

        obj_dict[f], _ = _h5_read_lgdo(
            f"{name}/{field}",
            h5f,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            # field_mask=field_mask,
            decompress=decompress,
        )

    return Struct(obj_dict=obj_dict, attrs=attrs), 1


def _h5_read_table(
    name,
    h5f,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    use_h5idx=False,
    field_mask=None,
    obj_buf=None,
    obj_buf_start=0,
    decompress=True,
):
    col_dict = {}

    # modify datatype in attrs if a field_mask was used
    attrs = dict(h5f[name].attrs)
    if field_mask is not None:
        selected_fields = []
        for field in utils.get_struct_fields(attrs["datatype"]):
            if field_mask[field]:
                selected_fields.append(field)
        attrs["datatype"] = "table{" + ",".join(selected_fields) + "}"
    else:
        selected_fields = utils.get_struct_fields(attrs["datatype"])

    # read out each of the fields
    rows_read = []
    for field in selected_fields:
        fld_buf = None
        if obj_buf is not None:
            if not isinstance(obj_buf, Table) or field not in obj_buf:
                msg = f"obj_buf for LGDO Table '{name}' not formatted correctly"
                raise ValueError(msg)

            fld_buf = obj_buf[field]

        col_dict[field], n_rows_read = _h5_read_lgdo(
            f"{name}/{field}",
            h5f,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            # field_mask=field_mask,
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


def _h5_read_encoded_array(
    name,
    h5f,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    use_h5idx=False,
    obj_buf=None,
    obj_buf_start=0,
    decompress=True,
):
    datatype = h5f[name].attrs["datatype"]
    elements = utils.get_nested_datatype_string(h5f[name].attrs["datatype"])

    for cond, enc_lgdo in [
        (
            datatype.startswith("array_of_encoded_equalsized_arrays"),
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

            decoded_size, _ = _h5_read_lgdo(
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
            encoded_data, n_rows_read = _h5_read_vector_of_vectors(
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

    msg = f"'{name}' does not look like of encoded array type"
    raise RuntimeError(msg)
