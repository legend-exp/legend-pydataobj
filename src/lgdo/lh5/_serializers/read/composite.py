from __future__ import annotations

import bisect
import logging
import sys

import h5py
import numpy as np

from ....types import (
    Array,
    ArrayOfEncodedEqualSizedArrays,
    ArrayOfEqualSizedArrays,
    FixedSizeArray,
    Histogram,
    Scalar,
    Struct,
    Table,
    VectorOfEncodedVectors,
    VectorOfVectors,
    WaveformTable,
)
from ... import datatype as dtypeutils
from ...exceptions import LH5DecodeError
from . import utils
from .array import (
    _h5_read_array,
    _h5_read_array_of_equalsized_arrays,
    _h5_read_fixedsize_array,
)
from .encoded import (
    _h5_read_array_of_encoded_equalsized_arrays,
    _h5_read_vector_of_encoded_vectors,
)
from .scalar import _h5_read_scalar
from .vector_of_vectors import _h5_read_vector_of_vectors

log = logging.getLogger(__name__)


def _h5_read_lgdo(
    h5o,
    fname,
    oname,
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
        f"reading {fname}:{oname}[{start_row}:{n_rows}], decompress = {decompress}, "
        + (f" with field mask {field_mask}" if field_mask else "")
    )

    attrs = utils.read_attrs(h5o, fname, oname)
    try:
        lgdotype = dtypeutils.datatype(attrs["datatype"])
    except KeyError as e:
        msg = "dataset not in file or missing 'datatype' attribute"
        raise LH5DecodeError(msg, fname, oname) from e

    if lgdotype is Scalar:
        return _h5_read_scalar(
            h5o,
            fname,
            oname,
            obj_buf=obj_buf,
        )

    # Convert whatever we input into a defaultdict
    field_mask = utils.build_field_mask(field_mask)

    if lgdotype is Struct:
        return _h5_read_struct(
            h5o,
            fname,
            oname,
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
        if (idx == np.arange(0, len(idx), 1)).all():
            n_rows = min(n_rows, len(idx))
            idx = None
        else:
            # chop off indices < start_row
            i_first_valid = bisect.bisect_left(idx, start_row)
            idxa = idx[i_first_valid:]
            # don't readout more than n_rows indices
            idx = idxa[:n_rows]  # works even if n_rows > len(idxa)

    if lgdotype is Table:
        return _h5_read_table(
            h5o,
            fname,
            oname,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            field_mask=field_mask,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
            decompress=decompress,
        )

    if lgdotype is Histogram:
        return _h5_read_histogram(
            h5o,
            fname,
            oname,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            field_mask=field_mask,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
            decompress=decompress,
        )

    if lgdotype is ArrayOfEncodedEqualSizedArrays:
        return _h5_read_array_of_encoded_equalsized_arrays(
            h5o,
            fname,
            oname,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
            decompress=decompress,
        )

    if lgdotype is VectorOfEncodedVectors:
        return _h5_read_vector_of_encoded_vectors(
            h5o,
            fname,
            oname,
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
            h5o,
            fname,
            oname,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
        )

    if lgdotype is FixedSizeArray:
        return _h5_read_fixedsize_array(
            h5o,
            fname,
            oname,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
        )

    if lgdotype is ArrayOfEqualSizedArrays:
        return _h5_read_array_of_equalsized_arrays(
            h5o,
            fname,
            oname,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
        )

    if lgdotype is Array:
        return _h5_read_array(
            h5o,
            fname,
            oname,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
        )

    msg = f"no rule to decode {lgdotype.__name__} from LH5"
    raise LH5DecodeError(msg, fname, oname)


def _h5_read_struct(
    h5g,
    fname,
    oname,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    use_h5idx=False,
    field_mask=None,
    decompress=True,
):
    # TODO: it's strange to pass start_row, n_rows, idx to struct
    # fields. If they all had shared indexing, they should be in a
    # table... Maybe should emit a warning? Or allow them to be
    # dicts keyed by field name?

    attrs = utils.read_attrs(h5g, fname, oname)

    # determine fields to be read out
    all_fields = dtypeutils.get_struct_fields(attrs["datatype"])
    selected_fields = utils.eval_field_mask(field_mask, all_fields)

    # modify datatype in attrs if a field_mask was used
    attrs["datatype"] = (
        "struct{" + ",".join(field for field, _ in selected_fields) + "}"
    )

    # loop over fields and read
    obj_dict = {}
    for field, submask in selected_fields:
        # support for integer keys
        field_key = int(field) if attrs.get("int_keys") else str(field)
        h5o = h5py.h5o.open(h5g, field.encode("utf-8"))
        obj_dict[field_key], _ = _h5_read_lgdo(
            h5o,
            fname,
            f"{oname}/{field}",
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            field_mask=submask,
            decompress=decompress,
        )
        h5o.close()

    return Struct(obj_dict=obj_dict, attrs=attrs), 1


def _h5_read_table(
    h5g,
    fname,
    oname,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    use_h5idx=False,
    field_mask=None,
    obj_buf=None,
    obj_buf_start=0,
    decompress=True,
):
    if obj_buf is not None and not isinstance(obj_buf, Table):
        msg = "provided object buffer is not a Table"
        raise LH5DecodeError(msg, fname, oname)

    attrs = utils.read_attrs(h5g, fname, oname)

    # determine fields to be read out
    all_fields = dtypeutils.get_struct_fields(attrs["datatype"])
    selected_fields = utils.eval_field_mask(field_mask, all_fields)

    # modify datatype in attrs if a field_mask was used
    attrs["datatype"] = "table{" + ",".join(field for field, _ in selected_fields) + "}"

    # read out each of the fields
    col_dict = {}
    rows_read = []
    for field, submask in selected_fields:
        fld_buf = None
        if obj_buf is not None:
            if not isinstance(obj_buf, Table) or field not in obj_buf:
                msg = "provided object buffer is not a Table or columns are missing"
                raise LH5DecodeError(msg, fname, oname)

            fld_buf = obj_buf[field]

        h5o = h5py.h5o.open(h5g, field.encode("utf-8"))
        col_dict[field], n_rows_read = _h5_read_lgdo(
            h5o,
            fname,
            f"{oname}/{field}",
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            obj_buf=fld_buf,
            obj_buf_start=obj_buf_start,
            field_mask=submask,
            decompress=decompress,
        )
        h5o.close()

        if obj_buf is not None and obj_buf_start + n_rows_read > len(obj_buf):
            obj_buf.resize(obj_buf_start + n_rows_read)

        rows_read.append(n_rows_read)

    # warn if all columns don't read in the same number of rows
    if len(rows_read) > 0:
        n_rows_read = rows_read[0]
    else:
        n_rows_read = 0
        log.warning(f"Table '{oname}' has no fields specified by {field_mask=}")

    for n in rows_read[1:]:
        if n != n_rows_read:
            log.warning(
                f"Table '{oname}' got strange n_rows_read = {n}, "
                "{n_rows_read} was expected ({rows_read})"
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
        table.resize(do_warn=True)
        return table, n_rows_read

    # We have read all fields into the object buffer. Run
    # checks: All columns should be the same size. So update
    # table's size as necessary, warn if any mismatches are found
    obj_buf.resize(do_warn=True)

    # check attributes
    utils.check_obj_buf_attrs(obj_buf.attrs, attrs, fname, oname)

    return obj_buf, n_rows_read


def _h5_read_histogram(
    h5g,
    fname,
    oname,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    use_h5idx=False,
    field_mask=None,
    obj_buf=None,
    obj_buf_start=0,
    decompress=True,
):
    if obj_buf is not None or obj_buf_start != 0:
        msg = "reading a histogram into an existing object buffer is not supported"
        raise LH5DecodeError(msg, fname, oname)

    struct, n_rows_read = _h5_read_struct(
        h5g,
        fname,
        oname,
        start_row=start_row,
        n_rows=n_rows,
        idx=idx,
        use_h5idx=use_h5idx,
        field_mask=field_mask,
        decompress=decompress,
    )

    binning = []
    for _, a in struct.binning.items():
        be = a.binedges
        if isinstance(be, Struct):
            b = (None, be.first.value, be.last.value, be.step.value, a.closedleft.value)
        elif isinstance(be, Array):
            b = (be, None, None, None, a.closedleft.value)
        else:
            msg = "unexpected binning of histogram"
            raise LH5DecodeError(msg, fname, oname)
        ax = Histogram.Axis(*b)
        # copy attrs to "clone" the "whole" struct.
        ax.attrs = a.getattrs(datatype=True)
        ax["binedges"].attrs = be.getattrs(datatype=True)
        binning.append(ax)

    isdensity = struct.isdensity.value
    weights = struct.weights
    attrs = struct.getattrs(datatype=True)
    histogram = Histogram(weights, binning, isdensity, attrs=attrs)

    return histogram, n_rows_read
