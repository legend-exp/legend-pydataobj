from __future__ import annotations

import bisect
import logging
import sys
from collections import defaultdict

import h5py
import numpy as np

from ....types import (
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
from ... import datatype as dtypeutils
from ...exceptions import LH5DecodeError
from ...utils import read_n_rows
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
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    use_h5idx=False,
    field_mask=None,
    obj_buf=None,
    obj_buf_start=0,
    decompress=True,
):
    # Handle list-of-files recursively
    if not isinstance(h5o, (h5py.Group, h5py.Dataset)):
        lh5_objs = list(h5o)
        n_rows_read = 0

        for i, _h5o in enumerate(lh5_objs):
            if isinstance(idx, list) and len(idx) > 0 and not np.isscalar(idx[0]):
                # a list of lists: must be one per file
                idx_i = idx[i]
            elif idx is not None:
                # make idx a proper tuple if it's not one already
                if not (isinstance(idx, tuple) and len(idx) == 1):
                    idx = (idx,)
                # idx is a long continuous array
                n_rows_i = read_n_rows(_h5o)
                # find the length of the subset of idx that contains indices
                # that are less than n_rows_i
                n_rows_to_read_i = bisect.bisect_left(idx[0], n_rows_i)
                # now split idx into idx_i and the remainder
                idx_i = (idx[0][:n_rows_to_read_i],)
                idx = (idx[0][n_rows_to_read_i:] - n_rows_i,)
            else:
                idx_i = None
            n_rows_i = n_rows - n_rows_read

            obj_buf, n_rows_read_i = _h5_read_lgdo(
                _h5o,
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

        return obj_buf, n_rows_read

    log.debug(
        f"reading {h5o.file.filename}:{h5o.name}[{start_row}:{n_rows}], decompress = {decompress}, "
        + (f" with field mask {field_mask}" if field_mask else "")
    )

    # make idx a proper tuple if it's not one already
    if not (isinstance(idx, tuple) and len(idx) == 1) and idx is not None:
        idx = (idx,)

    try:
        lgdotype = dtypeutils.datatype(h5o.attrs["datatype"])
    except KeyError as e:
        msg = "dataset not in file or missing 'datatype' attribute"
        raise LH5DecodeError(msg, h5o) from e

    if lgdotype is Scalar:
        return _h5_read_scalar(
            h5o,
            obj_buf=obj_buf,
        )

    # check field_mask and make it a default dict
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
        msg = "bad field_mask type"
        raise ValueError(msg, type(field_mask).__name__)

    if lgdotype is Struct:
        return _h5_read_struct(
            h5o,
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
            h5o,
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
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
        )

    msg = f"no rule to decode {lgdotype.__name__} from LH5"
    raise LH5DecodeError(msg, h5o)


def _h5_read_struct(
    h5g,
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

    attrs = dict(h5g.attrs)

    # determine fields to be read out
    all_fields = dtypeutils.get_struct_fields(attrs["datatype"])
    selected_fields = (
        [field for field in all_fields if field_mask[field]]
        if field_mask is not None
        else all_fields
    )

    # modify datatype in attrs if a field_mask was used
    attrs["datatype"] = "struct{" + ",".join(selected_fields) + "}"

    # loop over fields and read
    obj_dict = {}
    for field in selected_fields:
        # support for integer keys
        field_key = int(field) if attrs.get("int_keys") else str(field)
        obj_dict[field_key], _ = _h5_read_lgdo(
            h5g[field],
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            decompress=decompress,
        )

    return Struct(obj_dict=obj_dict, attrs=attrs), 1


def _h5_read_table(
    h5g,
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
        raise LH5DecodeError(msg, h5g)

    attrs = dict(h5g.attrs)

    # determine fields to be read out
    all_fields = dtypeutils.get_struct_fields(attrs["datatype"])
    selected_fields = (
        [field for field in all_fields if field_mask[field]]
        if field_mask is not None
        else all_fields
    )

    # modify datatype in attrs if a field_mask was used
    attrs["datatype"] = "table{" + ",".join(selected_fields) + "}"

    # read out each of the fields
    col_dict = {}
    rows_read = []
    for field in selected_fields:
        fld_buf = None
        if obj_buf is not None:
            if not isinstance(obj_buf, Table) or field not in obj_buf:
                msg = "provided object buffer is not a Table or columns are missing"
                raise LH5DecodeError(msg, h5g)

            fld_buf = obj_buf[field]

        col_dict[field], n_rows_read = _h5_read_lgdo(
            h5g[field],
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
        log.warning(f"Table '{h5g.name}' has no fields specified by {field_mask=}")

    for n in rows_read[1:]:
        if n != n_rows_read:
            log.warning(
                f"Table '{h5g.name}' got strange n_rows_read = {n}, "
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
        table.loc = n_rows_read
        return table, n_rows_read

    # We have read all fields into the object buffer. Run
    # checks: All columns should be the same size. So update
    # table's size as necessary, warn if any mismatches are found
    obj_buf.resize(do_warn=True)
    # set (write) loc to end of tree
    obj_buf.loc = obj_buf_start + n_rows_read

    # check attributes
    utils.check_obj_buf_attrs(obj_buf.attrs, attrs, h5g)

    return obj_buf, n_rows_read
