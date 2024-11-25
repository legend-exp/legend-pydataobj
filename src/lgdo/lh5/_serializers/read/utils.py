from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Collection, Mapping

import h5py
import numpy as np

from .... import types
from ... import datatype
from ...exceptions import LH5DecodeError
from . import scalar

log = logging.getLogger(__name__)


def check_obj_buf_attrs(attrs, new_attrs, fname, oname):
    if set(attrs.keys()) != set(new_attrs.keys()):
        msg = (
            f"existing buffer and new data chunk have different attributes: "
            f"obj_buf.attrs={attrs} != {fname}[{oname}].attrs={new_attrs}"
        )
        raise LH5DecodeError(msg, fname, oname)


def build_field_mask(field_mask: Mapping[str, bool] | Collection[str]) -> defaultdict:
    # check field_mask and make it a default dict
    if field_mask is None:
        return defaultdict(lambda: True)
    if isinstance(field_mask, dict):
        default = True
        if len(field_mask) > 0:
            default = not field_mask[next(iter(field_mask.keys()))]
        return defaultdict(lambda: default, field_mask)
    if isinstance(field_mask, (list, tuple, set)):
        return defaultdict(bool, {field: True for field in field_mask})
    if isinstance(field_mask, defaultdict):
        return field_mask
    msg = "bad field_mask type"
    raise ValueError(msg, type(field_mask).__name__)


def eval_field_mask(
    field_mask: defaultdict, all_fields: list[str]
) -> list[tuple(str, defaultdict)]:
    """Get list of fields that need to be loaded along with a sub-field-mask
    in case we have a nested Table"""

    if field_mask is None:
        return all_fields

    this_field_mask = defaultdict(field_mask.default_factory)
    sub_field_masks = {}

    for key, val in field_mask.items():
        field = key.strip("/")
        pos = field.find("/")
        if pos < 0:
            this_field_mask[field] = val
        else:
            sub_field = field[pos + 1 :]
            field = field[:pos]
            this_field_mask[field] = True
            sub_mask = sub_field_masks.setdefault(
                field, defaultdict(field_mask.default_factory)
            )
            sub_mask[sub_field] = val

    return [
        (field, sub_field_masks.get(field))
        for field in all_fields
        if this_field_mask[field]
    ]


def read_attrs(h5o, fname, oname):
    """Read all attributes for an hdf5 dataset or group using low level API
    and return them as a dict. Assume all are strings or scalar types."""
    attrs = {}
    for i_attr in range(h5py.h5a.get_num_attrs(h5o)):
        h5a = h5py.h5a.open(h5o, index=i_attr)
        name = h5a.get_name().decode()
        if h5a.shape != ():
            msg = f"attribute {oname} is not a string or scalar"
            raise LH5DecodeError(msg, fname, oname)
        val = np.empty((), h5a.dtype)
        h5a.read(val)
        if h5a.get_type().get_class() == h5py.h5t.STRING:
            attrs[name] = val.item().decode()
        else:
            attrs[name] = val.item()
        h5a.close()
    return attrs


def read_n_rows(h5o, fname, oname):
    """Read number of rows in LH5 object"""
    if not h5py.h5a.exists(h5o, b"datatype"):
        msg = "missing 'datatype' attribute"
        raise LH5DecodeError(msg, fname, oname)

    h5a = h5py.h5a.open(h5o, b"datatype")
    type_attr = np.empty((), h5a.dtype)
    h5a.read(type_attr)
    type_attr = type_attr.item().decode()
    lgdotype = datatype.datatype(type_attr)

    # scalars are dim-0 datasets
    if lgdotype is types.Scalar:
        return None

    # structs don't have rows
    if lgdotype is types.Struct:
        return None

    # tables should have elements with all the same length
    if lgdotype is types.Table:
        # read out each of the fields
        rows_read = None
        for field in datatype.get_struct_fields(type_attr):
            obj = h5py.h5o.open(h5o, field.encode())
            n_rows_read = read_n_rows(obj, fname, field)
            obj.close()
            if not rows_read:
                rows_read = n_rows_read
            elif rows_read != n_rows_read:
                log.warning(
                    f"'{field}' field in table '{oname}' has {rows_read} rows, "
                    f"{n_rows_read} was expected"
                )

        return rows_read

    # length of vector of vectors is the length of its cumulative_length
    if lgdotype is types.VectorOfVectors:
        obj = h5py.h5o.open(h5o, b"cumulative_length")
        n_rows = read_n_rows(obj, fname, "cumulative_length")
        obj.close()
        return n_rows

    # length of vector of encoded vectors is the length of its decoded_size
    if lgdotype in (types.VectorOfEncodedVectors, types.ArrayOfEncodedEqualSizedArrays):
        obj = h5py.h5o.open(h5o, b"encoded_data")
        n_rows = read_n_rows(obj, fname, "encoded_data")
        obj.close()
        return n_rows

    # return array length (without reading the array!)
    if issubclass(lgdotype, types.Array):
        # compute the number of rows to read
        return h5o.get_space().shape[0]

    msg = f"don't know how to read rows of LGDO {lgdotype.__name__}"
    raise LH5DecodeError(msg, fname, oname)


def read_size_in_bytes(h5o, fname, oname, field_mask=None):
    """Read number size in LH5 object in memory (in B)"""
    if not h5py.h5a.exists(h5o, b"datatype"):
        msg = "missing 'datatype' attribute"
        raise LH5DecodeError(msg, fname, oname)

    h5a = h5py.h5a.open(h5o, b"datatype")
    type_attr = np.empty((), h5a.dtype)
    h5a.read(type_attr)
    type_attr = type_attr.item().decode()
    lgdotype = datatype.datatype(type_attr)
    field_mask = build_field_mask(field_mask)

    # scalars are dim-0 datasets
    if lgdotype in (
        types.Scalar,
        types.Array,
        types.ArrayOfEqualSizedArrays,
        types.FixedSizeArray,
    ):
        return int(np.prod(h5o.shape) * h5o.dtype.itemsize)

    # tables should have elements with all the same length
    if lgdotype in (
        types.Struct,
        types.Histogram,
        types.Histogram.Axis,
        types.Table,
        types.WaveformTable,
    ):
        # read out each of the fields
        size = 0
        all_fields = datatype.get_struct_fields(type_attr)
        selected_fields = eval_field_mask(field_mask, all_fields)
        for field, submask in selected_fields:
            obj = h5py.h5o.open(h5o, field.encode())
            size += read_size_in_bytes(obj, fname, field, submask)
            obj.close()
        return size

    # length of vector of vectors is the length of its cumulative_length
    if lgdotype is types.VectorOfVectors:
        size = 0
        obj = h5py.h5o.open(h5o, b"cumulative_length")
        size += read_size_in_bytes(obj, fname, "cumulative_length")
        obj.close()
        obj = h5py.h5o.open(h5o, b"flattened_data")
        size += read_size_in_bytes(obj, fname, "flattened_data")
        obj.close()
        return size

    # length of vector of encoded vectors is the length of its decoded_size
    if lgdotype is types.ArrayOfEncodedEqualSizedArrays:
        obj = h5py.h5o.open(h5o, b"decoded_size")
        size = scalar._h5_read_scalar(obj, fname, "decoded_size")[0].value
        obj.close()

        obj = h5py.h5o.open(h5o, b"encoded_data")
        cl = h5py.h5o.open(obj, b"cumulative_length")
        size *= cl.shape[0]
        size *= 4  # TODO: UPDATE WHEN CODECS SUPPORT MORE DTYPES
        obj.close()

        return size

    msg = f"don't know how to read size of LGDO {lgdotype.__name__}"
    raise LH5DecodeError(msg, fname, oname)
