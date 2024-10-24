from __future__ import annotations

import logging

import h5py
import numpy as np

from .... import types
from ... import datatype
from ...exceptions import LH5DecodeError

log = logging.getLogger(__name__)


def check_obj_buf_attrs(attrs, new_attrs, fname, oname):
    if set(attrs.keys()) != set(new_attrs.keys()):
        msg = (
            f"existing buffer and new data chunk have different attributes: "
            f"obj_buf.attrs={attrs} != {fname}[{oname}].attrs={new_attrs}"
        )
        raise LH5DecodeError(msg, fname, oname)


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
            n_rows_read = read_n_rows(h5py.h5o.open(h5o, field.encode()), fname, field)
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
        return read_n_rows(
            h5py.h5o.open(h5o, b"cumulative_length"), fname, "cumulative_length"
        )

    # length of vector of encoded vectors is the length of its decoded_size
    if lgdotype in (types.VectorOfEncodedVectors, types.ArrayOfEncodedEqualSizedArrays):
        return read_n_rows(h5py.h5o.open(h5o, b"encoded_data"), fname, "encoded_data")

    # return array length (without reading the array!)
    if issubclass(lgdotype, types.Array):
        # compute the number of rows to read
        return h5o.get_space().shape[0]

    msg = f"don't know how to read rows of LGDO {lgdotype.__name__}"
    raise LH5DecodeError(msg, fname, oname)
