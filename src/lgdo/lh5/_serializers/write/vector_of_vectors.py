from __future__ import annotations

import logging

import numpy as np

from .... import types
from ... import utils
from ...exceptions import LH5EncodeError
from .array import _h5_write_array

log = logging.getLogger(__name__)


def _h5_write_vector_of_vectors(
    obj,
    name,
    lh5_file,
    group="/",
    start_row=0,
    n_rows=None,
    wo_mode="append",
    write_start=0,
    **h5py_kwargs,
):
    assert isinstance(obj, types.VectorOfVectors)

    group = utils.get_h5_group(
        name, group, grp_attrs=obj.attrs, overwrite=(wo_mode == "o")
    )
    if n_rows is None or n_rows > obj.cumulative_length.nda.shape[0] - start_row:
        n_rows = obj.cumulative_length.nda.shape[0] - start_row

    # if appending we need to add an appropriate offset to the
    # cumulative lengths as appropriate for the in-file object
    # declare here because we have to subtract it off at the end
    offset = np.int64(0)
    if (wo_mode in ("a", "o")) and "cumulative_length" in group:
        len_cl = len(group["cumulative_length"])
        # if append, ignore write_start and set it to total number of vectors
        if wo_mode == "a":
            write_start = len_cl
        if len_cl > 0:
            # set offset to correct number of elements in flattened_data until write_start
            offset = group["cumulative_length"][write_start - 1]

    # First write flattened_data array. Only write rows with data.
    fd_start = 0 if start_row == 0 else obj.cumulative_length.nda[start_row - 1]
    fd_n_rows = (
        obj.cumulative_length.nda[start_row + n_rows - 1] - fd_start
        if len(obj.cumulative_length) > 0
        else 0
    )

    if isinstance(obj.flattened_data, types.Array):
        _func = _h5_write_array
    elif isinstance(obj.flattened_data, types.VectorOfVectors):
        _func = _h5_write_vector_of_vectors
    else:
        msg = (
            "don't know how to serialize to disk flattened_data "
            "of {type(obj.flattened_data).__name__} type"
        )
        raise LH5EncodeError(msg, lh5_file, group, f"{name}.flattened_data")

    _func(
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
    # cumulative_length. Need to adjust it for start_row, if different from zero
    if start_row > 0:
        offset -= obj.cumulative_length.nda[start_row - 1]

    # Add offset to obj.cumulative_length itself to avoid memory allocation.
    # Then subtract it off after writing! (otherwise it will be changed
    # upon return)

    # NOTE: this operation is not numerically safe (uint overflow in the lower
    # part of the array), but this is not a problem because those values are
    # not written to disk and we are going to restore the offset at the end
    np.add(
        obj.cumulative_length.nda,
        offset,
        out=obj.cumulative_length.nda,
        casting="unsafe",
    )

    _h5_write_array(
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

    np.subtract(
        obj.cumulative_length.nda,
        offset,
        out=obj.cumulative_length.nda,
        casting="unsafe",
    )
