from __future__ import annotations

import logging
import sys

import numba
import numpy as np

from ....types import (
    Array,
    VectorOfVectors,
)
from ... import datatype as dtypeutils
from ...exceptions import LH5DecodeError
from .array import (
    _h5_read_array,
)

log = logging.getLogger(__name__)


def _h5_read_vector_of_vectors(
    h5g,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    use_h5idx=False,
    obj_buf=None,
    obj_buf_start=0,
):
    if obj_buf is not None and not isinstance(obj_buf, VectorOfVectors):
        msg = "object buffer is not a VectorOfVectors"
        raise LH5DecodeError(msg, h5g)

    # read out cumulative_length
    cumulen_buf = None if obj_buf is None else obj_buf.cumulative_length
    cumulative_length, n_rows_read = _h5_read_array(
        h5g["cumulative_length"],
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

        fd_starts, fds_n_rows_read = _h5_read_array(
            h5g["cumulative_length"],
            start_row=start_row,
            n_rows=n_rows,
            idx=idx2,
            use_h5idx=use_h5idx,
            obj_buf=None,
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
            fd_start = h5g["cumulative_length"][start_row - 1]

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
                    f"{start_row} and {start_row+n_rows_read}"
                )
                raise LH5DecodeError(msg, h5g)

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
    lgdotype = dtypeutils.datatype(h5g["flattened_data"].attrs["datatype"])
    if lgdotype is Array:
        _func = _h5_read_array
    elif lgdotype is VectorOfVectors:
        _func = _h5_read_vector_of_vectors
    else:
        msg = "type {lgdotype.__name__} is not supported"
        raise LH5DecodeError(msg, h5g, "flattened_data")

    flattened_data, _ = _func(
        h5g["flattened_data"],
        start_row=fd_start,
        n_rows=fd_n_rows,
        idx=fd_idx,
        use_h5idx=use_h5idx,
        obj_buf=fd_buf,
        obj_buf_start=fd_buf_start,
    )

    if obj_buf is not None:
        # if the buffer is partially filled, cumulative_length will be invalid
        # (i.e. non monotonically increasing). Let's fix that but filling the
        # rest of the array with the length of flattened_data
        end = obj_buf_start + n_rows_read
        obj_buf.cumulative_length.nda[end:] = obj_buf.cumulative_length.nda[end - 1]

        return obj_buf, n_rows_read

    return (
        VectorOfVectors(
            flattened_data=flattened_data,
            cumulative_length=cumulative_length,
            attrs=dict(h5g.attrs),
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
