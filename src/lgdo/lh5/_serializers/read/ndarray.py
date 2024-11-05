from __future__ import annotations

import logging
import sys
from bisect import bisect_left

import h5py
import numpy as np

from ....types import Array
from ... import datatype
from ...exceptions import LH5DecodeError
from .utils import read_attrs

log = logging.getLogger(__name__)


def _h5_read_ndarray(
    h5d,
    fname,
    oname,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    use_h5idx=False,
    obj_buf=None,
    obj_buf_start=0,
):
    if obj_buf is not None and not isinstance(obj_buf, Array):
        msg = "object buffer is not an Array"
        raise LH5DecodeError(msg, fname, oname)

    # compute the number of rows to read
    # we culled idx above for start_row and n_rows, now we have to apply
    # the constraint of the length of the dataset
    try:
        fspace = h5d.get_space()
        ds_n_rows = fspace.shape[0]
    except AttributeError as e:
        msg = "does not seem to be an HDF5 dataset"
        raise LH5DecodeError(msg, fname, oname) from e

    if idx is not None:
        if len(idx) > 0 and idx[-1] >= ds_n_rows:
            log.warning("idx indexed past the end of the array in the file. Culling...")
            n_rows_to_read = bisect_left(idx, ds_n_rows)
            idx = idx[:n_rows_to_read]
            if len(idx) == 0:
                log.warning("idx empty after culling.")
        n_rows_to_read = len(idx)
    else:
        n_rows_to_read = ds_n_rows - start_row
    n_rows_to_read = min(n_rows_to_read, n_rows)

    if idx is None:
        fspace.select_hyperslab(
            (start_row,) + (0,) * (h5d.rank - 1),
            (1,) * h5d.rank,
            None,
            (n_rows_to_read,) + fspace.shape[1:],
        )
    elif use_h5idx:
        # Note that h5s will automatically merge adjacent elements into a range
        fspace.select_none()
        for i in idx:
            fspace.select_hyperslab(
                (i,) + (0,) * (h5d.rank - 1),
                (1,) * h5d.rank,
                None,
                (1,) + fspace.shape[1:],
                h5py.h5s.SELECT_OR,
            )

    # Now read the array
    if obj_buf is not None and n_rows_to_read > 0:
        buf_size = obj_buf_start + n_rows_to_read
        if len(obj_buf) < buf_size:
            obj_buf.resize(buf_size)
        dest_sel = np.s_[obj_buf_start:buf_size]

        if idx is None or use_h5idx:
            mspace = h5py.h5s.create_simple(obj_buf.nda.shape)
            mspace.select_hyperslab(
                (obj_buf_start,) + (0,) * (h5d.rank - 1),
                (1,) * h5d.rank,
                None,
                (n_rows_to_read,) + fspace.shape[1:],
            )
            h5d.read(mspace, fspace, obj_buf.nda)
        else:
            tmp = np.empty(fspace.shape, h5d.dtype)
            h5d.read(fspace, fspace, tmp)
            obj_buf.nda[dest_sel, ...] = tmp[idx, ...]
        nda = obj_buf.nda
    elif n_rows == 0:
        tmp_shape = (0,) + h5d.shape[1:]
        nda = np.empty(tmp_shape, h5d.dtype)
    else:
        mspace = h5py.h5s.create_simple((n_rows_to_read,) + fspace.shape[1:])
        nda = np.empty(mspace.shape, h5d.dtype)
        if idx is None or use_h5idx:
            h5d.read(mspace, fspace, nda)
        else:
            tmp = np.empty(fspace.shape, h5d.dtype)
            h5d.read(fspace, fspace, tmp)
            nda[:, ...] = tmp[idx, ...]

    # Finally, set attributes and return objects
    attrs = read_attrs(h5d, fname, oname)

    # special handling for bools
    # (c and Julia store as uint8 so cast to bool)
    if datatype.get_nested_datatype_string(attrs["datatype"]) == "bool":
        nda = nda.astype(np.bool_, copy=False)

    return (nda, attrs, n_rows_to_read)
