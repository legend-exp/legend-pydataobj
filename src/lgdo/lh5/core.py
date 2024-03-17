from __future__ import annotations

import inspect
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import h5py
from numpy.typing import ArrayLike

from ..types import LGDO
from . import _serializers


def read(
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
) -> LGDO | tuple[LGDO, int]:
    """Read LH5 object data from a file.

    Note
    ----
    Use the ``idx`` parameter to read out particular rows of the data. The
    ``use_h5idx`` flag controls whether *only* those rows are read from
    disk or if the rows are indexed after reading the entire object.
    Reading individual rows can be orders of magnitude slower than reading
    the whole object and then indexing the desired rows. The default
    behavior (``use_h5idx=False``) is to use slightly more memory for a
    much faster read. See `legend-pydataobj/issues/#29
    <https://github.com/legend-exp/legend-pydataobj/issues/29>`_ for
    additional information.

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
        For NumPy-style "fancying indexing" for the read to select only
        some rows, e.g. after applying some cuts to particular columns.
        Only selection along the first axis is supported, so tuple
        arguments must be one-tuples.  If `n_rows` is not false, `idx` will
        be truncated to `n_rows` before reading. To use with a list of
        files, can pass in a list of `idx`'s (one for each file) or use a
        long contiguous list (e.g. built from a previous identical read).
        If used in conjunction with `start_row` and `n_rows`, will be
        sliced to obey those constraints, where `n_rows` is interpreted as
        the (max) number of *selected* values (in `idx`) to be read out.
        Note that the ``use_h5idx`` parameter controls some behaviour of
        the read and that the default behavior (``use_h5idx=False``)
        prioritizes speed over a small memory penalty.
    use_h5idx
        ``True`` will directly pass the ``idx`` parameter to the underlying
        ``h5py`` call such that only the selected rows are read directly
        into memory, which conserves memory at the cost of speed. There can
        be a significant penalty to speed for larger files (1 - 2 orders of
        magnitude longer time).  ``False`` (default) will read the entire
        object into memory before performing the indexing. The default is
        much faster but requires additional memory, though a relatively
        small amount in the typical use case. It is recommended to leave
        this parameter as its default.
    field_mask
        For tables and structs, determines which fields get read out.
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
    obj, n_rows_read = _serializers._h5_read_lgdo(
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

    return obj if obj_buf is None else (obj, n_rows_read)


def read_as(
    name: str,
    lh5_file: str | h5py.File | Sequence[str | h5py.File],
    library: str,
    **kwargs,
) -> Any:
    """Read LH5 data from disk straight into a third-party data format view.

    This function is nothing more than a shortcut chained call to
    :meth:`.LH5Store.read` and to :meth:`.LGDO.view_as`.

    Parameters
    ----------
    name
        LH5 object name on disk.
    lh5_file
        LH5 file name.
    library
        string ID of the third-party data format library (``np``, ``pd``,
        ``ak``, etc).

    See Also
    --------
    .read, .LGDO.view_as
    """
    # determine which keyword arguments should be forwarded to read() and which
    # should be forwarded to view_as()
    read_kwargs = inspect.signature(read).parameters.keys()

    kwargs1 = {}
    kwargs2 = {}
    for k, v in kwargs.items():
        if k in read_kwargs:
            kwargs1[k] = v
        else:
            kwargs2[k] = v

    # read the LGDO from disk
    # NOTE: providing a buffer does not make much sense
    obj = read(name, lh5_file, **kwargs1)

    # and finally return a view
    return obj.view_as(library, **kwargs2)
