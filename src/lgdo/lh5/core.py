from __future__ import annotations

import inspect
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import h5py
from numpy.typing import ArrayLike

from .. import types
from . import _serializers


def read(
    name: str,
    lh5_file: str | h5py.File | Sequence[str | h5py.File],
    start_row: int = 0,
    n_rows: int = sys.maxsize,
    idx: ArrayLike = None,
    use_h5idx: bool = False,
    field_mask: Mapping[str, bool] | Sequence[str] | None = None,
    obj_buf: types.LGDO = None,
    obj_buf_start: int = 0,
    decompress: bool = True,
) -> types.LGDO | tuple[types.LGDO, int]:
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
        :mod:`h5py` call such that only the selected rows are read directly
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
        ``table.loc``. If `obj_buf` is ``None``, only `object` is returned.
    """
    if isinstance(lh5_file, h5py.File):
        lh5_obj = lh5_file[name]
    elif isinstance(lh5_file, str):
        lh5_file = h5py.File(lh5_file, mode="r")
        lh5_obj = lh5_file[name]
    else:
        lh5_obj = []
        for h5f in lh5_file:
            if isinstance(lh5_file, str):
                h5f = h5py.File(h5f, mode="r")  # noqa: PLW2901
            lh5_obj += h5f[name]

    obj, n_rows_read = _serializers._h5_read_lgdo(
        lh5_obj,
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


def write(
    obj: types.LGDO,
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
        - ``append_column`` or ``ac``: append columns from an
          :class:`~.lgdo.table.Table` `obj` only if there is an existing
          :class:`~.lgdo.table.Table` in the `lh5_file` with the same
          `name` and :class:`~.lgdo.table.Table.size`. If the sizes don't
          match, or if there are matching fields, it errors out.
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
    return _serializers._h5_write_lgdo(
        obj,
        name,
        lh5_file,
        group=group,
        start_row=start_row,
        n_rows=n_rows,
        wo_mode=wo_mode,
        write_start=write_start,
        **h5py_kwargs,
    )


def read_as(
    name: str,
    lh5_file: str | h5py.File | Sequence[str | h5py.File],
    library: str,
    **kwargs,
) -> Any:
    """Read LH5 data from disk straight into a third-party data format view.

    This function is nothing more than a shortcut chained call to
    :func:`.read` and to :meth:`.LGDO.view_as`.

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
