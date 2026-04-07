"""Zero-copy conversion between LGDO and Arrow types.

Note on allocation: we use ``to_numpy(zero_copy_only=False)`` throughout.
PyArrow performs zero-copy for all numeric types and only allocates when it
must (e.g. booleans, which are bit-packed in Arrow but byte-packed in NumPy,
or columns containing nulls that need sentinel values). Multi-chunk columns
are combined automatically with a warning.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pyarrow as pa

from .array import Array
from .arrayofequalsizedarrays import ArrayOfEqualSizedArrays
from .table import Table
from .vectorofvectors import VectorOfVectors
from .waveformtable import WaveformTable

# ============ Attrs serialization ============


def _serialize_attr(value) -> str:
    """Serialize an attr value to a JSON string for Arrow metadata."""
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _deserialize_attr(raw: bytes):
    """Deserialize an Arrow metadata value back to a Python object."""
    s = raw.decode()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


# ============ LGDO → Arrow ============


def lgdo_to_arrow(obj) -> pa.Table | pa.Array:
    """Convert an LGDO object to its Arrow equivalent.

    Type mapping:

    ========================  ==========================
    LGDO type                 Arrow type
    ========================  ==========================
    Table                     pa.Table
    WaveformTable             pa.StructArray
    Array                     pa.Array
    ArrayOfEqualSizedArrays   pa.FixedSizeListArray
    VectorOfVectors           pa.ListArray
    ========================  ==========================

    Preserves all attrs as JSON-encoded Arrow field metadata.

    Parameters
    ----------
    obj
        Any supported LGDO object.

    Returns
    -------
    pa.Table or pa.Array
        Arrow table (for Table) or Arrow array (for all other types).
    """
    if isinstance(obj, Table) and not isinstance(obj, WaveformTable):
        struct_arr = _lgdo_col_to_arrow(obj)
        table = pa.Table.from_batches([pa.RecordBatch.from_struct_array(struct_arr)])
        if obj.attrs:
            meta = {k: _serialize_attr(v) for k, v in obj.attrs.items()}
            table = table.replace_schema_metadata(meta)
        return table
    return _lgdo_col_to_arrow(obj)


def _lgdo_col_to_arrow(col) -> pa.Array:
    """Convert single LGDO column to Arrow array.

    Tables (including WaveformTable) become StructArrays with child field
    metadata preserving attrs like units.
    """
    if isinstance(col, Table):
        child_arrays = []
        child_fields = []
        for name, sub_col in col.items():
            child_arr = _lgdo_col_to_arrow(sub_col)
            meta = None
            if hasattr(sub_col, "attrs") and sub_col.attrs:
                meta = {k: _serialize_attr(v) for k, v in sub_col.attrs.items()}
            child_fields.append(pa.field(name, child_arr.type, metadata=meta))
            child_arrays.append(child_arr)
        return pa.StructArray.from_arrays(child_arrays, fields=child_fields)

    if isinstance(col, ArrayOfEqualSizedArrays):
        arr = pa.array(col.nda.ravel())
        for dim in reversed(col.nda.shape[1:]):
            arr = pa.FixedSizeListArray.from_arrays(arr, dim)
        return arr

    if isinstance(col, VectorOfVectors):
        return pa.ListArray.from_arrays(
            col._offsets.nda, _lgdo_col_to_arrow(col.flattened_data)
        )

    if isinstance(col, Array):
        return pa.array(col.nda)

    msg = f"Unsupported LGDO type: {type(col)}"
    raise TypeError(msg)


# ============ Arrow → LGDO ============


def arrow_to_lgdo(obj):
    """Convert an Arrow object to its LGDO equivalent.

    Type mapping:

    ==================================  ==========================
    Arrow type                          LGDO type
    ==================================  ==========================
    pa.Table                            Table
    StructArray with {t0, dt, values}   WaveformTable
    StructArray (other)                 Table
    FixedSizeListArray                  ArrayOfEqualSizedArrays
    ListArray                           VectorOfVectors
    primitive Array                     Array
    ==================================  ==========================

    Zero-copy where possible. Multi-chunk columns are combined
    automatically (with a warning, since this allocates).

    Parameters
    ----------
    obj
        Any supported Arrow object.

    Returns
    -------
    Table, WaveformTable, Array, ArrayOfEqualSizedArrays, or VectorOfVectors
    """
    if isinstance(obj, pa.Table):
        col_dict = {}
        for name in obj.column_names:
            field = obj.schema.field(name)
            col = obj.column(name)
            if col.num_chunks != 1:
                warnings.warn(
                    f"Column '{name}' has {col.num_chunks} chunks; "
                    "combining into one contiguous buffer (allocates memory)",
                    stacklevel=2,
                )
            col_dict[name] = _arrow_col_to_lgdo(col.combine_chunks(), field)
        attrs = (
            {k.decode(): _deserialize_attr(v) for k, v in obj.schema.metadata.items()}
            if obj.schema.metadata
            else None
        )
        return Table(col_dict=col_dict, attrs=attrs)

    if isinstance(obj, pa.ChunkedArray):
        if obj.num_chunks != 1:
            warnings.warn(
                f"ChunkedArray has {obj.num_chunks} chunks; "
                "combining into one contiguous buffer (allocates memory)",
                stacklevel=2,
            )
        return _arrow_col_to_lgdo(obj.combine_chunks(), None)

    return _arrow_col_to_lgdo(obj, None)


def _arrow_col_to_lgdo(col: pa.Array, field: pa.Field | None):
    """Convert Arrow array to LGDO column (zero-copy).

    StructArrays whose fields are {t0, dt, values} become WaveformTables;
    other StructArrays become plain Tables.
    """
    attrs = (
        {k.decode(): _deserialize_attr(v) for k, v in field.metadata.items()}
        if field and field.metadata
        else None
    )

    if isinstance(col.type, pa.StructType):
        col_dict = {}
        for i in range(col.type.num_fields):
            sub_field = col.type.field(i)
            col_dict[sub_field.name] = _arrow_col_to_lgdo(
                col.field(sub_field.name), sub_field
            )

        if col_dict.keys() == {"t0", "dt", "values"}:
            # t0 and dt need to be writable as required by dspeed.build_processing_chain
            t0 = col_dict["t0"]
            dt = col_dict["dt"]
            return WaveformTable(
                t0=Array(nda=np.array(t0.nda, copy=True), attrs=t0.attrs),
                dt=Array(nda=np.array(dt.nda, copy=True), attrs=dt.attrs),
                values=col_dict["values"],
                attrs=attrs,
            )

        return Table(col_dict=col_dict, attrs=attrs)

    if isinstance(col.type, pa.FixedSizeListType):
        return ArrayOfEqualSizedArrays(nda=_nested_fixed_list_to_nda(col), attrs=attrs)

    if isinstance(col.type, pa.ListType):
        offsets = col.offsets.to_numpy(zero_copy_only=True, writable=False)

        if isinstance(
            col.values.type, (pa.ListType, pa.FixedSizeListType, pa.StructType)
        ):
            flattened = _arrow_col_to_lgdo(col.values, None)
        else:
            flattened = col.values.to_numpy(zero_copy_only=False, writable=False)

        return VectorOfVectors(flattened_data=flattened, offsets=offsets, attrs=attrs)

    return Array(nda=col.to_numpy(zero_copy_only=False, writable=False), attrs=attrs)


def _nested_fixed_list_to_nda(arr: pa.Array) -> np.ndarray:
    """Convert nested Arrow fixed_size_list to N-D numpy array."""
    dims = []
    while isinstance(arr.type, pa.FixedSizeListType):
        dims.append(arr.type.list_size)
        arr = arr.values
    return arr.to_numpy(zero_copy_only=False, writable=False).reshape(-1, *dims)
