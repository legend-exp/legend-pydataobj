"""Implements utilities for LEGEND Data Objects."""
from __future__ import annotations

import logging

import numpy as np

from . import types as lgdo

log = logging.getLogger(__name__)


def get_element_type(obj: object) -> str:
    """Get the LGDO element type of a scalar or array.

    For use in LGDO datatype attributes.

    Parameters
    ----------
    obj
        if a ``str``, will automatically return ``string`` if the object has
        a :class:`numpy.dtype`, that will be used for determining the element
        type otherwise will attempt to case the type of the object to a
        :class:`numpy.dtype`.

    Returns
    -------
    element_type
        A string stating the determined element type of the object.
    """

    # special handling for strings
    if isinstance(obj, str):
        return "string"

    # the rest use dtypes
    dt = obj.dtype if hasattr(obj, "dtype") else np.dtype(type(obj))
    kind = dt.kind

    if kind == "b":
        return "bool"
    if kind == "V":
        return "blob"
    if kind in ["i", "u", "f"]:
        return "real"
    if kind == "c":
        return "complex"
    if kind in ["S", "U"]:
        return "string"

    # couldn't figure it out
    raise ValueError(
        "cannot determine lgdo element_type for object of type", type(obj).__name__
    )


def copy(obj: lgdo.LGDO, dtype: np.dtype = None) -> lgdo.LGDO:
    """Return a copy of an LGDO.

    Parameters
    ----------
    obj
        the LGDO to be copied.
    dtype
        NumPy dtype to be used for the copied object.

    """
    if dtype is None:
        dtype = obj.dtype

    if isinstance(obj, lgdo.Array):
        return lgdo.Array(
            np.array(obj.nda, dtype=dtype, copy=True), attrs=dict(obj.attrs)
        )

    if isinstance(obj, lgdo.VectorOfVectors):
        return lgdo.VectorOfVectors(
            flattened_data=copy(obj.flattened_data, dtype=dtype),
            cumulative_length=copy(obj.cumulative_length),
            attrs=dict(obj.attrs),
        )

    else:
        raise ValueError(f"copy of {type(obj)} not supported")
