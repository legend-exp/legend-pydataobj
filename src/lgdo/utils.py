"""Implements utilities for LEGEND Data Objects."""
from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping
from typing import Any, Iterator

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


def getenv_bool(name: str, default: bool = False) -> bool:
    """Get environment value as a boolean, returning True for 1, t and true
    (caps-insensitive), and False for any other value and default if undefined.
    """
    val = os.getenv(name)
    if not val:
        return default
    elif val.lower() in ("1", "t", "true"):
        return True
    else:
        return False


class NumbaDefaults(MutableMapping):
    """Bare-bones class to store some Numba default options. Defaults values
    are set from environment variables

    Examples
    --------
    Set all default option values for a processor at once by expanding the
    provided dictionary:

    >>> from numba import guvectorize
    >>> from lgdo.utils import numba_defaults_kwargs as nb_kwargs
    >>> @guvectorize([], "", **nb_kwargs, nopython=True) # def proc(...): ...

    Customize one argument but still set defaults for the others:

    >>> from lgdo.utils import numba_defaults as nb_defaults
    >>> @guvectorize([], "", **nb_defaults(cache=False) # def proc(...): ...

    Override global options at runtime:

    >>> from lgdo.utils import numba_defaults
    >>> # must set options before explicitly importing lgdo modules!
    >>> numba_defaults.cache = False
    >>> numba_defaults.boundscheck = True
    >>> from lgdo import compression # imports of numbified functions happen here
    >>> compression.encode(...)
    """

    def __init__(self) -> None:
        self.cache: bool = getenv_bool("LGDO_CACHE", default=True)
        self.boundscheck: bool = getenv_bool("LGDO_BOUNDSCHECK", default=False)

    def __getitem__(self, item: str) -> Any:
        return self.__dict__[item]

    def __setitem__(self, item: str, val: Any) -> None:
        self.__dict__[item] = val

    def __delitem__(self, item: str) -> None:
        del self.__dict__[item]

    def __iter__(self) -> Iterator:
        return self.__dict__.__iter__()

    def __len__(self) -> int:
        return len(self.__dict__)

    def __call__(self, **kwargs) -> dict:
        mapping = self.__dict__.copy()
        mapping.update(**kwargs)
        return mapping

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self.__dict__)


numba_defaults = NumbaDefaults()
numba_defaults_kwargs = numba_defaults
