"""
Implements a LEGEND Data Object representing an n-dimensional array and
corresponding utilities.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import awkward as ak
import awkward_pandas as akpd
import numpy as np
import pandas as pd
import pint_pandas  # noqa: F401

from .. import utils
from ..units import default_units_registry as u
from .lgdo import LGDOCollection

log = logging.getLogger(__name__)


class Array(LGDOCollection):
    r"""Holds an :class:`numpy.ndarray` and attributes.

    :class:`Array` (and the other various array types) holds an `nda` instead
    of deriving from :class:`numpy.ndarray` for the following reasons:

    - It keeps management of the `nda` totally under the control of the user. The
      user can point it to another object's buffer, grab the `nda` and toss the
      :class:`Array`, etc.
    - It allows the management code to send just the `nda`'s the central routines
      for data manpulation. Keeping LGDO's out of that code allows for more
      standard, reusable, and (we expect) performant Python.
    - It allows the first axis of the `nda` to be treated as "special" for storage
      in :class:`.Table`\ s.
    """

    def __init__(
        self,
        nda: np.ndarray = None,
        shape: tuple[int, ...] = (),
        dtype: np.dtype = None,
        fill_val: float | int | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        nda
            An :class:`numpy.ndarray` to be used for this object's internal
            array. Note: the array is used directly, not copied. If not
            supplied, internal memory is newly allocated based on the shape and
            dtype arguments.
        shape
            A numpy-format shape specification for shape of the internal
            ndarray. Required if `nda` is ``None``, otherwise unused.
        dtype
            Specifies the type of the data in the array. Required if `nda` is
            ``None``, otherwise unused.
        fill_val
            If ``None``, memory is allocated without initialization. Otherwise,
            the array is allocated with all elements set to the corresponding
            fill value. If `nda` is not ``None``, this parameter is ignored.
        attrs
            A set of user attributes to be carried along with this LGDO.
        """
        if nda is None:
            if fill_val is None:
                nda = np.empty(shape, dtype=dtype)
            elif fill_val == 0:
                nda = np.zeros(shape, dtype=dtype)
            else:
                nda = np.full(shape, fill_val, dtype=dtype)

        elif isinstance(nda, Array):
            nda = nda.nda

        self.nda = nda

        super().__init__(attrs)

    def datatype_name(self) -> str:
        return "array"

    def form_datatype(self) -> str:
        dt = self.datatype_name()
        nd = str(len(self.nda.shape))
        et = utils.get_element_type(self)
        return dt + "<" + nd + ">{" + et + "}"

    def __len__(self) -> int:
        return self._size

    @property
    def nda(self):
        return self._nda[: self._size, ...] if self._nda.shape != () else self._nda

    @nda.setter
    def nda(self, value):
        self._nda = value if isinstance(value, np.ndarray) else np.array(value)
        self._size = len(self._nda) if self._nda.shape != () else 0

    @property
    def dtype(self):
        return self._nda.dtype

    @property
    def shape(self):
        return (len(self),) + self._nda.shape[1:]

    def reserve_capacity(self, capacity: int) -> None:
        "Set size (number of rows) of internal memory buffer"
        if capacity < len(self):
            msg = "Cannot reduce capacity below Array length"
            raise ValueError(msg)
        self._nda.resize((capacity,) + self._nda.shape[1:], refcheck=False)

    def get_capacity(self) -> int:
        "Get capacity (i.e. max size before memory must be re-allocated)"
        return len(self._nda)

    def trim_capacity(self) -> None:
        "Set capacity to be minimum needed to support Array size"
        self.reserve_capacity(np.prod(self.shape))

    def resize(self, new_size: int, trim=False) -> None:
        """Set size of Array in rows. Only change capacity if it must be
        increased to accommodate new rows; in this case double capacity.
        If trim is True, capacity will be set to match size."""

        self._size = new_size

        if trim and new_size != self.get_capacity:
            self.reserve_capacity(new_size)

        # If capacity is not big enough, set to next power of 2 big enough
        if new_size > self.get_capacity():
            self.reserve_capacity(int(2 ** (np.ceil(np.log2(new_size)))))

    def append(self, value: np.ndarray) -> None:
        "Append value to end of array (with copy)"
        self.insert(len(self), value)

    def insert(self, i: int, value: int | float) -> None:
        "Insert value into row i (with copy)"
        if i > len(self):
            msg = f"index {i} is out of bounds for array with size {len(self)}"
            raise IndexError(msg)

        value = np.array(value)
        if value.shape == self.shape[1:]:
            self.resize(len(self) + 1)
            self[i + 1 :] = self[i:-1]
            self[i] = value
        elif value.shape[1:] == self.shape[1:]:
            self.resize(len(self) + len(value))
            self[i + len(value) :] = self[i : -len(value)]
            self[i : i + len(value)] = value
        else:
            msg = f"Could not insert value with shape {value.shape} into Array with shape {self.shape}"
            raise ValueError(msg)

    def replace(self, i: int, value: int | float) -> None:
        "Replace value at row i"
        if i >= len(self):
            msg = f"index {i} is out of bounds for array with size {len(self)}"
            raise IndexError(msg)
        self[i] = value

    def __getitem__(self, key):
        return self.nda[key]

    def __setitem__(self, key, value):
        return self.nda.__setitem__(key, value)

    def __eq__(self, other: Array) -> bool:
        if isinstance(other, Array):
            return self.attrs == other.attrs and np.array_equal(self.nda, other.nda)

        return False

    def __iter__(self) -> Iterator:
        yield from self.nda

    def __str__(self) -> str:
        attrs = self.getattrs()
        string = str(self.nda)
        if attrs:
            string += f" with attrs={attrs}"
        return string

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + np.array2string(
                self.nda,
                prefix=self.__class__.__name__ + " ",
                formatter={
                    "int": lambda x: f"0x{x:02x}" if self.dtype == np.ubyte else str(x)
                },
            )
            + f", attrs={self.attrs!r})"
        )

    def view_as(
        self, library: str, with_units: bool = False
    ) -> pd.DataFrame | np.NDArray | ak.Array:
        """View the Array data as a third-party format data structure.

        This is a zero-copy operation. Supported third-party formats are:

        - ``pd``: returns a :class:`pandas.Series`
        - ``np``: returns the internal `nda` attribute (:class:`numpy.ndarray`)
        - ``ak``: returns an :class:`ak.Array` initialized with `self.nda`

        Parameters
        ----------
        library
            format of the returned data view.
        with_units
            forward physical units to the output data.

        See Also
        --------
        .LGDO.view_as
        """
        # TODO: does attaching units imply a copy?
        attach_units = with_units and "units" in self.attrs

        if library == "pd":
            if attach_units:
                if self.nda.ndim == 1:
                    return pd.Series(
                        self.nda, dtype=f"pint[{self.attrs['units']}]", copy=False
                    )

                msg = "Pint does not support Awkward yet, you must view the data with_units=False"
                raise ValueError(msg)

            if self.nda.ndim == 1:
                return pd.Series(self.nda, copy=False)

            # if array is multi-dim, use awkward
            return akpd.from_awkward(self.view_as("ak"))

        if library == "np":
            if attach_units:
                return self.nda * u(self.attrs["units"])

            return self.nda

        if library == "ak":
            if attach_units:
                msg = "Pint does not support Awkward yet, you must view the data with_units=False"
                raise ValueError(msg)

            # NOTE: this is zero-copy!
            return ak.Array(self.nda)

        msg = f"{library} is not a supported third-party format."
        raise ValueError(msg)
