"""
Implements a LEGEND Data Object representing a variable-length array of
variable-length arrays and corresponding utilities.
"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

import awkward as ak
import awkward_pandas as akpd
import numba
import numpy as np
import pandas as pd
from numpy.typing import DTypeLike, NDArray

from .. import utils
from ..utils import numba_defaults_kwargs as nb_kwargs
from . import arrayofequalsizedarrays as aoesa
from .array import Array
from .lgdo import LGDO

log = logging.getLogger(__name__)


class VectorOfVectors(LGDO):
    """A variable-length array of variable-length arrays.

    For now only a 1D vector of 1D vectors is supported. Internal
    representation is as :class:`ak.Array`.
    """

    def __init__(
        self,
        data: Any = None,
        flattened_data: Array | NDArray = None,
        cumulative_length: Array | NDArray = None,
        dtype: DTypeLike = None,
        fill_val: int | float | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        data
            any data container type accepted by the :class:`awkward.Array`
            constructor, interpretable as a 2D jagged vector. Takes
            priority over `flattened_data` and `cumulative_length`.
        flattened_data
            if not ``None``, interpret this 1D sequence as a flattened vector
            of vectors and use it together with `cumulative_length` to
            initialize (zero-copy) the internal :class:`awkward.Array`.  If
            ``None``, a dummy `flattened_data` is allocated internally based on
            `cumulative_length`, `dtype` and `fill_val`.
        cumulative_length
            if not ``None``, interpret as an array of (integer) offsets of the
            2D vector (see above).
        dtype
            sets the type of data stored in the vector. Required if `data` and
            `flattened_data` are ``None``.
        fill_val
            fill the vector with this value.
        attrs
            a set of user attributes to be carried along with this LGDO.
        """
        # TODO: what to do with the LGDO attributes carried by flattened_data
        # and cumulative_length?

        # let's give priority to the "data" argument. this will make a copy
        if data is not None:
            self.nda = ak.Array(data)
            if self.nda.ndim != 2:
                msg = "only two-dimensional data structures are supported"
                raise ValueError(msg)

            try:
                # ...is this OK?
                self.dtype = ak.types.numpytype.primitive_to_dtype(
                    self.nda.type.content.content.primitive
                )
            except AttributeError as err:
                msg = "input data does not look like a 2D jagged vector"
                raise ValueError(msg) from err

        # otherwise see if we can use flattened_data and cumulative_length as
        # buffers (aka zero-copy)
        elif flattened_data is not None and cumulative_length is not None:
            self.nda = _ak_from_buffers(flattened_data, cumulative_length)
            self.dtype = flattened_data.dtype
        elif flattened_data is not None and cumulative_length is None:
            msg = "cumulative_length must be always specified, when flattened_data is"
            raise ValueError(msg)
        elif flattened_data is None and cumulative_length is not None:
            if dtype is None:
                msg = "flattened_data and dtype cannot both be None!"
                raise ValueError(msg)

            if fill_val is not None:
                flattened_data = np.full(
                    shape=(cumulative_length[-1],), dtype=dtype, fill_value=fill_val
                )
            else:
                flattened_data = np.empty(shape=(cumulative_length[-1],), dtype=dtype)

            # finally build awkward array
            self.nda = _ak_from_buffers(flattened_data, cumulative_length)

            # finally set dtype
            self.dtype = flattened_data.dtype
        else:
            self.nda = ak.Array(np.array([], dtype=dtype))
            self.dtype = np.dtype(dtype)

        super().__init__(attrs)

    def datatype_name(self) -> str:
        return "array"

    def form_datatype(self) -> str:
        et = utils.get_element_type(self)
        return "array<1>{array<1>{" + et + "}}"

    @property
    def cumulative_length(self) -> Array:
        return Array(self.nda.layout.offsets.data[1:])

    @property
    def flattened_data(self) -> Array:
        # HACK: seems like resizing the Awkward array down does not result in a
        # resized flattened_data, so this is just a precaution
        return Array(self.nda.layout.content.data[: self.cumulative_length[-1]])

    def __len__(self) -> int:
        """Return the number of stored vectors."""
        return len(self.nda)

    def __eq__(self, other: VectorOfVectors) -> bool:
        if isinstance(other, VectorOfVectors):
            els_equal = False
            with contextlib.suppress(ValueError):
                els_equal = ak.all(self.nda == other.nda)

            return els_equal and self.dtype == other.dtype and self.attrs == other.attrs

        return False

    def __getitem__(self, i: int) -> ak.Array:
        """Return vector at index `i`."""
        return self.nda.__getitem__(i)

    def resize(self, new_size: int) -> None:
        """Resize **down** vector along the first axis."""
        if new_size > len(self):
            msg = f"cannot resize with new_size > len(vov) ({new_size} > {len(self)})"
            raise ValueError(msg)

        self.nda = self.nda[:new_size]

    def __iter__(self) -> Iterator[ak.Array]:
        return self.nda.__iter__()

    def __str__(self) -> str:
        string = self.nda.show(stream=None)
        tmp_attrs = self.attrs.copy()
        tmp_attrs.pop("datatype")
        if len(tmp_attrs) > 0:
            string += f" with attrs={tmp_attrs}"

        return string

    def __repr__(self) -> str:
        npopt = np.get_printoptions()
        np.set_printoptions(threshold=5, edgeitems=2, linewidth=100)
        out = f"VectorOfVectors({self.nda}, attrs=" + repr(self.attrs) + ")"
        np.set_printoptions(**npopt)
        return out

    def to_aoesa(
        self,
        max_len: int | None = None,
        fill_val: bool | int | float = np.nan,
        preserve_dtype: bool = False,
    ) -> aoesa.ArrayOfEqualSizedArrays:
        """Convert to :class:`ArrayOfEqualSizedArrays`.

        Note
        ----
        The dtype of the original vector is typically not strictly preserved.
        The output dtype will be either :class:`np.float64` or :class:`np.int64`.
        If you want to use the same exact dtype, set `preserve_dtype` to
        ``True``.

        Parameters
        ----------
        max_len
            the length of the returned array along its second dimension. Longer
            vectors will be truncated, shorter will be padded with `fill_val`.
            If ``None``, the length will be equal to the length of the longest
            vector.
        fill_val
            value used to pad shorter vectors up to `max_len`. The dtype of the
            output array will be such that both `fill_val` and the vector
            values can be represented in the same data structure.
        preserve_dtype
            whether the output array should have exactly the same dtype as the
            original vector of vectors. The type `fill_val` must be a
            compatible one.
        """
        if max_len is None:
            max_len = int(ak.max(ak.count(self.nda, axis=-1)))

        nda = ak.fill_none(
            ak.pad_none(self.nda, max_len, clip=True), fill_val
        ).to_numpy(allow_missing=False)

        if preserve_dtype:
            nda = nda.astype(self.flattened_data.dtype, copy=False)

        return aoesa.ArrayOfEqualSizedArrays(nda=nda, attrs=self.getattrs())

    def view_as(
        self,
        library: str,
        with_units: bool = False,
        fill_val: bool | int | float = np.nan,
        preserve_dtype: bool = False,
    ) -> pd.DataFrame | np.NDArray | ak.Array:
        r"""View the vector data as a third-party format data structure.

        This is typically a zero-copy or nearly zero-copy operation.

        Supported third-party formats are:

        - ``pd``: returns a :class:`pandas.Series` (supported through the
          ``awkward-pandas`` package)
        - ``np``: returns a :class:`numpy.ndarray`, padded with zeros to make
          it rectangular. This implies memory re-allocation.
        - ``ak``: returns an :class:`ak.Array`. ``self.cumulative_length`` is
          currently re-allocated for technical reasons.

        Notes
        -----
        Awkward array views partially involve memory re-allocation (the
        `cumulative_length`\ s), while NumPy "exploded" views clearly imply a
        full copy.

        Parameters
        ----------
        library
            format of the returned data view.
        with_units
            forward physical units to the output data.
        fill_val
            forwarded to :meth:`.to_aoesa`, if `library` is ``np``.
        preserve_dtype
            forwarded to :meth:`.to_aoesa`, if `library` is ``np``.

        See Also
        --------
        .LGDO.view_as
        """
        attach_units = with_units and "units" in self.attrs

        if library == "ak":
            if attach_units:
                msg = "Pint does not support Awkward yet, you must view the data with_units=False"
                raise ValueError(msg)

            return self.nda

        if library == "np":
            if preserve_dtype:
                return self.to_aoesa(fill_val=fill_val, preserve_dtype=True).view_as(
                    "np", with_units=with_units
                )

            return self.to_aoesa().view_as("np", with_units=with_units)

        if library == "pd":
            if attach_units:
                msg = "Pint does not support Awkward yet, you must view the data with_units=False"
                raise ValueError(msg)

            return akpd.from_awkward(self.nda)

        msg = f"{library} is not a supported third-party format."
        raise ValueError(msg)


def build_cl(
    sorted_array_in: NDArray, cumulative_length_out: NDArray = None
) -> NDArray:
    """Build a cumulative length array from an array of sorted data.

    Examples
    --------
    >>> build_cl(np.array([3, 3, 3, 4])
    array([3., 4.])

    For a `sorted_array_in` of indices, this is the inverse of
    :func:`.explode_cl`, in the sense that doing
    ``build_cl(explode_cl(cumulative_length))`` would recover the original
    `cumulative_length`.

    Parameters
    ----------
    sorted_array_in
        array of data already sorted; each N matching contiguous entries will
        be converted into a new row of `cumulative_length_out`.
    cumulative_length_out
        a pre-allocated array for the output `cumulative_length`. It will
        always have length <= `sorted_array_in`, so giving them the same length
        is safe if there is not a better guess.

    Returns
    -------
    cumulative_length_out
        the output cumulative length array. If the user provides a
        `cumulative_length_out` that is too long, this return value is sliced
        to contain only the used portion of the allocated memory.
    """
    if len(sorted_array_in) == 0:
        return None
    sorted_array_in = np.asarray(sorted_array_in)
    if cumulative_length_out is None:
        cumulative_length_out = np.zeros(len(sorted_array_in), dtype=np.uint64)
    else:
        cumulative_length_out.fill(0)
    if len(cumulative_length_out) == 0 and len(sorted_array_in) > 0:
        msg = "cumulative_length_out too short ({len(cumulative_length_out)})"
        raise ValueError(msg)
    return _nb_build_cl(sorted_array_in, cumulative_length_out)


@numba.njit(**nb_kwargs)
def _nb_build_cl(sorted_array_in: NDArray, cumulative_length_out: NDArray) -> NDArray:
    """numbified inner loop for build_cl"""
    ii = 0
    last_val = sorted_array_in[0]
    for val in sorted_array_in:
        if val != last_val:
            ii += 1
            cumulative_length_out[ii] = cumulative_length_out[ii - 1]
            if ii >= len(cumulative_length_out):
                msg = "cumulative_length_out too short"
                raise RuntimeError(msg)
            last_val = val
        cumulative_length_out[ii] += 1
    ii += 1
    return cumulative_length_out[:ii]


@numba.guvectorize(
    [
        f"{data_type}[:,:],{size_type}[:],{data_type}[:]"
        for data_type in [
            "b1",
            "i1",
            "i2",
            "i4",
            "i8",
            "u1",
            "u2",
            "u4",
            "u8",
            "f4",
            "f8",
            "c8",
            "c16",
        ]
        for size_type in ["i4", "i8", "u4", "u8"]
    ],
    "(l,m),(l),(n)",
    **nb_kwargs,
)
def _nb_fill(aoa_in: NDArray, len_in: NDArray, flattened_array_out: NDArray):
    """Vectorized function to fill flattened array from array of arrays and
    lengths. Values in aoa_in past lengths will not be copied.

    Parameters
    ----------
    aoa_in
        array of arrays containing values to be copied
    len_in
        array of vector lengths for each row of aoa_in
    flattened_array_out
        flattened array to copy values into. Must be longer than sum of
        lengths in len_in
    """

    if len(flattened_array_out) < len_in.sum():
        msg = "flattened array not large enough to hold values"
        raise ValueError(msg)

    start = 0
    for i, ll in enumerate(len_in):
        stop = start + ll
        flattened_array_out[start:stop] = aoa_in[i, :ll]
        start = stop


def explode_cl(cumulative_length: NDArray, array_out: NDArray = None) -> NDArray:
    """Explode a `cumulative_length` array.

    Examples
    --------
    >>> explode_cl(np.array([2, 3]))
    array([0., 0., 1.])

    This is the inverse of :func:`.build_cl`, in the sense that doing
    ``build_cl(explode_cl(cumulative_length))`` would recover the original
    `cumulative_length`.

    Parameters
    ----------
    cumulative_length
        the cumulative length array to be exploded.
    array_out
        a pre-allocated array to hold the exploded cumulative length array.
        The length should be equal to ``cumulative_length[-1]``.

    Returns
    -------
    array_out
        the exploded cumulative length array.
    """
    cumulative_length = np.asarray(cumulative_length)
    out_len = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    if array_out is None:
        array_out = np.empty(int(out_len), dtype=np.uint64)
    if len(array_out) != out_len:
        msg = f"bad lengths: cl[-1] ({cumulative_length[-1]}) != out ({len(array_out)})"
        raise ValueError(msg)
    return _nb_explode_cl(cumulative_length, array_out)


@numba.njit(**nb_kwargs)
def _nb_explode_cl(cumulative_length: NDArray, array_out: NDArray) -> NDArray:
    """numbified inner loop for explode_cl"""
    out_len = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    if len(array_out) != out_len:
        msg = "bad lengths"
        raise ValueError(msg)
    start = 0
    for ii in range(len(cumulative_length)):
        nn = int(cumulative_length[ii] - start)
        for jj in range(nn):
            array_out[int(start + jj)] = ii
        start = cumulative_length[ii]
    return array_out


def explode(
    cumulative_length: NDArray, array_in: NDArray, array_out: NDArray = None
) -> NDArray:
    """Explode a data array using a `cumulative_length` array.

    This is identical to :func:`.explode_cl`, except `array_in` gets exploded
    instead of `cumulative_length`.

    Examples
    --------
    >>> explode(np.array([2, 3]), np.array([3, 4]))
    array([3., 3., 4.])

    Parameters
    ----------
    cumulative_length
        the cumulative length array to use for exploding.
    array_in
        the data to be exploded. Must have same length as `cumulative_length`.
    array_out
        a pre-allocated array to hold the exploded data. The length should be
        equal to ``cumulative_length[-1]``.

    Returns
    -------
    array_out
        the exploded cumulative length array.
    """
    cumulative_length = np.asarray(cumulative_length)
    array_in = np.asarray(array_in)
    out_len = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    if array_out is None:
        array_out = np.empty(out_len, dtype=array_in.dtype)
    if len(cumulative_length) != len(array_in) or len(array_out) != out_len:
        msg = (
            f"bad lengths: cl ({len(cumulative_length)}) != in ({len(array_in)}) "
            f"and cl[-1] ({cumulative_length[-1]}) != out ({len(array_out)})"
        )
        raise ValueError(msg)
    return nb_explode(cumulative_length, array_in, array_out)


@numba.njit(**nb_kwargs)
def nb_explode(
    cumulative_length: NDArray, array_in: NDArray, array_out: NDArray
) -> NDArray:
    """Numbified inner loop for :func:`.explode`."""
    out_len = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    if len(cumulative_length) != len(array_in) or len(array_out) != out_len:
        msg = "bad lengths"
        raise ValueError(msg)
    ii = 0
    for jj in range(len(array_out)):
        while ii < len(cumulative_length) and jj >= cumulative_length[ii]:
            ii += 1
        array_out[jj] = array_in[ii]
    return array_out


def explode_arrays(
    cumulative_length: Array,
    arrays: list[NDArray],
    arrays_out: list[NDArray] | None = None,
) -> list:
    """Explode a set of arrays using a `cumulative_length` array.

    Parameters
    ----------
    cumulative_length
        the cumulative length array to use for exploding.
    arrays
        the data arrays to be exploded. Each array must have same length as
        `cumulative_length`.
    arrays_out
        a list of pre-allocated arrays to hold the exploded data. The length of
        the list should be equal to the length of `arrays`, and each entry in
        arrays_out should have length ``cumulative_length[-1]``. If not
        provided, output arrays are allocated for the user.

    Returns
    -------
    arrays_out
        the list of exploded cumulative length arrays.
    """
    cumulative_length = np.asarray(cumulative_length)
    for ii in range(len(arrays)):
        arrays[ii] = np.asarray(arrays[ii])
    out_len = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    if arrays_out is None:
        arrays_out = []
        for array in arrays:
            arrays_out.append(np.empty(out_len, dtype=array.dtype))
    for ii in range(len(arrays)):
        explode(cumulative_length, arrays[ii], arrays_out[ii])
    return arrays_out


def _ak_from_buffers(
    flattened_data: Array | NDArray, cumulative_length: Array | NDArray
):
    if isinstance(flattened_data, Array):
        flattened_data = flattened_data.nda
    if isinstance(cumulative_length, Array):
        cumulative_length = cumulative_length.nda

    offsets = np.empty(len(cumulative_length) + 1, dtype=cumulative_length.dtype)
    offsets[1:] = cumulative_length
    offsets[0] = 0

    layout = ak.contents.ListOffsetArray(
        offsets=ak.index.Index(offsets), content=ak.contents.NumpyArray(flattened_data)
    )
    return ak.Array(layout)
