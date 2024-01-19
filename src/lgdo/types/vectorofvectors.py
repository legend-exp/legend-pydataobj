"""
Implements a LEGEND Data Object representing a variable-length array of
variable-length arrays and corresponding utilities.
"""
from __future__ import annotations

import itertools
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

    For now only a 1D vector of 1D vectors is supported. Internal representation
    is as two NumPy arrays, one to store the flattened data contiguosly and one
    to store the cumulative sum of lengths of each vector.
    """

    def __init__(
        self,
        array: ak.Array | list[list[int | float]] = None,
        flattened_data: Array | NDArray = None,
        cumulative_length: Array | NDArray = None,
        shape_guess: tuple[int, int] | None = None,
        dtype: DTypeLike = None,
        fill_val: int | float | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        array
            create a ``VectorOfVectors`` out of a Python list of lists or an
            :class:`ak.Array`. Takes priority over `flattened_data` and
            `cumulative_length`.
        flattened_data
            if not ``None``, used as the internal array for
            `self.flattened_data`.  Otherwise, an internal `flattened_data` is
            allocated based on `cumulative_length` (or `shape_guess`) and `dtype`.
        cumulative_length
            if not ``None``, used as the internal array for
            `self.cumulative_length`. Should be `dtype` :any:`numpy.uint32`. If
            `cumulative_length` is ``None``, an internal `cumulative_length` is
            allocated based on the first element of `shape_guess`.
        shape_guess
            a NumPy-format shape specification, required if either of
            `flattened_data` or `cumulative_length` are not supplied.  The
            first element should not be a guess and sets the number of vectors
            to be stored. The second element is a guess or approximation of the
            typical length of a stored vector, used to set the initial length
            of `flattened_data` if it was not supplied.
        dtype
            sets the type of data stored in `flattened_data`. Required if
            `flattened_data` and `array` are ``None``.
        fill_val
            fill all of `self.flattened_data` with this value.
        attrs
            a set of user attributes to be carried along with this LGDO.
        """
        if array is not None:
            if isinstance(array, ak.Array):
                if array.ndim != 2:
                    msg = (
                        "cannot initialize a VectorOfVectors with "
                        f"{array.ndim}-dimensional data"
                    )
                    raise ValueError(msg)

                form, length, container = ak.to_buffers(array)

                self.__init__(
                    flattened_data=container["node1-data"],
                    cumulative_length=container["node0-offsets"][1:],
                )

            else:
                cl_nda = np.cumsum([len(ll) for ll in array])
                if dtype is None:
                    if len(cl_nda) == 0 or cl_nda[-1] == 0:
                        msg = "array can't be empty with dtype=None!"
                        raise ValueError(msg)

                    # Set dtype from the first element in the list
                    # Find it efficiently, allowing for zero-length lists as some of the entries
                    first_element = next(itertools.chain.from_iterable(array))
                    dtype = type(first_element)

                self.dtype = np.dtype(dtype)
                self.cumulative_length = Array(cl_nda)
                self.flattened_data = Array(
                    np.fromiter(itertools.chain.from_iterable(array), dtype=self.dtype)
                )

        else:
            if cumulative_length is None:
                if shape_guess is None:
                    # just make an empty vector
                    self.cumulative_length = Array(np.empty((0,), dtype="uint32"))
                # initialize based on shape_guess
                elif shape_guess[1] <= 0:
                    self.cumulative_length = Array(
                        shape=(shape_guess[0],), dtype="uint32", fill_val=0
                    )
                else:
                    self.cumulative_length = Array(
                        np.arange(
                            shape_guess[1],
                            np.prod(shape_guess) + 1,
                            shape_guess[1],
                            dtype="uint32",
                        )
                    )
            else:
                self.cumulative_length = Array(cumulative_length)

            if flattened_data is None:
                if dtype is None:
                    msg = "flattened_data and dtype cannot both be None!"
                    raise ValueError(msg)

                length = 0
                if cumulative_length is None:
                    # just make an empty vector or use shape_guess
                    length = 0 if shape_guess is None else np.prod(shape_guess)
                else:
                    # use cumulative_length
                    length = cumulative_length[-1]

                self.flattened_data = Array(
                    shape=(length,), dtype=dtype, fill_val=fill_val
                )
            else:
                self.flattened_data = Array(flattened_data)

            # finally set dtype
            self.dtype = self.flattened_data.dtype

        super().__init__(attrs)

    def datatype_name(self) -> str:
        return "array"

    def form_datatype(self) -> str:
        et = utils.get_element_type(self)
        return "array<1>{array<1>{" + et + "}}"

    def __len__(self) -> int:
        """Return the number of stored vectors."""
        return len(self.cumulative_length)

    def __eq__(self, other: VectorOfVectors) -> bool:
        if isinstance(other, VectorOfVectors):
            return (
                self.cumulative_length == other.cumulative_length
                and (
                    len(self.cumulative_length) == 0
                    or np.all(
                        self.flattened_data[: self.cumulative_length[-1]]
                        == other.flattened_data[: other.cumulative_length[-1]]
                    )
                )
                and self.dtype == other.dtype
                and self.attrs == other.attrs
            )

        return False

    def __getitem__(self, i: int) -> list:
        """Return vector at index `i`."""
        stop = self.cumulative_length[i]
        if i in (0, -len(self)):
            return self.flattened_data[0:stop]

        return self.flattened_data[self.cumulative_length[i - 1] : stop]

    def __setitem__(self, i: int, new: NDArray) -> None:
        self.__getitem__(i)[:] = new

    def resize(self, new_size: int) -> None:
        """Resize vector along the first axis.

        `self.flattened_data` is resized only if `new_size` is smaller than the
        current vector length.

        If `new_size` is larger than the current vector length,
        `self.cumulative_length` is padded with its last element.  This
        corresponds to appending empty vectors.

        Examples
        --------
        >>> vov = VectorOfVectors([[1, 2, 3], [4, 5]])
        >>> vov.resize(3)
        >>> print(vov)
        [[1 2 3],
         [4 5],
         [],
        ]

        >>> vov = VectorOfVectors([[1, 2], [3], [4, 5]])
        >>> vov.resize(2)
        >>> print(vov)
        [[1 2],
         [3],
        ]
        """

        vidx = self.cumulative_length
        old_s = len(self)
        dlen = new_size - old_s
        csum = vidx[-1] if len(self) > 0 else 0

        # first resize the cumulative length
        self.cumulative_length.resize(new_size)

        # if new_size > size, new elements are filled with zeros, let's fix
        # that
        if dlen > 0:
            self.cumulative_length[old_s:] = csum

        # then resize the data array
        # if dlen > 0 this has no effect
        if len(self.cumulative_length) > 0:
            self.flattened_data.resize(self.cumulative_length[-1])

    def append(self, new: NDArray) -> None:
        """Append a 1D vector `new` at the end.

        Examples
        --------
        >>> vov = VectorOfVectors([[1, 2, 3], [4, 5]])
        >>> vov.append([8, 9])
        >>> print(vov)
        [[1 2 3],
         [4 5],
         [8 9],
        ]
        """
        # first extend cumulative_length by +1
        self.cumulative_length.resize(len(self) + 1)
        # set it at the right value
        newlen = self.cumulative_length[-2] + len(new) if len(self) > 1 else len(new)
        self.cumulative_length[-1] = newlen
        # then resize flattened_data to accommodate the new vector
        self.flattened_data.resize(len(self.flattened_data) + len(new))
        # finally set it
        self[-1] = new

    def insert(self, i: int, new: NDArray) -> None:
        """Insert a vector at index `i`.

        `self.flattened_data` (and therefore `self.cumulative_length`) is
        resized in order to accommodate the new element.

        Examples
        --------
        >>> vov = VectorOfVectors([[1, 2, 3], [4, 5]])
        >>> vov.insert(1, [8, 9])
        >>> print(vov)
        [[1 2 3],
         [8 9],
         [4 5],
        ]

        Warning
        -------
        This method involves a significant amount of memory re-allocation and
        is expected to perform poorly on large vectors.
        """
        if i >= len(self):
            msg = f"index {i} is out of bounds for vector owith size {len(self)}"
            raise IndexError(msg)

        self.flattened_data = Array(
            np.insert(self.flattened_data, self.cumulative_length[i - 1], new)
        )
        self.cumulative_length = Array(
            np.insert(self.cumulative_length, i, self.cumulative_length[i - 1])
        )
        self.cumulative_length[i:] += np.uint32(len(new))

    def replace(self, i: int, new: NDArray) -> None:
        """Replace the vector at index `i` with `new`.

        `self.flattened_data` (and therefore `self.cumulative_length`) is
        resized, if the length of `new` is different from the vector currently
        at index `i`.

        Examples
        --------
        >>> vov = VectorOfVectors([[1, 2, 3], [4, 5]])
        >>> vov.replace(0, [8, 9])
        >>> print(vov)
        [[8 9],
         [4 5],
        ]

        Warning
        -------
        This method involves a significant amount of memory re-allocation and
        is expected to perform poorly on large vectors.
        """
        if i >= len(self):
            msg = f"index {i} is out of bounds for vector with size {len(self)}"
            raise IndexError(msg)

        vidx = self.cumulative_length
        dlen = len(new) - len(self[i])

        if dlen == 0:
            # don't waste resources
            self[i] = new
        elif dlen < 0:
            start = vidx[i - 1]
            stop = start + len(new)
            # set the already allocated indices
            self.flattened_data[start:stop] = new
            # then delete the extra indices
            self.flattened_data = Array(
                np.delete(self.flattened_data, np.s_[stop : vidx[i]])
            )
        else:
            # set the already allocated indices
            self.flattened_data[vidx[i - 1] : vidx[i]] = new[: len(self[i])]
            # then insert the remaining
            self.flattened_data = Array(
                np.insert(self.flattened_data, vidx[i], new[len(self[i]) :])
            )

        vidx[i:] = vidx[i:] + dlen

    def _set_vector_unsafe(self, i: int, vec: NDArray, lens: NDArray = None) -> None:
        r"""Insert vector `vec` at position `i`.

        Assumes that ``j = self.cumulative_length[i-1]`` is the index (in
        `self.flattened_data`) of the end of the `(i-1)`\ th vector and copies
        `vec` in ``self.flattened_data[j:sum(lens)]``. Finally updates
        ``self.cumulative_length[i]`` with the new flattened data array length.

        Vectors stored after index `i` can be overridden, producing unintended
        behavior. This method is typically used for fast sequential fill of a
        pre-allocated vector of vectors.

        If vec is 1D array and lens is None, set using full array. If vec
        is 2D, require lens to be included, and fill each array only up to
        lengths in lens.

        Danger
        ------
        This method can lead to undefined behavior or vector invalidation if
        used improperly. Use it only if you know what you are doing.

        See Also
        --------
        append, replace, insert
        """
        start = 0 if i == 0 else self.cumulative_length[i - 1]
        if len(vec.shape) == 1:
            vec = np.expand_dims(vec, axis=0)
            if lens is None:
                lens = np.array([vec.shape[1]], dtype="u4")
        if not isinstance(lens, np.ndarray):
            lens = np.array([lens], dtype="u4")
        cum_lens = start + lens.cumsum()
        _nb_fill(vec, lens, self.flattened_data.nda[start : cum_lens[-1]])
        self.cumulative_length[i : i + len(lens)] = cum_lens

    def __iter__(self) -> Iterator[NDArray]:
        for j, stop in enumerate(self.cumulative_length):
            if j == 0:
                yield self.flattened_data[0:stop]
            else:
                yield self.flattened_data[self.cumulative_length[j - 1] : stop]

    def __str__(self) -> str:
        string = ""
        pos = 0
        for vec in self:
            if pos != 0:
                string += " "

            string += np.array2string(vec, prefix=" ")

            if pos < len(self.cumulative_length):
                string += ",\n"

            pos += 1

        string = f"[{string}]"

        tmp_attrs = self.attrs.copy()
        tmp_attrs.pop("datatype")
        if len(tmp_attrs) > 0:
            string += f" with attrs={tmp_attrs}"

        return string

    def __repr__(self) -> str:
        npopt = np.get_printoptions()
        np.set_printoptions(threshold=5, edgeitems=2, linewidth=100)
        out = (
            "VectorOfVectors(flattened_data="
            + repr(self.flattened_data)
            + ", cumulative_length="
            + repr(self.cumulative_length)
            + ", attrs="
            + repr(self.attrs)
            + ")"
        )
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
        ak_arr = self.view_as("ak")

        if max_len is None:
            max_len = int(ak.max(ak.count(ak_arr, axis=-1)))

        nda = ak.fill_none(ak.pad_none(ak_arr, max_len, clip=True), fill_val).to_numpy(
            allow_missing=False
        )

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

            # cannot avoid making a copy here. we should add the leading 0 to
            # cumulative_length inside VectorOfVectors at some point in the
            # future
            offsets = np.empty(
                len(self.cumulative_length) + 1, dtype=self.cumulative_length.dtype
            )
            offsets[1:] = self.cumulative_length
            offsets[0] = 0

            layout = ak.contents.ListOffsetArray(
                offsets=ak.index.Index(offsets),
                content=ak.contents.NumpyArray(self.flattened_data),
            )
            return ak.Array(layout)

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

            return akpd.from_awkward(self.view_as("ak"))

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
