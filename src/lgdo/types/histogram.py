from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import hist
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .array import Array
from .lgdo import LGDO
from .scalar import Scalar
from .struct import Struct

log = logging.getLogger(__name__)


class Histogram(Struct):
    class Axis(Struct):
        def __init__(
            self,
            edges: NDArray | Array | None,
            first: float | None,
            last: float | None,
            step: float | None,
            closedleft: bool = True,
            binedge_attrs: dict[str, Any] | None = None,
        ) -> None:
            """
            A special struct to group axis parameters for use in a :class:`Histogram`.

            Depending on the parameters, an axis either can have

            * a binning described by a range object, if ``first``, ``last`` and ``step``
              are passed, or
            * a variable binning described by the ``edges`` array.

            Parameters
            ----------
            edges
                an array of edges that describe the binning of this axis.
            first
                left edge of the leftmost bin
            last
                right edge of the rightmost bin
            step
                step size (width of each bin)
            closedleft
                if True, the bin intervals are left-closed :math:`[a,b)`;
                if False, intervals are right-closed :math:`(a,b]`.
            binedge_attrs
                attributes that will be added to the ``binedges`` LGDO that
                is part of the axis struct.
            """
            if edges is not None and (
                first is not None or last is not None or step is not None
            ):
                msg = "can only construct Axis either from edges or from range"
                raise ValueError(msg)
            if edges is None and (first is None or last is None or step is None):
                msg = "did not pass all range parameters"
                raise ValueError(msg)

            if edges is None:
                edges = Struct(
                    {
                        "first": Scalar(first),
                        "last": Scalar(last),
                        "step": Scalar(step),
                    },
                    binedge_attrs,
                )
            else:
                if not isinstance(edges, Array):
                    edges = Array(edges, attrs=binedge_attrs)
                elif binedge_attrs is not None:
                    msg = "passed both binedge as Array LGDO instance and binedge_attrs"
                    raise ValueError(msg)

                if len(edges.nda.shape) != 1:
                    msg = "must pass an array<1>{real} as edges vector"
                    raise ValueError(msg)

            super().__init__({"binedges": edges, "closedleft": Scalar(closedleft)})

        @classmethod
        def from_edges(
            cls,
            edges: NDArray | Iterable[float],
            binedge_attrs: dict[str, Any] | None = None,
        ) -> Histogram.Axis:
            """Create a new axis with variable binning described by ``edges``."""
            edges = np.array(edges)
            return cls(edges, None, None, None, True, binedge_attrs)

        @classmethod
        def from_range_edges(
            cls,
            edges: NDArray | Iterable[float],
            binedge_attrs: dict[str, Any] | None = None,
        ) -> Histogram.Axis:
            """Create a new axis from the binning described by ``edges``, but try to convert it to
            a evenly-spaced range object first.

            .. warning ::

                This function might return a wrong binning, especially in the case of very small
                magnitudes of the spacing. See the documentation of :func:`numpy.isclose` for
                details. Use this function only with caution, if you know the binning's order of
                magniutude.
            """
            edges = np.array(edges)
            edge_diff = np.diff(edges)
            if np.any(~np.isclose(edge_diff, edge_diff[0])):
                return cls(edges, None, None, None, True, binedge_attrs)
            return cls(None, edges[0], edges[-1], edge_diff[0], True, binedge_attrs)

        @property
        def is_range(self) -> bool:
            return isinstance(self["binedges"], Struct)

        @property
        def first(self) -> float:
            if not self.is_range:
                msg = "Axis is not a range"
                raise TypeError(msg)
            return self["binedges"]["first"].value

        @property
        def last(self) -> float:
            if not self.is_range:
                msg = "Axis is not a range"
                raise TypeError(msg)
            return self["binedges"]["last"].value

        @property
        def step(self) -> float:
            if not self.is_range:
                msg = "Axis is not a range"
                raise TypeError(msg)
            return self["binedges"]["step"].value

        @property
        def closedleft(self) -> bool:
            return self["closedleft"].value

        @property
        def nbins(self) -> int:
            """Return the number of bins, both for variable and range binning."""
            if self.is_range:
                bins = (self.last - self.first) / self.step
                bins_int = int(np.rint(bins))
                assert np.isclose(bins, bins_int)
                return bins_int
            return len(self["binedges"].nda) - 1

        @property
        def edges(self) -> NDArray:
            """Return all binedges, both for variable and range binning."""
            if self.is_range:
                return np.linspace(self.first, self.last, self.nbins + 1)
            return self["binedges"].nda

        def __str__(self) -> str:
            thr_orig = np.get_printoptions()["threshold"]
            np.set_printoptions(threshold=8)

            if self.is_range:
                string = f"first={self.first}, last={self.last}, step={self.step}"
            else:
                string = f"edges={self.edges}"
            string += f", closedleft={self.closedleft}"

            attrs = self.get_binedgeattrs()
            if attrs:
                string += f" with attrs={attrs}"

            np.set_printoptions(threshold=thr_orig)
            return string

        def get_binedgeattrs(self, datatype: bool = False) -> dict:
            """Return a copy of the LGDO attributes dictionary of the binedges

            Parameters
            ----------
            datatype
                if ``False``, remove ``datatype`` attribute from the output
                dictionary.
            """
            return self["binedges"].getattrs(datatype)

    def __init__(
        self,
        weights: hist.Hist | NDArray | Array,
        binning: None
        | Iterable[Histogram.Axis]
        | Iterable[NDArray]
        | Iterable[tuple[float, float, float]] = None,
        isdensity: bool = False,
        attrs: dict[str, Any] | None = None,
        binedge_attrs: dict[str, Any] | None = None,
        flow: bool = True,
    ) -> None:
        """A special struct to contain histogrammed data.

        Parameters
        ----------
        weights
            An :class:`numpy.ndarray` to be used for this object's internal
            array, or a :class:`hist.Hist` object, whose data view is used
            for this object's internal array.
            Note: the array/histogram view is used directly, not copied
        binning
            * has to by None if a :class:`hist.Hist` has been passed as ``weights``
            * can be a list of pre-initialized :class:`Histogram.Axis`
            * can be a list of tuples, each representing a range, ``(first, last, step)``
            * can be a list of numpy arrays, as returned by :func:`numpy.histogramdd`.
        isdensity
            If True, all bin contents represent a density (amount per volume), and not
            an absolute amount.
        binedge_attrs
            attributes that will be added to the all ``binedges`` of all axes.
            This does not work if :class:`Histogram.Axis` instances are directly passed
            as binning.
        attrs
            a set of user attributes to be carried along with this LGDO.
        flow
            If ``False``, discard counts in over-/underflow bins of the passed
            :class:`hist.Hist` instance. If ``True``, this data will also be discarded,
            but a warning is emitted.

            .. note ::

                :class:`Histogram` does not support storing counts in overflow or
                underflow bins. This parameter just controls, whether a warning will
                be emitted.
        """
        if isinstance(weights, hist.Hist):
            if binning is not None:
                msg = "not allowed to pass custom binning if constructing from hist.Hist instance"
                raise ValueError(msg)
            if isdensity:
                msg = "not allowed to pass isdensity=True if constructing from hist.Hist instance"
                raise ValueError(msg)

            if weights.sum(flow=True) != weights.sum(flow=False) and flow:
                log.warning(
                    "flow bins of hist.Hist cannot be represented, their counts are discarded"
                )
            weights_view = weights.view(flow=False)
            if type(weights_view) is not np.ndarray:
                msg = "only simple numpy-backed storages can be used in a hist.Hist"
                raise ValueError(msg)
            w = Array(weights_view)

            b = []
            for ax in weights.axes:
                if not isinstance(ax, (hist.axis.Regular, hist.axis.Variable)):
                    msg = "only regular or variable axes of hist.Hist can be converted"
                    raise ValueError(msg)
                if isinstance(ax, hist.axis.Regular):
                    step = (ax.edges[-1] - ax.edges[0]) / ax.size
                    bax = Histogram.Axis(
                        None, ax.edges[0], ax.edges[-1], step, True, binedge_attrs
                    )
                    b.append(bax)
                else:
                    b.append(Histogram.Axis.from_edges(ax.edges, binedge_attrs))
        else:
            if binning is None:
                msg = "need to pass binning to construct Histogram"
                raise ValueError(msg)

            # set up binning
            if all(isinstance(ax, Histogram.Axis) for ax in binning):
                if binedge_attrs is not None:
                    msg = "passed both binedges as Axis instances and binedge_attrs"
                    raise ValueError(msg)
                b = binning
            elif all(isinstance(ax, np.ndarray) for ax in binning):
                b = [Histogram.Axis.from_edges(ax, binedge_attrs) for ax in binning]
            elif all(isinstance(ax, tuple) for ax in binning):
                b = [Histogram.Axis(None, *ax, True, binedge_attrs) for ax in binning]
            else:
                msg = "invalid binning object passed"
                raise ValueError(msg)

            # set up bin weights
            if isinstance(weights, Array):
                w = weights
            elif weights is None:
                w = Array(shape=[ax.nbins for ax in b], fill_val=0, dtype=np.float32)
            else:
                w = Array(weights)

            if len(binning) != len(w.nda.shape):
                msg = "binning and weight dimensions do not match"
                raise ValueError(msg)
            for i, ax in enumerate(b):
                if ax.nbins != w.nda.shape[i]:
                    msg = f"bin count does not match weight count along axis {i}"
                    raise ValueError(msg)

        b = Struct({f"axis_{i}": a for i, a in enumerate(b)})

        super().__init__(
            {"binning": b, "weights": w, "isdensity": Scalar(isdensity)},
            attrs,
        )

    @property
    def isdensity(self) -> bool:
        return self["isdensity"].value

    @property
    def weights(self) -> Array:
        return self["weights"]

    @property
    def binning(self) -> tuple[Histogram.Axis, ...]:
        bins = sorted(self["binning"].items())
        assert all(isinstance(v, Histogram.Axis) for k, v in bins)
        return tuple(v for _, v in bins)

    def fill(self, data, w: NDArray = None, keys: Sequence[str] = None) -> None:
        """Fill histogram by incrementing bins with data points weighted by w

        Parameters
        ----------
        data
            a ndarray with inner dimension equal to number of axes, or a list
            of equal-length 1d-arrays containing data for each axis, or a
            Mapping to 1d-arrays containing data for each axis (requires keys),
            or a Pandas dataframe (optionally takes a list of keys)
        w
            weight to use for incrementing data points. If None, use 1 for all
        keys
            list of keys to use if data is a pandas ''DataFrame'' or ''Mapping''
        """
        if keys is not None:
            if isinstance(keys, str):
                keys = [keys]
            elif not isinstance(keys, list):
                keys = list(keys)

        if (
            isinstance(data, np.ndarray)
            and len(data.shape) == 1
            and len(self.binning) == 1
        ):
            N = len(data)
            data = [data]
        elif (
            isinstance(data, np.ndarray)
            and len(data.shape) == 2
            and data.shape[1] == len(self.binning)
        ):
            N = data.shape[0]
            data = data.T
        elif isinstance(data, pd.DataFrame) and (
            (keys is not None and len(keys) == len(self.binning))
            or data.ndim == len(self.binning)
        ):
            if keys is not None:
                data = data[keys]
            N = len(data)
            data = data.values.T
        elif isinstance(data, Sequence) and len(data) == len(self.binning):
            data = [d if isinstance(d, np.ndarray) else np.array(d) for d in data]
            N = len(data[0])
            if not all(len(d) == N for d in data):
                msg = "length of all data arrays must be equal"
                raise ValueError(msg)
        elif isinstance(data, Mapping):
            if not isinstance(keys, Sequence) or len(keys) != len(self.binning):
                msg = "filling hist with Mapping data requires a list of keys with same length as histogram rank"
                raise ValueError(msg)
            data = [
                data[k] if isinstance(data[k], np.ndarray) else np.array(data[k])
                for k in keys
            ]
            N = len(data[0])
            if not all(len(d) == N for d in data):
                msg = "length of all data arrays must be equal"
                raise ValueError(msg)
        else:
            msg = "data must be 2D numpy array or list of 1D arrays with length equal to number of axes"
            raise ValueError(msg)

        idx = np.zeros(N, np.float64)  # bin indices for flattened array
        oor_mask = np.ones(N, np.bool_)  # mask to remove out of range values
        stride = [s // self.weights.dtype.itemsize for s in self.weights.nda.strides]
        for col, ax, s in zip(data, self.binning, stride):
            if ax.is_range:
                idx += s * np.floor((col - ax.first) / ax.step - int(not ax.closedleft))
                if ax.closedleft:
                    oor_mask &= (ax.first <= col) & (col < ax.last)
                else:
                    oor_mask &= (ax.first < col) & (col <= ax.last)
            else:
                idx += s * (
                    np.searchsorted(
                        ax.edges, col, side=("right" if ax.closedleft else "left")
                    )
                    - 1
                )
                if ax.closedleft:
                    oor_mask &= (ax.edges[0] <= col) & (col < ax.edges[-1])
                else:
                    oor_mask &= (ax.edges[0] < col) & (col <= ax.edges[-1])

        # increment bin contents
        idx = idx[oor_mask].astype(np.int64)
        w = w[oor_mask] if w is not None else 1
        np.add.at(self.weights.nda.reshape(-1), idx, w)

    def __setitem__(self, name: str, obj: LGDO) -> None:
        # do not allow for new attributes on this
        known_keys = ("binning", "weights", "isdensity")
        if name in known_keys and not dict.__contains__(self, name):
            # but allow initialization while unpickling (after __init__() this is unreachable)
            dict.__setitem__(self, name, obj)
        else:
            msg = "histogram fields cannot be mutated "
            raise AttributeError(msg)

    def __getattr__(self, name: str) -> None:
        # do not allow for new attributes on this
        msg = "histogram fields cannot be mutated"
        raise AttributeError(msg)

    def add_field(self, name: str | int, obj: LGDO) -> None:  # noqa: ARG002
        """
        .. error ::

            Not applicable: A histogram cannot be used as a struct
        """
        msg = "histogram fields cannot be mutated"
        raise TypeError(msg)

    def remove_field(self, name: str | int, delete: bool = False) -> None:  # noqa: ARG002
        """
        .. error ::

            Not applicable: A histogram cannot be used as a struct
        """
        msg = "histogram fields cannot be mutated"
        raise TypeError(msg)

    def __str__(self) -> str:
        string = "{\n"
        for k, v in enumerate(self.binning):
            string += f" 'axis_{k}': {v},\n"
        string += "}"

        attrs = self.getattrs()
        if attrs:
            string += f" with attrs={attrs}"

        return string

    def view_as(
        self,
        library: str,
    ) -> tuple[NDArray] | hist.Hist:
        r"""View the histogram data as a third-party format data structure.

        This is typically a zero-copy or nearly zero-copy operation.

        Supported third-party formats are:

        - ``np``: returns a tuple of binning and an :class:`np.ndarray`, similar
          to the return value of :func:`numpy.histogramdd`.
        - ``hist``: returns an :class:`hist.Hist` that holds **a copy** of this
          histogram's data.

        Warning
        -------
        Viewing as ``hist`` will perform a copy of the stored histogram data.

        Parameters
        ----------
        library
            format of the returned data view.

        See Also
        --------
        .LGDO.view_as
        """
        if library == "hist":
            if self.isdensity:
                msg = "hist.Hist cannot represent density histograms"
                raise ValueError(msg)

            hist_axes = []
            for a in self.binning:
                if not a.closedleft:
                    msg = "hist.Hist cannot represent right-closed intervals"
                    raise ValueError(msg)
                if a.is_range:
                    hist_ax = hist.axis.Regular(
                        bins=a.nbins,
                        start=a.first,
                        stop=a.last,
                        underflow=False,
                        overflow=False,
                    )
                else:
                    hist_ax = hist.axis.Variable(
                        a.edges,
                        underflow=False,
                        overflow=False,
                    )
                hist_axes.append(hist_ax)

            return hist.Hist(*hist_axes, data=self.weights.view_as("np"))

        if library == "np":
            edges = tuple([a.edges for a in self.binning])
            return self.weights.view_as("np"), edges

        msg = f"{library!r} is not a supported third-party format."
        raise TypeError(msg)
