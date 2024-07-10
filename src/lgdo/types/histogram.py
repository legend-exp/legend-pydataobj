from __future__ import annotations

from typing import Any

import hist
import numpy as np

from .array import Array
from .lgdo import LGDO
from .scalar import Scalar
from .struct import Struct


class HistogramAxis(Struct):
    def __init__(
        self, first: float, last: float, step: float, closedleft: bool = True
    ) -> None:
        """
        A special struct to group axis parameters for use in a :class:`Histogram`.

        Parameters
        ----------
        first
            left edge of the leftmost bin
        last
            right edge of the rightmost bin
        step
            step size (width of each bin)
        closedleft
            if True, the bin intervals are left-closed :math:`[a,b)`;
            if False, intervals are right-closed :math:`(a,b]`.
        """
        super().__init__(
            {
                "binedges": Struct(
                    {"first": Scalar(first), "last": Scalar(last), "step": Scalar(step)}
                ),
                "closedleft": Scalar(closedleft),
            },
        )

    @classmethod
    def from_edges(cls, edges: np.ndarray) -> HistogramAxis:
        edge_diff = np.diff(edges)
        if np.any(~np.isclose(edge_diff, edge_diff[0])):
            msg = "only evenly distributed edges can be converted"
            raise ValueError(msg)
        return cls(edges[0], edges[-1], edge_diff[0], True)

    @property
    def first(self) -> float:
        return self["binedges"]["first"].value

    @property
    def last(self) -> float:
        return self["binedges"]["last"].value

    @property
    def step(self) -> float:
        return self["binedges"]["step"].value

    @property
    def closedleft(self) -> bool:
        return self["closedleft"].value

    @property
    def nbins(self) -> int:
        bins = (self.last - self.first) / self.step
        bins_int = int(np.rint(bins))
        assert np.isclose(bins, bins_int)
        return bins_int

    def __str__(self) -> str:
        return (
            f"first={self.first}, last={self.last}, step={self.step}, "
            + f"closedleft={self.closedleft}"
        )


class Histogram(Struct):
    def __init__(
        self,
        weights: hist.Hist | np.ndarray,
        binning: None
        | list[HistogramAxis]
        | list[np.ndarray]
        | list[tuple[float, float, float, bool]] = None,
        isdensity: bool = False,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """A special struct to contain histogrammed data.

        Parameters
        ----------
        weights
            An :class:`numpy.ndarray` to be used for this object's internal
            array. Note: the array is used directly, not copied; or a :class:`hist.Hist`
            object, that will be copied and not used directly.
        binning
            * has to by None if a :class:`hist.Hist` has been passed as ``weights``
            * can be a list of pre-initialized :class:`HistogramAxes`
            * can be a list of tuples, representing arguments to :meth:`HistogramAxes.__init__()`
            * can be a list of numpy arrays, as returned by :func:`numpy.histogramdd`.
        """
        if isinstance(weights, hist.Hist):
            if binning is not None:
                msg = "not allowed to pass custom binning if constructing from hist.Hist instance"
                raise ValueError(msg)
            if isdensity:
                msg = "not allowed to pass isdensity=True if constructing from hist.Hist instance"
                raise ValueError(msg)

            if weights.sum(flow=True) != weights.sum(flow=False):
                msg = "flow bins of hist.Hist cannopt be represented"
                raise ValueError(msg)
            w = Array(weights.view(flow=False))

            b = []
            for ax in weights.axes:
                if not isinstance(ax, hist.axis.Regular):
                    msg = "only regular axes of hist.Hist can be converted"
                    raise ValueError(msg)
                b.append(HistogramAxis.from_edges(ax.edges))
            b = self._create_binning(b)
        else:
            if binning is None:
                msg = "need to also pass binning if passing histogram as array"
                raise ValueError(msg)
            w = Array(weights)

            if all(isinstance(ax, HistogramAxis) for ax in binning):
                b = binning
            elif all(isinstance(ax, np.ndarray) for ax in binning):
                b = [HistogramAxis.from_edges(ax) for ax in binning]
            elif all(isinstance(ax, tuple) for ax in binning):
                b = [HistogramAxis(*ax) for ax in binning]
            else:
                msg = "invalid binning object passed"
                raise ValueError(msg)

            if len(binning) != len(weights.shape):
                msg = "binning and weight dimensions do not match"
                raise ValueError(msg)
            for i, ax in enumerate(b):
                if ax.nbins != weights.shape[i]:
                    msg = f"bin count does not match weight count along axis {i}"
                    raise ValueError(msg)
            b = self._create_binning(b)

        super().__init__(
            {"binning": b, "weights": w, "isdensity": Scalar(isdensity)},
            attrs,
        )

    def _create_binning(self, axes: list[HistogramAxis]) -> Struct:
        return Struct({f"axis_{i}": a for i, a in enumerate(axes)})

    @property
    def isdensity(self) -> bool:
        return self["isdensity"].value

    @property
    def weights(self) -> Array:
        return self["weights"]

    @property
    def binning(self) -> tuple[HistogramAxis, ...]:
        bins = sorted(self["binning"].items())
        assert all(isinstance(v, HistogramAxis) for k, v in bins)
        return tuple(v for _, v in bins)

    def __setitem__(self, name: str, obj: LGDO) -> None:
        # do not allow for new attributes on this
        msg = "histogram fields cannot be mutated"
        raise TypeError(msg)

    def __getattr__(self, name: str) -> None:
        # do not allow for new attributes on this
        msg = "histogram fields cannot be mutated"
        raise TypeError(msg)

    def add_field(self, name: str | int, obj: LGDO) -> None:  # noqa: ARG002
        msg = "histogram fields cannot be mutated"
        raise TypeError(msg)

    def remove_field(self, name: str | int, delete: bool = False) -> None:  # noqa: ARG002
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
    ) -> tuple[np.ndarray] | hist.Hist:
        r"""View the histogram data as a third-party format data structure.

        This is typically a zero-copy or nearly zero-copy operation.

        Supported third-party formats are:

        - ``np``: returns a tuple of binning and an :class:`np.ndarray`
        - ``hist``: returns an :class:`hist.Hist`

        Notes
        -----
        For the return value of the ``np`` output type, see :func:`numpy.histogramdd`.

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
                hist_axes.append(
                    hist.axis.Regular(
                        bins=a.nbins,
                        start=a.first,
                        stop=a.last,
                        underflow=False,
                        overflow=False,
                    )
                )

            hist_hist = hist.Hist(*hist_axes)
            hist_hist[:] = self.weights.view_as("np")
            return hist_hist

        if library == "np":
            edges = []
            for a in self.binning:
                edges.append(np.linspace(a.first, a.last, a.nbins))
            return self.weights.view_as("np"), tuple(edges)

        msg = f"{library!r} is not a supported third-party format."
        raise TypeError(msg)
