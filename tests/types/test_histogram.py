from __future__ import annotations

import hist
import numpy as np
import pytest

from lgdo import Histogram, Scalar


def test_init_hist():
    h = hist.Hist(
        hist.axis.Regular(bins=10, start=0, stop=1, name="x"),
        hist.axis.Regular(bins=10, start=0, stop=1, name="y"),
    )
    rng = np.random.default_rng()
    h.fill(rng.uniform(size=500), rng.uniform(size=500))
    h = Histogram(h, None)

    assert len(h.binning) == 2

    h = hist.Hist(hist.axis.Integer(start=0, stop=10))
    with pytest.raises(ValueError, match="only regular axes"):
        h = Histogram(h)

    h = hist.Hist(hist.axis.Regular(bins=10, start=0, stop=1))
    with pytest.raises(ValueError, match="isdensity=True"):
        Histogram(h, isdensity=True)
    with pytest.raises(ValueError, match="custom binning"):
        Histogram(h, binning=(np.array([0, 1, 2]),))

    h.fill([-1, 0.8, 2])
    with pytest.raises(ValueError, match="flow bins"):
        Histogram(h)

    # assert that the hist data is not copied into the LGDO.
    h = hist.Hist(hist.axis.Regular(bins=10, start=0, stop=10))
    h.fill([1, 2, 3])
    hi = Histogram(h)
    assert np.sum(hi.weights.nda) == 3
    h.fill([1, 2, 3])
    assert np.sum(hi.weights.nda) == 6


def test_init_np():
    h = Histogram(np.array([1, 1]), (np.array([0, 1, 2]),))
    assert len(h.binning) == 1
    assert not h.isdensity

    h = Histogram(np.array([1, 1]), (np.array([0, 1, 2]),), isdensity=True)
    assert h.isdensity

    with pytest.raises(ValueError, match="also pass binning"):
        h = Histogram(np.array([1, 1]), None)

    with pytest.raises(ValueError, match="dimensions do not match"):
        h = Histogram(np.array([1, 1]), (np.array([0, 1, 2]), np.array([0, 1, 2])))

    with pytest.raises(ValueError, match="bin count does not match weight count"):
        h = Histogram(
            np.array([[1, 1], [1, 1]]), (np.array([0, 1, 2]), np.array([0, 1]))
        )

    with pytest.raises(ValueError, match="only evenly"):
        h = Histogram(np.array([1, 1, 1]), (np.array([0, 1, 2.5, 3]),))


def test_datatype_name():
    h = Histogram(np.array([1, 1]), (np.array([0, 1, 2]),))
    assert h.datatype_name() == "struct"
    assert h.form_datatype() == "struct{binning,weights,isdensity}"


def test_axes():
    h = Histogram(
        np.array([[1, 1], [1, 1]]),
        (np.array([0, 1, 2]), np.array([2.1, 2.2, 2.3])),
        attrs={"units": "m"},
    )
    assert len(h.binning) == 2
    str(h)

    ax0 = h.binning[0]
    assert ax0.first == 0
    assert ax0.last == 2
    assert ax0.step == 1
    assert isinstance(ax0.nbins, int)
    assert str(ax0) == "first=0, last=2, step=1, closedleft=True"

    ax1 = h.binning[1]
    assert ax1.first == 2.1
    assert ax1.last == 2.3
    assert np.isclose(ax1.step, 0.1)
    assert isinstance(ax0.nbins, int)

    h = Histogram(np.array([[1, 1], [1, 1]]), [(1, 3, 1, True), (4, 6, 1, False)])
    ax0 = h.binning[0]
    str(h)

    h = Histogram(np.array([[1, 1], [1, 1]]), [(1, 3, 1, True), (4, 6, 1, False)])
    ax0 = h.binning[0]
    str(h)

    with pytest.raises(ValueError, match="invalid binning object"):
        h = Histogram(np.array([[1, 1], [1, 1]]), (range(2), range(2)))

    h = Histogram(
        np.array([[1, 1], [1, 1]]),
        [Histogram.Axis(1, 3, 1, True), Histogram.Axis(4, 6, 1, False)],
    )

    with pytest.raises(ValueError, match="invalid binning object"):
        h = Histogram(
            np.array([[1, 1], [1, 1]]),
            [(1, 3, 1, True), Histogram.Axis(4, 6, 1, False)],
        )


def test_view_as_hist():
    h = Histogram(np.array([1, 1]), (np.array([0, 1, 2]),))
    h.view_as("hist")

    h = Histogram(np.array([1, 1]), (np.array([0, 1, 2]),), isdensity=True)
    with pytest.raises(ValueError, match="cannot represent density"):
        h.view_as("hist")

    h = Histogram(np.array([[1, 1], [1, 1]]), [(1, 3, 1, True), (4, 6, 1, False)])
    with pytest.raises(ValueError, match="cannot represent right-closed"):
        h.view_as("hist")


def test_view_as_np():
    h = Histogram(np.array([1, 1]), (np.array([0, 1, 2]),))
    h.view_as("np")


def test_not_like_table():
    h = Histogram(np.array([1, 1]), (np.array([0, 1, 2]),))
    assert h.form_datatype() == "struct{binning,weights,isdensity}"
    with pytest.raises(TypeError):
        x = h.x  # noqa: F841
    with pytest.raises(TypeError):
        h["x"] = Scalar(1.0)
    with pytest.raises(TypeError):
        h.add_field("x", Scalar(1.0))
    with pytest.raises(TypeError):
        h.remove_field("binning")
