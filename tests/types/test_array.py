from __future__ import annotations

import awkward as ak
import numpy as np
import pandas as pd
import pint
import pytest

from lgdo import Array


def test_datatype_name():
    array = Array()
    assert array.datatype_name() == "array"


def test_form_datatype():
    array = Array(shape=(12, 34))
    assert array.form_datatype() == "array<2>{real}"


def test_init():
    attrs = {"attr1": 1}
    array = Array(shape=(3,), dtype=np.float32, fill_val=42, attrs=attrs)
    assert (array.nda == np.full((3,), 42, np.float32)).all()
    assert array.attrs == attrs | {"datatype": "array<1>{real}"}


def test_resize():
    array = Array(nda=np.array([1, 2, 3, 4]))
    array.resize(3)
    assert (array.nda == np.array([1, 2, 3])).all()


def test_insert():
    a = Array(np.array([1, 2, 3, 4]))
    a.insert(2, [-1, -1])
    assert a == Array([1, 2, -1, -1, 3, 4])


def test_view():
    a = Array(np.array([1, 2, 3, 4]), attrs={"units": "m"})

    v = a.view_as("np", with_units=True)
    assert isinstance(v, pint.Quantity)
    assert v.u == "meter"
    assert np.array_equal(v.m, a.nda)

    v = a.view_as("np", with_units=False)
    assert isinstance(v, np.ndarray)

    v = a.view_as("pd", with_units=True)
    assert isinstance(v, pd.Series)
    assert v.dtype == "meter"

    v = a.view_as("pd", with_units=False)
    assert v.dtype == "int64"

    v = a.view_as("ak", with_units=False)
    assert isinstance(v, ak.Array)

    with pytest.raises(ValueError):
        a.view_as("ak", with_units=True)
