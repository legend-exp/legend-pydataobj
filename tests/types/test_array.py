from __future__ import annotations

import pickle

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


def test_resize_and_capacity():
    array = Array(nda=np.array([1, 2, 3, 4]))
    assert array.get_capacity() == 4

    array.resize(3)
    assert array.get_capacity() == 4
    assert (array.nda == np.array([1, 2, 3])).all()

    array.resize(5)
    assert array.get_capacity() >= 5

    array.clear(trim=True)
    assert array.get_capacity() == 0
    assert len(array) == 0


def test_insert():
    a = Array(np.array([1, 2, 3, 4]))
    a.insert(2, [-1, -1])
    assert a == Array([1, 2, -1, -1, 3, 4])

    with pytest.raises(IndexError):
        a.insert(10, 10)


def test_append():
    a = Array(np.array([1, 2, 3, 4]))
    a.append(-1)
    assert a == Array([1, 2, 3, 4, -1])


def test_replace():
    a = Array(np.array([1, 2, 3, 4]))
    a.replace(2, -1)
    assert a == Array([1, 2, -1, 4])

    with pytest.raises(IndexError):
        a.replace(10, 10)


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


def test_pickle():
    obj = Array(nda=np.array([1, 2, 3, 4]))
    obj.attrs["attr1"] = 1

    ex = pickle.loads(pickle.dumps(obj))
    assert isinstance(ex, Array)
    assert ex.attrs["attr1"] == 1
    assert ex.attrs["datatype"] == obj.attrs["datatype"]
    assert np.all(ex.nda == np.array([1, 2, 3, 4]))
