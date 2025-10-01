from __future__ import annotations

import awkward as ak
import numpy as np
import pandas as pd
import pint
import pytest

import lgdo


def test_datatype_name():
    aoesa = lgdo.ArrayOfEqualSizedArrays()
    assert aoesa.datatype_name() == "array_of_equalsized_arrays"


def test_form_datatype():
    aoesa = lgdo.ArrayOfEqualSizedArrays(dims=(2, 3))
    assert aoesa.form_datatype() == "array_of_equalsized_arrays<2,3>{real}"


def test_init():
    attrs = {"attr1": 1}
    aoesa = lgdo.ArrayOfEqualSizedArrays(
        dims=(2, 3), dtype=np.float32, fill_val=42, attrs=attrs
    )
    assert aoesa.dims == (2, 3)
    assert (aoesa.nda == np.full((2, 3), 42, np.float32)).all()
    assert aoesa.attrs == attrs | {"datatype": "array_of_equalsized_arrays<2,3>{real}"}


def test_to_vov():
    aoesa = lgdo.ArrayOfEqualSizedArrays(
        nda=np.array([[53, 91, 66, 58, 8], [78, 57, 66, 88, 73], [85, 99, 86, 68, 53]])
    )
    vov = aoesa.to_vov()
    assert isinstance(vov, lgdo.VectorOfVectors)
    assert np.array_equal(vov[0], [53, 91, 66, 58, 8])
    assert np.array_equal(vov[1], [78, 57, 66, 88, 73])
    assert np.array_equal(vov[2], [85, 99, 86, 68, 53])

    vov = aoesa.to_vov([2, 5, 6])
    assert np.array_equal(vov[0], [53, 91])
    assert np.array_equal(vov[1], [78, 57, 66])
    assert np.array_equal(vov[2], [85])


def test_view():
    aoesa = lgdo.ArrayOfEqualSizedArrays(
        nda=np.array([[53, 91, 66, 58, 8], [78, 57, 66, 88, 73], [85, 99, 86, 68, 53]]),
        attrs={"units": "m"},
    )

    v = aoesa.view_as("np", with_units=True)
    assert isinstance(v, pint.Quantity)
    assert v.u == "meter"
    assert np.array_equal(v.m, aoesa.nda)

    v = aoesa.view_as("np", with_units=False)
    assert isinstance(v, np.ndarray)

    v = aoesa.view_as("pd", with_units=False)
    assert isinstance(v, pd.Series)

    v = aoesa.view_as("ak", with_units=False)
    assert isinstance(v, ak.Array)

    with pytest.raises(ValueError):
        aoesa.view_as("pd", with_units=True)

    v = aoesa.view_as("ak", with_units=True)
    assert isinstance(v, ak.Array)
    assert ak.parameters(v) == {"units": "m"}
