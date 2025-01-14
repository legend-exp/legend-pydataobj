from __future__ import annotations

import pickle

import pytest

import lgdo


def test_datatype_name():
    scalar = lgdo.Scalar(value=42)
    assert scalar.datatype_name() == "real"


def test_form_datatype():
    scalar = lgdo.Scalar(value=42)
    assert scalar.form_datatype() == "real"


def test_init():
    attrs = {"attr1": 1}
    scalar = lgdo.Scalar(value=42, attrs=attrs)
    assert scalar.value == 42
    assert scalar.attrs == attrs | {"datatype": "real"}

    with pytest.raises(ValueError):
        lgdo.Scalar(value=42, attrs={"datatype": "string"})


def test_getattrs():
    scalar = lgdo.Scalar(value=42, attrs={"attr1": 1})
    assert scalar.getattrs() == {"attr1": 1}
    assert scalar.getattrs(True) == {"attr1": 1, "datatype": "real"}


def test_equality():
    assert lgdo.Scalar(value=42) == lgdo.Scalar(value=42)


def test_pickle():
    obj = lgdo.Scalar(value=10)
    obj.attrs["attr1"] = 1

    ex = pickle.loads(pickle.dumps(obj))
    assert isinstance(ex, lgdo.Scalar)
    assert ex.attrs["attr1"] == 1
    assert ex.attrs["datatype"] == obj.attrs["datatype"]
    assert ex.value == 10
