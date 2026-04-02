from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

from lgdo import ArrayOfDetectorIDs, VectorOfVectors


def test_datatype_name():
    array = ArrayOfDetectorIDs()
    assert array.datatype_name() == "array"


def test_form_datatype():
    array = ArrayOfDetectorIDs(shape=(12, 34))
    assert array.form_datatype() == "array<2>{detectorid}"
    assert array.dtype == np.uint32


def test_init():
    attrs = {"attr1": 1}
    with pytest.raises(ValueError):
        ArrayOfDetectorIDs(nda=np.full((3,), 42, np.float32), attrs=attrs)
    array = ArrayOfDetectorIDs(shape=(3,), fill_val=42, attrs=attrs)
    assert (array.nda == np.full((3,), 42, np.uint32)).all()
    assert array.attrs == attrs | {"datatype": "array<1>{detectorid}"}

    # test 2D array
    array = ArrayOfDetectorIDs(shape=(3, 4), fill_val=42, attrs=attrs)
    assert (array.nda == np.full((3, 4), 42, np.uint32)).all()
    assert array.attrs == attrs | {"datatype": "array<2>{detectorid}"}

    # test VectorOfVectors
    array = VectorOfVectors(
        flattened_data=ArrayOfDetectorIDs(shape=(12,), fill_val=42),
        cumulative_length=np.array([4, 8, 12]),
        attrs=attrs,
    )
    assert (array.flattened_data.nda == np.full((12,), 42, np.uint32)).all()
    assert (array.cumulative_length.nda == np.array([4, 8, 12])).all
    assert array.attrs == attrs | {"datatype": "array<1>{array<1>{detectorid}}"}

    # test Awkward array input
    array = ArrayOfDetectorIDs(
        nda=ak.with_parameter(np.array([1, 2, 3], dtype=np.uint32), "units", "mm"),
        attrs=attrs,
    )
    assert (array.nda == np.array([1, 2, 3], dtype=np.uint32)).all()
    assert array.attrs == attrs | {"datatype": "array<1>{detectorid}", "units": "mm"}
