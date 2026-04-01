from __future__ import annotations

import numpy as np
import pytest

from lgdo import ArrayOfDetectorIDs


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
        array = ArrayOfDetectorIDs(nda=np.full((3,), 42, np.float32), attrs=attrs)
    array = ArrayOfDetectorIDs(shape=(3,), fill_val=42, attrs=attrs)
    assert (array.nda == np.full((3,), 42, np.uint32)).all()
    assert array.attrs == attrs | {"datatype": "array<1>{detectorid}"}
