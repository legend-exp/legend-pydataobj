from __future__ import annotations

import h5py
import numpy as np

from lgdo import utils


def test_get_element_type():
    # variable length HD5 string datatype.
    h5py_str_dtype = h5py.string_dtype(encoding="ascii", length=None)

    objs = [
        ("hi", "string"),
        (True, "bool"),
        (np.void(0), "blob"),
        (0, "real"),
        (np.uint8(0), "real"),
        (float(0), "real"),
        (1 + 1j, "complex"),
        (b"hi", "string"),
        (np.array(["hi"]), "string"),
        (np.array([b"hi"], h5py_str_dtype), "string"),
    ]

    for obj, name in objs:
        get_name = utils.get_element_type(obj)
        assert get_name == name
