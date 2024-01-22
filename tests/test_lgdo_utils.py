from __future__ import annotations

import numpy as np

from lgdo import utils


def test_get_element_type():
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
    ]

    for obj, name in objs:
        get_name = utils.get_element_type(obj)
        assert get_name == name
