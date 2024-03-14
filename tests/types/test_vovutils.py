from __future__ import annotations

from collections import namedtuple

import awkward as ak
import numpy as np
import pytest

from lgdo import VectorOfVectors
from lgdo.types import vovutils

VovColl = namedtuple("VovColl", ["v2d", "v3d", "v4d"])


@pytest.fixture()
def testvov():
    v2d = VectorOfVectors([[1, 2], [3, 4, 5], [2], [4, 8, 9, 7], [5, 3, 1]])
    v3d = VectorOfVectors([[[1, 2], [3, 4, 5]], [[2], [4, 8, 9, 7]], [[5, 3, 1]]])
    v4d = VectorOfVectors(
        [
            [[[1], [2]], [[3, 4], [5]]],
            [[[2, 6]], [[4, 8, 9, 7], [8, 3]]],
            [[[5, 3], [1]]],
        ]
    )

    return VovColl(v2d, v3d, v4d)


def test_ak_input_validity(testvov):
    for v in testvov:
        assert vovutils._ak_is_jagged(v) is True
        assert vovutils._ak_is_valid(v) is True

    assert vovutils._ak_is_jagged(ak.Array([[1], [1, 2], [1, 3, 4]])) is True
    assert vovutils._ak_is_jagged(ak.Array(np.empty(shape=(2, 3, 4)))) is False

    assert vovutils._ak_is_valid(ak.Array([[1], [1, 2], [1, 3, 4]])) is True
    assert vovutils._ak_is_valid(ak.Array(np.empty(shape=(2, 3, 4)))) is True
    assert vovutils._ak_is_valid(ak.Array([[1, None], [1, 2], [1, 3, 4]])) is False
    assert vovutils._ak_is_valid(ak.Array({"a": [1, 2], "b": [3, 4]})) is False


def test_build_cl_and_explodes():
    cl = np.array([3, 4], dtype=np.uint64)
    exp = np.array([0, 0, 0, 1], dtype=np.uint64)
    array = np.array([5, 7], dtype=np.uint64)
    array_exp = np.array([5, 5, 5, 7], dtype=np.uint64)
    # build_cl
    assert (vovutils.build_cl(exp, cl) == cl).all()
    assert (vovutils.build_cl(exp) == cl).all()
    assert (vovutils.build_cl([0, 0, 0, 1]) == cl).all()
    assert (vovutils.build_cl(array_exp, cl) == cl).all()
    assert (vovutils.build_cl(array_exp) == cl).all()
    assert (vovutils.build_cl([5, 5, 5, 7]) == cl).all()
    # explode_cl
    assert (vovutils.explode_cl(cl, exp) == exp).all()
    assert (vovutils.explode_cl(cl) == exp).all()
    assert (vovutils.explode_cl([3, 4]) == exp).all()
    # inverse functionality
    assert (vovutils.build_cl(vovutils.explode_cl(cl)) == cl).all()
    assert (vovutils.explode_cl(vovutils.build_cl(array_exp)) == exp).all()
    # explode
    assert (vovutils.explode(cl, array, array_exp) == array_exp).all()
    assert (vovutils.explode(cl, array) == array_exp).all()
    assert (vovutils.explode([3, 4], [5, 7]) == array_exp).all()
    assert (vovutils.explode(cl, range(len(cl))) == exp).all()
    # explode_arrays
    arrays_out = vovutils.explode_arrays(cl, [array, range(len(cl))])
    assert len(arrays_out) == 2
    assert (arrays_out[0] == array_exp).all()
    assert (arrays_out[1] == exp).all()
    arrays_out = vovutils.explode_arrays(
        cl, [array, range(len(cl))], arrays_out=arrays_out
    )
    assert len(arrays_out) == 2
    assert (arrays_out[0] == array_exp).all()
    assert (arrays_out[1] == exp).all()
