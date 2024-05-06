from __future__ import annotations

import copy
from collections import namedtuple

import awkward as ak
import awkward_pandas as akpd
import numpy as np
import pandas as pd
import pint
import pytest

import lgdo
from lgdo import Array, VectorOfVectors, lh5

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


def test_init(testvov):
    for v in testvov:
        assert ak.is_valid(v.view_as("ak"))

    assert len(VectorOfVectors()) == 0
    assert len(VectorOfVectors(dtype="ubyte")) == 0
    assert VectorOfVectors(dtype="ubyte").flattened_data.dtype == "ubyte"

    v = VectorOfVectors(
        cumulative_length=np.array([5, 10, 15], dtype="uint32"), dtype="ubyte"
    )
    assert len(v.flattened_data) == 15
    assert len(v[-1]) == 5
    assert v.flattened_data.dtype == "ubyte"
    assert v.cumulative_length.dtype == "uint32"

    v = VectorOfVectors(
        flattened_data=Array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1]),
        cumulative_length=Array([2, 5, 6, 10, 13]),
    )
    assert v == testvov.v2d

    v = VectorOfVectors(ak.Array([[1, 2], [3, 4, 5], [2], [4, 8, 9, 7], [5, 3, 1]]))
    assert v == testvov.v2d

    # shape guess

    v = VectorOfVectors(shape_guess=(10, 20), dtype="int32", fill_val=2)
    assert v.flattened_data == lgdo.Array(shape=(10 * 20,), fill_val=2, dtype="int32")
    assert v.cumulative_length == lgdo.Array(
        np.arange(20, 10 * 20 + 1, 20, dtype="uint32")
    )
    assert ak.is_valid(v.view_as("ak"))

    v = VectorOfVectors(shape_guess=(5, 0), dtype="int32")
    assert v.cumulative_length == lgdo.Array([0, 0, 0, 0, 0])
    assert ak.is_valid(v.view_as("ak"))

    # multi-dimensional
    v = VectorOfVectors(shape_guess=(5, 3, 2), dtype="int16", fill_val=1)
    assert isinstance(v.flattened_data, VectorOfVectors)
    assert isinstance(v.flattened_data.flattened_data, Array)
    assert v.flattened_data.flattened_data.dtype == "int16"
    assert ak.is_valid(v.view_as("ak"))

    assert v.cumulative_length == lgdo.Array([3, 6, 9, 12, 15])
    assert v.flattened_data.cumulative_length == lgdo.Array(
        [i * 2 for i in range(1, 16)]
    )

    assert v == VectorOfVectors(
        [
            [[1, 1], [1, 1], [1, 1]],
            [[1, 1], [1, 1], [1, 1]],
            [[1, 1], [1, 1], [1, 1]],
            [[1, 1], [1, 1], [1, 1]],
            [[1, 1], [1, 1], [1, 1]],
        ],
        dtype="int16",
    )


def test_eq(testvov):
    assert testvov.v2d == VectorOfVectors(
        [[1, 2], [3, 4, 5], [2], [4, 8, 9, 7], [5, 3, 1]]
    )
    assert testvov.v3d == VectorOfVectors(
        [[[1, 2], [3, 4, 5]], [[2], [4, 8, 9, 7]], [[5, 3, 1]]]
    )
    assert testvov.v4d == VectorOfVectors(
        [
            [[[1], [2]], [[3, 4], [5]]],
            [[[2, 6]], [[4, 8, 9, 7], [8, 3]]],
            [[[5, 3], [1]]],
        ]
    )

    assert VectorOfVectors(
        [[1, 2], [3, 4, 5], [2], [4, 8, 9, 7], [5, 3, 1]], attrs={"ciao": "bello"}
    ) == VectorOfVectors(
        [[1, 2], [3, 4, 5], [2], [4, 8, 9, 7], [5, 3, 1]], attrs={"ciao": "bello"}
    )


def test_len(testvov):
    assert len(testvov.v2d) == 5
    assert len(testvov.v3d) == 3
    assert len(testvov.v4d) == 3

    assert testvov.v2d.ndim == 2
    assert testvov.v3d.ndim == 3
    assert testvov.v4d.ndim == 4


def test_serialization(testvov):
    assert testvov.v2d.cumulative_length == Array([2, 5, 6, 10, 13])
    assert testvov.v2d.flattened_data == Array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1])

    assert testvov.v3d.cumulative_length == Array([2, 4, 5])
    assert isinstance(testvov.v3d.flattened_data, VectorOfVectors)
    assert testvov.v3d.flattened_data.cumulative_length == Array([2, 5, 6, 10, 13])
    assert testvov.v3d.flattened_data.flattened_data == Array(
        [1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1]
    )

    assert isinstance(testvov.v4d.flattened_data, VectorOfVectors)
    assert isinstance(testvov.v4d.flattened_data.flattened_data, VectorOfVectors)
    assert isinstance(testvov.v4d.flattened_data.flattened_data, VectorOfVectors)


def test_datatype_name(testvov):
    for v in testvov:
        assert v.datatype_name() == "array"


def test_form_datatype(testvov):
    assert testvov.v2d.form_datatype() == "array<1>{array<1>{real}}"
    assert testvov.v3d.form_datatype() == "array<1>{array<1>{array<1>{real}}}"
    assert testvov.v4d.form_datatype() == "array<1>{array<1>{array<1>{array<1>{real}}}}"


def test_getitem(testvov):
    testvov = testvov.v2d

    desired = [
        np.array([1, 2]),
        np.array([3, 4, 5]),
        np.array([2]),
        np.array([4, 8, 9, 7]),
        np.array([5, 3, 1]),
    ]

    for i in range(len(desired)):
        assert np.array_equal(desired[i], testvov[i])

    assert np.array_equal(testvov[-1], desired[-1])
    assert np.array_equal(testvov[-2], desired[-2])

    v = VectorOfVectors([[1, 2]], dtype="uint32")
    assert np.array_equal(v[-1], [1, 2])


def test_resize(testvov):
    vov = testvov.v2d

    vov.resize(3)
    assert ak.is_valid(vov.view_as("ak"))
    assert len(vov.cumulative_length) == 3
    assert len(vov.flattened_data) == vov.cumulative_length[-1]
    assert vov == VectorOfVectors([[1, 2], [3, 4, 5], [2]])

    vov.resize(5)
    assert ak.is_valid(vov.view_as("ak"))
    assert len(vov) == 5
    assert len(vov[3]) == 0
    assert len(vov[4]) == 0
    assert vov == VectorOfVectors([[1, 2], [3, 4, 5], [2], [], []])

    vov = testvov.v3d

    vov.resize(3)
    assert ak.is_valid(vov.view_as("ak"))
    assert len(vov.cumulative_length) == 3
    assert len(vov.flattened_data) == vov.cumulative_length[-1]
    assert vov == VectorOfVectors(
        [[[1, 2], [3, 4, 5]], [[2], [4, 8, 9, 7]], [[5, 3, 1]]]
    )

    vov.resize(5)
    assert ak.is_valid(vov.view_as("ak"))
    assert len(vov) == 5
    assert vov == VectorOfVectors(
        [[[1, 2], [3, 4, 5]], [[2], [4, 8, 9, 7]], [[5, 3, 1]], [], []]
    )

    v = VectorOfVectors(dtype="i")
    v.resize(3)
    assert ak.is_valid(v.view_as("ak"))
    assert v == VectorOfVectors([[], [], []], dtype="i")


def test_aoesa(testvov):
    testvov = testvov.v2d

    arr = testvov.to_aoesa()
    desired = np.array(
        [
            [1, 2, np.nan, np.nan],
            [3, 4, 5, np.nan],
            [2, np.nan, np.nan, np.nan],
            [4, 8, 9, 7],
            [5, 3, 1, np.nan],
        ]
    )
    assert isinstance(arr, lgdo.ArrayOfEqualSizedArrays)
    assert np.issubdtype(arr.dtype, np.floating)
    assert np.array_equal(arr.nda, desired, True)

    v = VectorOfVectors(
        flattened_data=lgdo.Array(
            nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1], dtype="int16")
        ),
        cumulative_length=lgdo.Array(nda=np.array([2, 5, 6, 10, 13])),
    )
    aoesa = v.to_aoesa()

    assert np.issubdtype(aoesa.dtype, np.floating)

    aoesa = v.to_aoesa(fill_val=-999.9)
    assert np.issubdtype(aoesa.nda.dtype, np.floating)

    aoesa = v.to_aoesa(fill_val=-999)
    assert np.issubdtype(aoesa.nda.dtype, np.integer)

    aoesa = v.to_aoesa(fill_val=-999, preserve_dtype=True)
    assert aoesa.nda.dtype == np.int16


def test_set_vector(testvov):
    testvov = testvov.v2d

    testvov[0] = np.zeros(2)
    assert testvov == VectorOfVectors(
        [
            [0, 0],
            [3, 4, 5],
            [2],
            [4, 8, 9, 7],
            [5, 3, 1],
        ]
    )
    assert ak.is_valid(testvov.view_as("ak"))

    with pytest.raises(ValueError):
        testvov[0] = np.zeros(3)

    testvov[1] = np.zeros(3)

    assert testvov == VectorOfVectors(
        [
            [0, 0],
            [0, 0, 0],
            [2],
            [4, 8, 9, 7],
            [5, 3, 1],
        ]
    )
    assert ak.is_valid(testvov.view_as("ak"))


def test_append(testvov):
    testvov = testvov.v2d

    testvov.append(np.zeros(3))
    assert np.array_equal(testvov[-1], np.zeros(3))

    v = VectorOfVectors(dtype="int64")
    v.append(np.zeros(3))
    assert v == VectorOfVectors([[0, 0, 0]])
    assert ak.is_valid(v.view_as("ak"))


def test_insert(testvov):
    testvov = testvov.v2d

    testvov.insert(2, np.zeros(3))
    assert testvov == VectorOfVectors(
        [
            [1, 2],
            [3, 4, 5],
            [0, 0, 0],
            [2],
            [4, 8, 9, 7],
            [5, 3, 1],
        ]
    )
    assert ak.is_valid(testvov.view_as("ak"))

    v = VectorOfVectors(shape_guess=(3, 5), dtype="int32", fill_val=0)
    v.insert(2, [1, 2, 3])
    assert np.array_equal(v.cumulative_length, [5, 10, 13, 18])
    assert np.array_equal(v[2], [1, 2, 3])


def test_replace(testvov):
    testvov = testvov.v2d

    v = copy.deepcopy(testvov)
    v.replace(1, np.zeros(3))
    assert v == VectorOfVectors(
        [
            [1, 2],
            [0, 0, 0],
            [2],
            [4, 8, 9, 7],
            [5, 3, 1],
        ]
    )
    assert ak.is_valid(v.view_as("ak"))

    v = copy.deepcopy(testvov)
    v.replace(1, np.zeros(2))
    assert v == VectorOfVectors(
        [
            [1, 2],
            [0, 0],
            [2],
            [4, 8, 9, 7],
            [5, 3, 1],
        ]
    )
    assert ak.is_valid(v.view_as("ak"))

    v = copy.deepcopy(testvov)
    v.replace(1, np.zeros(4))
    assert v == VectorOfVectors(
        [
            [1, 2],
            [0, 0, 0, 0],
            [2],
            [4, 8, 9, 7],
            [5, 3, 1],
        ]
    )
    assert ak.is_valid(v.view_as("ak"))


def test_set_vector_unsafe(testvov):
    testvov = testvov.v2d

    desired = [
        np.array([1, 2], dtype=testvov.dtype),
        np.array([3, 4, 5], dtype=testvov.dtype),
        np.array([2], dtype=testvov.dtype),
        np.array([4, 8, 9, 7], dtype=testvov.dtype),
        np.array([5, 3, 1], dtype=testvov.dtype),
    ]
    desired_aoa = np.zeros(shape=(5, 5), dtype=testvov.dtype)
    desired_lens = np.array([len(arr) for arr in desired])

    # test sequential filling
    second_vov = lgdo.VectorOfVectors(shape_guess=(5, 5), dtype=testvov.dtype)
    for i, arr in enumerate(desired):
        second_vov._set_vector_unsafe(i, arr)
        desired_aoa[i, : len(arr)] = arr
    assert testvov == second_vov

    # test vectorized filling
    third_vov = lgdo.VectorOfVectors(shape_guess=(5, 5), dtype=testvov.dtype)
    third_vov._set_vector_unsafe(0, desired_aoa, desired_lens)
    assert testvov == third_vov


def test_iter(testvov):
    testvov = testvov.v2d

    desired = [[1, 2], [3, 4, 5], [2], [4, 8, 9, 7], [5, 3, 1]]

    for c, v in enumerate(testvov):
        assert np.array_equal(v, desired[c])


def test_view_as(testvov):
    v2d = testvov.v2d

    v2d.attrs["units"] = "s"
    with pytest.raises(ValueError):
        v2d.view_as("ak", with_units=True)

    ak_arr = v2d.view_as("ak", with_units=False)

    assert isinstance(ak_arr, ak.Array)
    assert ak.is_valid(ak_arr)
    assert len(ak_arr) == len(v2d)
    assert ak.all(ak_arr == [[1, 2], [3, 4, 5], [2], [4, 8, 9, 7], [5, 3, 1]])

    np_arr = v2d.view_as("np", with_units=True)
    assert isinstance(np_arr, pint.Quantity)
    assert np_arr.u == "second"
    assert isinstance(np_arr.m, np.ndarray)

    np_arr = v2d.view_as("np", with_units=False)
    assert isinstance(np_arr, np.ndarray)
    assert np.issubdtype(np_arr.dtype, np.floating)

    np_arr = v2d.view_as("np", with_units=False, fill_val=0, preserve_dtype=True)
    assert np.issubdtype(np_arr.dtype, np.integer)

    np_arr = v2d.view_as("pd", with_units=False)
    assert isinstance(np_arr, pd.Series)
    assert isinstance(np_arr.ak, akpd.accessor.AwkwardAccessor)

    v3d = testvov.v3d

    ak_arr = v3d.view_as("ak", with_units=False)

    assert isinstance(ak_arr, ak.Array)
    assert ak.is_valid(ak_arr)
    assert len(ak_arr) == len(v3d)
    assert ak.all(ak_arr == [[[1, 2], [3, 4, 5]], [[2], [4, 8, 9, 7]], [[5, 3, 1]]])


def test_lh5_iterator_view_as(lgnd_test_data):
    it = lh5.LH5Iterator(
        lgnd_test_data.get_path("lh5/l200-p03-r000-phy-20230312T055349Z-tier_psp.lh5"),
        "ch1067205/dsp/energies",
    )

    for obj, _, _ in it:
        assert ak.is_valid(obj.view_as("ak"))
