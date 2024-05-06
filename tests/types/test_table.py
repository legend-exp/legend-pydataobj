from __future__ import annotations

import warnings

import awkward as ak
import numpy as np
import pandas as pd
import pytest

import lgdo
from lgdo import Table


def test_init():
    tbl = Table()
    assert not tbl.size
    assert tbl.loc == 0

    tbl = Table(size=10)
    assert tbl.size == 10

    col_dict = {
        "a": lgdo.Array(nda=np.array([1, 2, 3, 4])),
        "b": lgdo.Array(nda=np.array([5, 6, 7, 8])),
    }

    tbl = Table(col_dict=col_dict)
    assert tbl.size == 4

    tbl = Table(size=3, col_dict=col_dict)
    assert tbl.size == 3


def test_init_nested():
    col_dict = {
        "a": lgdo.Array(nda=np.array([1, 2, 3, 4])),
        "b": lgdo.Array(nda=np.array([5, 6, 7, 8])),
        "c": {
            "f1": lgdo.Array([1, 2, 3, 4]),
            "f2": lgdo.Array([1, 2, 3, 4]),
        },
    }

    tbl = Table(col_dict=col_dict)
    assert isinstance(tbl.c, Table)
    assert isinstance(tbl.c.f1, lgdo.Array)
    assert tbl.c.f1 == lgdo.Array([1, 2, 3, 4])


def test_pandas_df_init():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    tbl = Table(col_dict=df)
    assert sorted(tbl.keys()) == ["a", "b"]
    assert isinstance(tbl.a, lgdo.Array)
    assert isinstance(tbl.b, lgdo.Array)
    assert tbl.a == lgdo.Array([1, 2, 3, 4])
    assert tbl.b == lgdo.Array([5, 6, 7, 8])


def test_ak_array_init():
    array = ak.Array(
        {
            "a": [1, 2, 3, 4],
            "b": [[1, 2], [3], [4], [5, 6, 7]],
            "c": {"f1": [[], [5], [3, 7, 6], []], "f2": [5, 6, 7, 8]},
        }
    )
    tbl = Table(array)
    assert isinstance(tbl.a, lgdo.Array)
    assert isinstance(tbl.b, lgdo.VectorOfVectors)
    assert isinstance(tbl.c, Table)
    assert isinstance(tbl.c.f1, lgdo.VectorOfVectors)
    assert isinstance(tbl.c.f2, lgdo.Array)


def test_datatype_name():
    tbl = Table()
    assert tbl.datatype_name() == "table"


def test_push_row():
    tbl = Table()
    tbl.push_row()
    assert tbl.loc == 1


def test_is_full():
    tbl = Table(size=2)
    tbl.push_row()
    assert tbl.is_full() is False
    tbl.push_row()
    assert tbl.is_full() is True


def test_clear():
    tbl = Table()
    tbl.push_row()
    tbl.clear()
    assert tbl.loc == 0


def test_add_field():
    tbl = Table()
    tbl.add_field("a", lgdo.Array(np.array([1, 2, 3])), use_obj_size=True)
    assert tbl.size == 3

    with pytest.raises(TypeError):
        tbl.add_field("s", lgdo.Scalar(value=69))


def test_add_column():
    tbl = Table()
    tbl.add_column("a", lgdo.Array(np.array([1, 2, 3])), use_obj_size=True)
    assert tbl.size == 3
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tbl.add_column("b", lgdo.Array(np.array([1, 2, 3, 4])))
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)


def test_join():
    tbl1 = Table(size=3)
    tbl1.add_field("a", lgdo.FixedSizeArray(np.array([1, 2, 3])))
    tbl1.add_field("b", lgdo.Array(np.array([1, 2, 3])))
    assert list(tbl1.keys()) == ["a", "b"]

    tbl2 = Table(size=3)
    tbl2.add_field("c", lgdo.Array(np.array([4, 5, 6])))
    tbl2.add_field("d", lgdo.Array(np.array([9, 9, 10])))

    tbl1.join(tbl2)
    assert list(tbl1.keys()) == ["a", "b", "c", "d"]

    tbl2.join(tbl1, cols=("a"))
    assert list(tbl2.keys()) == ["c", "d", "a"]


def test_view_as():
    tbl = Table(size=3)
    tbl.add_column("a", lgdo.Array(np.array([1, 2, 3]), attrs={"units": "m"}))
    tbl.add_column("b", lgdo.Array(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])))
    tbl.add_column(
        "c",
        lgdo.VectorOfVectors(
            flattened_data=lgdo.Array(np.array([0, 1, 2, 3, 4, 5, 6])),
            cumulative_length=lgdo.Array(np.array([3, 4, 7])),
        ),
    )
    tbl.add_column(
        "d",
        lgdo.Table(
            col_dict={
                "a": lgdo.Array(np.array([2, 4, 6]), attrs={"units": "m"}),
                "b": lgdo.Array(np.array([[1, 1], [2, 4], [3, 9]])),
            }
        ),
    )

    df = tbl.view_as("pd", with_units=False)
    assert isinstance(df, pd.DataFrame)
    assert list(df.keys()) == ["a", "b", "c", "d_a", "d_b"]

    df = tbl.view_as("pd", with_units=True)
    assert isinstance(df, pd.DataFrame)
    assert list(df.keys()) == ["a", "b", "c", "d_a", "d_b"]
    assert df["a"].dtype == "meter"
    assert df["d_a"].dtype == "meter"

    ak_arr = tbl.view_as("ak", with_units=False)
    assert isinstance(ak_arr, ak.Array)
    assert list(ak_arr.fields) == ["a", "b", "c", "d"]

    with pytest.raises(ValueError):
        tbl.view_as("ak", with_units=True)

    with pytest.raises(TypeError):
        tbl.view_as("np")

    tbl.add_column(
        "e",
        lgdo.VectorOfVectors(
            flattened_data=lgdo.Array(np.array([0, 1, 2, 3, 4, 5, 6])),
            cumulative_length=lgdo.Array(np.array([3, 4, 7])),
            attrs={"units": "m"},
        ),
    )

    with pytest.raises(ValueError):
        tbl.view_as("pd", with_units=True)


def test_flatten():
    tbl = Table(
        col_dict={
            "a": lgdo.Array(np.array([1, 2, 3])),
            "tbl": Table(
                col_dict={
                    "b": lgdo.Array(np.array([4, 5, 6])),
                    "tbl1": Table(col_dict={"z": lgdo.Array(np.array([9, 9, 9]))}),
                }
            ),
        }
    )

    fl_tbl = tbl.flatten()
    assert sorted(fl_tbl.keys()) == ["a", "tbl__b", "tbl__tbl1__z"]


def test_remove_column():
    col_dict = {
        "a": lgdo.Array(nda=np.array([1, 2, 3, 4])),
        "b": lgdo.Array(nda=np.array([5, 6, 7, 8])),
        "c": lgdo.Array(nda=np.array([9, 10, 11, 12])),
    }

    tbl = Table(col_dict=col_dict)

    tbl.remove_column("a")
    assert list(tbl.keys()) == ["b", "c"]

    tbl.remove_column("c")
    assert list(tbl.keys()) == ["b"]
