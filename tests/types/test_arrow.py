"""Tests for Arrow conversion integration."""

from __future__ import annotations

import warnings

import numpy as np
import pyarrow as pa
import pytest
from numpy.testing import assert_array_equal

from lgdo import Array, ArrayOfEqualSizedArrays, Table, VectorOfVectors
from lgdo.types.arrow import arrow_to_lgdo, lgdo_to_arrow
from lgdo.types.waveformtable import WaveformTable

# ============ Helpers ============


def _make_waveform_table():
    return WaveformTable(
        t0=Array(nda=np.array([0.0, 0.1]), attrs={"units": "us"}),
        dt=Array(nda=np.array([16.0, 16.0]), attrs={"units": "ns"}),
        values=ArrayOfEqualSizedArrays(
            nda=np.arange(10, dtype=np.float32).reshape(2, 5)
        ),
    )


# ============ lgdo_to_arrow output types ============


class TestLgdoToArrowTypes:
    def test_table_returns_pa_table(self):
        tbl = Table(col_dict={"x": Array(nda=np.array([1, 2]))})
        assert isinstance(lgdo_to_arrow(tbl), pa.Table)

    def test_waveformtable_returns_struct_array(self):
        result = lgdo_to_arrow(_make_waveform_table())
        assert isinstance(result, pa.StructArray)

    def test_array_returns_pa_array(self):
        result = lgdo_to_arrow(Array(nda=np.array([1.0, 2.0])))
        assert isinstance(result, pa.Array)
        assert not isinstance(result, pa.StructArray)

    def test_aoesa_returns_fixed_size_list(self):
        result = lgdo_to_arrow(ArrayOfEqualSizedArrays(nda=np.zeros((3, 4))))
        assert isinstance(result.type, pa.FixedSizeListType)

    def test_vov_returns_list_array(self):
        vov = VectorOfVectors(
            flattened_data=np.array([1, 2, 3]),
            offsets=np.array([0, 2, 3]),
        )
        assert isinstance(lgdo_to_arrow(vov), pa.ListArray)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported LGDO type"):
            lgdo_to_arrow("not an lgdo object")


# ============ arrow_to_lgdo output types ============


class TestArrowToLgdoTypes:
    def test_pa_table_returns_table(self):
        tbl = pa.table({"x": [1, 2, 3]})
        assert isinstance(arrow_to_lgdo(tbl), Table)

    def test_struct_with_waveform_fields_returns_waveformtable(self):
        struct = lgdo_to_arrow(_make_waveform_table())
        assert isinstance(arrow_to_lgdo(struct), WaveformTable)

    def test_struct_without_waveform_fields_returns_table(self):
        struct = pa.StructArray.from_arrays(
            [pa.array([1, 2]), pa.array([3, 4])],
            names=["a", "b"],
        )
        result = arrow_to_lgdo(struct)
        assert isinstance(result, Table)
        assert not isinstance(result, WaveformTable)

    def test_fixed_size_list_returns_aoesa(self):
        arr = pa.FixedSizeListArray.from_arrays(pa.array(np.arange(12)), 4)
        assert isinstance(arrow_to_lgdo(arr), ArrayOfEqualSizedArrays)

    def test_list_array_returns_vov(self):
        arr = pa.ListArray.from_arrays(
            pa.array([0, 2, 3, 5]),
            pa.array([1, 2, 3, 4, 5]),
        )
        assert isinstance(arrow_to_lgdo(arr), VectorOfVectors)

    def test_primitive_array_returns_array(self):
        assert isinstance(arrow_to_lgdo(pa.array([1, 2, 3])), Array)

    def test_chunked_array_warns(self):
        chunked = pa.chunked_array([pa.array([1, 2]), pa.array([3, 4])])
        with pytest.warns(UserWarning, match="2 chunks"):
            result = arrow_to_lgdo(chunked)
        assert isinstance(result, Array)
        assert_array_equal(result.nda, [1, 2, 3, 4])

    def test_single_chunk_no_warning(self):
        chunked = pa.chunked_array([pa.array([1, 2, 3])])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = arrow_to_lgdo(chunked)
        assert isinstance(result, Array)


# ============ Round-trip data integrity ============


class TestRoundTrip:
    def test_array(self):
        original = Array(nda=np.array([1.5, 2.5, 3.5]))
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert isinstance(back, Array)
        assert_array_equal(back.nda, original.nda)

    def test_array_int(self):
        original = Array(nda=np.array([10, 20, 30], dtype=np.int32))
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert_array_equal(back.nda, original.nda)
        assert back.nda.dtype == np.int32

    def test_aoesa_2d(self):
        original = ArrayOfEqualSizedArrays(nda=np.arange(12).reshape(3, 4))
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert isinstance(back, ArrayOfEqualSizedArrays)
        assert_array_equal(back.nda, original.nda)

    def test_aoesa_3d(self):
        original = ArrayOfEqualSizedArrays(nda=np.arange(24).reshape(2, 3, 4))
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert_array_equal(back.nda, original.nda)

    def test_vov(self):
        flat = np.array([10, 20, 30, 40, 50])
        offsets = np.array([0, 2, 3, 5])
        original = VectorOfVectors(flattened_data=flat, offsets=offsets)
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert isinstance(back, VectorOfVectors)
        assert_array_equal(back.flattened_data, flat)
        assert_array_equal(back._offsets.nda, offsets)

    def test_vov_with_aoesa_flattened_data(self):
        aoesa = ArrayOfEqualSizedArrays(nda=np.arange(20).reshape(5, 4))
        original = VectorOfVectors(flattened_data=aoesa, offsets=np.array([0, 2, 5]))
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert isinstance(back, VectorOfVectors)
        assert isinstance(back.flattened_data, ArrayOfEqualSizedArrays)
        assert_array_equal(back.flattened_data.nda, aoesa.nda)

    def test_nested_vov(self):
        inner = VectorOfVectors(
            flattened_data=np.array([1, 2, 3, 4, 5, 6]),
            offsets=np.array([0, 2, 3, 6]),
        )
        original = VectorOfVectors(flattened_data=inner, offsets=np.array([0, 1, 3]))
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert isinstance(back, VectorOfVectors)
        assert isinstance(back.flattened_data, VectorOfVectors)

    def test_table(self):
        original = Table(
            col_dict={
                "energy": Array(nda=np.array([1.0, 2.0, 3.0])),
                "channel": Array(nda=np.array([0, 1, 2])),
            }
        )
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert isinstance(back, Table)
        assert_array_equal(back["energy"].nda, original["energy"].nda)
        assert_array_equal(back["channel"].nda, original["channel"].nda)

    def test_waveform_table(self):
        original = _make_waveform_table()
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert isinstance(back, WaveformTable)
        assert_array_equal(back["t0"].nda, original["t0"].nda)
        assert_array_equal(back["dt"].nda, original["dt"].nda)
        assert_array_equal(back["values"].nda, original["values"].nda)

    def test_table_with_nested_waveform(self):
        original = Table(
            col_dict={
                "energy": Array(nda=np.array([1.0, 2.0])),
                "waveform": _make_waveform_table(),
            }
        )
        back = arrow_to_lgdo(lgdo_to_arrow(original))
        assert isinstance(back["waveform"], WaveformTable)
        assert_array_equal(
            back["waveform"]["values"].nda,
            original["waveform"]["values"].nda,
        )


# ============ Attrs round-trip ============


class TestAttrsRoundTrip:
    def test_string_attr(self):
        tbl = Table(col_dict={"x": Array(nda=np.array([1.0]), attrs={"units": "keV"})})
        back = arrow_to_lgdo(lgdo_to_arrow(tbl))
        assert back["x"].attrs["units"] == "keV"

    def test_numeric_attr(self):
        tbl = Table(
            col_dict={
                "x": Array(nda=np.array([1.0]), attrs={"version": 3, "scale": 1.5})
            }
        )
        back = arrow_to_lgdo(lgdo_to_arrow(tbl))
        assert back["x"].attrs["version"] == 3
        assert isinstance(back["x"].attrs["version"], int)
        assert back["x"].attrs["scale"] == 1.5

    def test_bool_attr(self):
        tbl = Table(
            col_dict={"x": Array(nda=np.array([1.0]), attrs={"calibrated": True})}
        )
        back = arrow_to_lgdo(lgdo_to_arrow(tbl))
        assert back["x"].attrs["calibrated"] is True

    def test_dict_attr(self):
        tbl = Table(
            col_dict={"x": Array(nda=np.array([1.0]), attrs={"info": {"a": 1, "b": 2}})}
        )
        back = arrow_to_lgdo(lgdo_to_arrow(tbl))
        assert back["x"].attrs["info"] == {"a": 1, "b": 2}

    def test_waveform_attrs(self):
        wft = _make_waveform_table()
        tbl = Table(col_dict={"wf": wft})
        back = arrow_to_lgdo(lgdo_to_arrow(tbl))
        assert back["wf"]["t0"].attrs["units"] == "us"
        assert back["wf"]["dt"].attrs["units"] == "ns"

    def test_table_level_attrs(self):
        tbl = Table(
            col_dict={"x": Array(nda=np.array([1.0]))},
            attrs={"description": "test table", "version": 42},
        )
        back = arrow_to_lgdo(lgdo_to_arrow(tbl))
        assert back.attrs["description"] == "test table"
        assert back.attrs["version"] == 42


# ============ view_as("arrow") ============


class TestViewAs:
    def test_array_view_as(self):
        arr = Array(nda=np.array([1, 2, 3]))
        result = arr.view_as("arrow")
        assert isinstance(result, pa.Array)

    def test_aoesa_view_as(self):
        aoesa = ArrayOfEqualSizedArrays(nda=np.arange(12).reshape(3, 4))
        result = aoesa.view_as("arrow")
        assert isinstance(result.type, pa.FixedSizeListType)

    def test_vov_view_as(self):
        vov = VectorOfVectors(
            flattened_data=np.array([1, 2, 3]),
            offsets=np.array([0, 2, 3]),
        )
        result = vov.view_as("arrow")
        assert isinstance(result, pa.ListArray)

    def test_table_view_as(self):
        tbl = Table(col_dict={"x": Array(nda=np.array([1, 2]))})
        result = tbl.view_as("arrow")
        assert isinstance(result, pa.Table)

    def test_waveformtable_view_as(self):
        result = _make_waveform_table().view_as("arrow")
        assert isinstance(result, pa.StructArray)


# ============ Constructor acceptance ============


class TestConstructorFromArrow:
    def test_array_from_pa_array(self):
        arr = Array(pa.array([1, 2, 3]))
        assert isinstance(arr, Array)
        assert_array_equal(arr.nda, [1, 2, 3])

    def test_array_from_pa_chunked_array(self):
        arr = Array(pa.chunked_array([pa.array([1, 2, 3])]))
        assert isinstance(arr, Array)
        assert_array_equal(arr.nda, [1, 2, 3])

    def test_table_from_pa_table(self):
        tbl = Table(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}))
        assert isinstance(tbl, Table)
        assert list(tbl.keys()) == ["a", "b"]
        assert_array_equal(tbl["a"].nda, [1, 2, 3])

    def test_vov_from_pa_list_array(self):
        pa_list = pa.ListArray.from_arrays([0, 2, 5], [10, 20, 30, 40, 50])
        vov = VectorOfVectors(data=pa_list)
        assert isinstance(vov, VectorOfVectors)
        assert_array_equal(vov.flattened_data.nda, [10, 20, 30, 40, 50])

    def test_aoesa_from_pa_fixed_size_list(self):
        pa_fsl = pa.FixedSizeListArray.from_arrays(pa.array(np.arange(12)), 4)
        aoesa = ArrayOfEqualSizedArrays(nda=pa_fsl)
        assert isinstance(aoesa, ArrayOfEqualSizedArrays)
        assert aoesa.nda.shape == (3, 4)
