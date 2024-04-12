from __future__ import annotations

import pytest

from lgdo import types
from lgdo.lh5 import datatype


def test_datatype2lgdo():
    d = datatype.datatype
    assert d("real") == types.Scalar
    assert d("bool") == types.Scalar
    assert d("complex") == types.Scalar

    assert d(" real ") == types.Scalar

    with pytest.raises(RuntimeError):
        assert d("int") == types.Scalar

    assert d("array<1>{real}") == types.Array
    assert d("fixedsize_array<1>{real}") == types.FixedSizeArray
    assert d("array_of_equalsized_arrays<1,1>{real}") == types.ArrayOfEqualSizedArrays

    assert d("array<1>{array<1>{real}}") == types.VectorOfVectors
    assert d("array<1>{array<1>{array<1>{real}}}") == types.VectorOfVectors

    assert d("array<1>{encoded_array<1>{real}}") == types.VectorOfEncodedVectors
    assert (
        d("array_of_encoded_equalsized_arrays<1,1>{real}")
        == types.ArrayOfEncodedEqualSizedArrays
    )

    assert d("struct{a,b,c,d}") == types.Struct
    assert d("struct{}") == types.Struct
    assert d("table{a,b,c,d}") == types.Table


def test_utils():
    assert (
        datatype.get_nested_datatype_string("array<1>{encoded_array<1>{real}}")
        == "encoded_array<1>{real}"
    )
    assert datatype.get_nested_datatype_string("table{a,b,c,d}") == "a,b,c,d"
    assert datatype.get_nested_datatype_string("table{}") == ""
    assert (
        datatype.get_nested_datatype_string("array_of_equalsized_arrays<1,1>{real}")
        == "real"
    )

    assert datatype.get_struct_fields("table{a,b,c,d}") == ["a", "b", "c", "d"]
    assert datatype.get_struct_fields("table{}") == []
