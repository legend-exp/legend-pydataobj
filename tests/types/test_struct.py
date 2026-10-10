from __future__ import annotations

import pickle

import pytest

import lgdo


def test_datatype_name():
    struct = lgdo.Struct()
    assert struct.datatype_name() == "struct"


def test_form_datatype():
    struct = lgdo.Struct()
    assert struct.form_datatype() == "struct{}"


def test_update_datatype():
    struct = lgdo.Struct()
    struct.update_datatype()
    assert struct.attrs["datatype"] == "struct{}"


def test_init():
    obj_dict = {"scalar1": lgdo.Scalar(value=10)}
    attrs = {"attr1": 1}
    struct = lgdo.Struct(obj_dict=obj_dict, attrs=attrs)
    assert dict(struct) == obj_dict
    assert struct.attrs == attrs | {"datatype": "struct{scalar1}"}

    with pytest.raises(ValueError):
        lgdo.Struct(obj_dict={"scalar1": 1})

    with pytest.raises(ValueError):
        lgdo.Struct(obj_dict={"scalar1": lgdo.Scalar(value=10), "thing": int})


def test_init_nested():
    obj_dict = {
        "scalar1": lgdo.Scalar(10),
        "struct1": {"field1": lgdo.Scalar(11), "field2": lgdo.Array([1, 2, 3, 4])},
    }
    struct = lgdo.Struct(obj_dict)
    assert isinstance(struct.struct1, lgdo.Struct)
    assert isinstance(struct.struct1.field1, lgdo.Scalar)
    assert struct.struct1.field1.value == 11
    assert isinstance(struct.struct1.field2, lgdo.Array)
    assert struct.struct1.field2 == lgdo.Array([1, 2, 3, 4])


def test_add_field():
    struct = lgdo.Struct()
    struct.add_field("scalar1", lgdo.Scalar(value=10))

    assert struct.attrs["datatype"] == "struct{scalar1}"
    assert struct["scalar1"].__class__.__name__ == "Scalar"

    struct.add_field("array1", lgdo.Array(shape=(700, 21), dtype="f", fill_val=2))
    assert struct.attrs["datatype"] == "struct{array1,scalar1}"

    struct["array2"] = lgdo.Array(shape=(700, 21), dtype="f", fill_val=2)
    assert struct.attrs["datatype"] == "struct{array1,array2,scalar1}"


def test_getattr():
    struct = lgdo.Struct()
    struct["scalar1"] = lgdo.Scalar(value=10)
    assert struct.scalar1.value == 10


def test_getitem():
    struct = lgdo.Struct(attrs={"test": "Hello, world!"})
    struct.add_field("scalar1", lgdo.Scalar(value=10))
    struct.add_field("array1", lgdo.Array(shape=(10), fill_val=0))
    struct.add_field("scalar2", lgdo.Scalar(value=20))

    assert struct["scalar1"] == lgdo.Scalar(value=10)
    assert struct[["scalar1", "array1"]] == lgdo.Struct(
        {
            "scalar1": lgdo.Scalar(value=10),
            "array1": lgdo.Array(shape=(10), fill_val=0),
        },
        attrs={"test": "Hello, world!"},
    )


def test_remove_field():
    struct = lgdo.Struct()
    struct.add_field("scalar1", lgdo.Scalar(value=10))
    struct.add_field("array1", lgdo.Array(shape=(10), fill_val=0))

    struct.remove_field("scalar1")
    assert list(struct.keys()) == ["array1"]

    struct.remove_field("array1", delete=True)
    assert list(struct.keys()) == []


def test_nested_access():
    struct = lgdo.Struct()
    struct.add_field("struct1/scalar1", lgdo.Scalar(value=10))
    struct.add_field("struct1/struct2/scalar2", lgdo.Scalar(value=20))
    struct.add_field("struct1/scalar3", lgdo.Scalar(value=30))

    assert "struct1" in struct
    assert "struct1/struct2" in struct
    assert "struct1/struct2/scalar2" in struct
    assert "struct1/scalar2" not in struct

    assert set(struct.keys()) == {"struct1"}
    assert set(struct["struct1"].keys()) == {"scalar1", "struct2", "scalar3"}
    assert set(struct["struct1/struct2"].keys()) == {"scalar2"}
    assert struct["struct1/scalar1"] == lgdo.Scalar(value=10)
    assert struct["struct1/struct2/scalar2"] == lgdo.Scalar(value=20)
    assert struct["struct1/scalar3"] == lgdo.Scalar(value=30)

    struct.remove_field("struct1/scalar1")
    struct.remove_field("struct1/struct2")
    assert set(struct["struct1"].keys()) == {"scalar3"}


def test_update():
    st_final = lgdo.Struct(
        {
            "a": {
                "b": lgdo.Scalar(1),
                "c": lgdo.Scalar(3),
                "d": lgdo.Scalar(5),
            }
        }
    )

    st = lgdo.Struct({"a": {"b": lgdo.Scalar(1), "c": lgdo.Scalar(2)}})
    st.update(
        lgdo.Struct(
            {
                "a": {
                    "c": lgdo.Scalar(3),
                    "d": lgdo.Scalar(5),
                }
            }
        )
    )
    assert st == st_final

    st = lgdo.Struct({"a": {"b": lgdo.Scalar(1), "c": lgdo.Scalar(2)}})
    st.update(
        {
            "a/c": lgdo.Scalar(3),
            "a/d": lgdo.Scalar(5),
        }
    )
    assert st == st_final

    st = lgdo.Struct({"a": {"b": lgdo.Scalar(1), "c": lgdo.Scalar(2)}})
    st.update(
        a={
            "c": lgdo.Scalar(3),
            "d": lgdo.Scalar(5),
        }
    )
    assert st == st_final

    st = lgdo.Struct({"a": {"b": lgdo.Scalar(1), "c": lgdo.Scalar(2)}})
    st.update(
        [
            ("a/c", lgdo.Scalar(3)),
            ("a/d", lgdo.Scalar(5)),
        ]
    )
    assert st == st_final


def test_defer_update_datatype():
    struct = lgdo.Struct({"s1": lgdo.Scalar(value=1)})

    with struct.defer_update_datatype():
        struct.add_field("s2", lgdo.Scalar(value=2))
        struct["s3"] = lgdo.Scalar(value=3)
        assert struct.attrs["datatype"] == "struct{s1}"
        assert "s1" in struct
        assert "s2" in struct
        assert "s3" in struct

    assert not struct._defer_datatype
    assert struct.attrs["datatype"] == "struct{s1,s2,s3}"
    assert "s1" in struct
    assert "s2" in struct
    assert "s3" in struct

    with struct.defer_update_datatype():
        struct.update(
            {
                "s4": lgdo.Scalar(value=4),
                "s5": lgdo.Scalar(value=5),
            }
        )
        assert struct.attrs["datatype"] == "struct{s1,s2,s3}"
        assert "s1" in struct
        assert "s2" in struct
        assert "s3" in struct
        assert "s4" in struct
        assert "s5" in struct

    assert not struct._defer_datatype
    assert struct.attrs["datatype"] == "struct{s1,s2,s3,s4,s5}"
    assert "s1" in struct
    assert "s2" in struct
    assert "s3" in struct
    assert "s4" in struct
    assert "s5" in struct

    with struct.defer_update_datatype():
        struct.remove_field("s1")
        struct.remove_field("s2")
        struct.remove_field("s3")
        assert struct.attrs["datatype"] == "struct{s1,s2,s3,s4,s5}"
        assert "s1" not in struct
        assert "s2" not in struct
        assert "s3" not in struct
        assert "s4" in struct
        assert "s5" in struct

    assert not struct._defer_datatype
    assert struct.attrs["datatype"] == "struct{s4,s5}"
    assert "s1" not in struct
    assert "s2" not in struct
    assert "s3" not in struct
    assert "s4" in struct
    assert "s5" in struct

    with pytest.raises(ValueError), struct.defer_update_datatype():  # noqa: PT012
        struct.add_field("s1", lgdo.Scalar(value=1))
        struct.add_field("s2", 2)  # raises

    assert not struct._defer_datatype
    assert struct.attrs["datatype"] == "struct{s1,s4,s5}"
    assert "s1" in struct
    assert "s2" not in struct
    assert "s4" in struct
    assert "s5" in struct


def test_pickle():
    obj_dict = {"scalar1": lgdo.Scalar(value=10)}
    attrs = {"attr1": 1}
    struct = lgdo.Struct(obj_dict=obj_dict, attrs=attrs)

    ex = pickle.loads(pickle.dumps(struct))
    assert isinstance(ex, lgdo.Struct)
    assert ex.attrs["attr1"] == 1
    assert ex.attrs["datatype"] == struct.attrs["datatype"]
    assert ex["scalar1"].value == 10
    assert ex._defer_datatype == 0
