from __future__ import annotations

import numpy as np

import lgdo
from lgdo import lh5


def test_read(lh5_file):
    lh5_obj = lh5.read("/data/struct/scalar", lh5_file)
    assert isinstance(lh5_obj, lgdo.Scalar)
    assert lh5_obj.value == 10
    assert lh5_obj.attrs["sth"] == 1


def test_write(tmptestdir):
    struct = lgdo.Struct()
    struct.add_field("scalar", lgdo.Scalar(value=10, attrs={"sth": 1}))
    lh5.write(
        struct,
        "struct",
        f"{tmptestdir}/tmp-pygama-lgdo-types2.lh5",
        group="/data",
        start_row=1,
        n_rows=3,
        wo_mode="overwrite_file",
    )
    assert lh5.ls(f"{tmptestdir}/tmp-pygama-lgdo-types2.lh5")


def test_read_as(lh5_file):
    store = lh5.LH5Store()
    obj1 = store.read("/data/struct/table", lh5_file, start_row=1)
    obj1 = obj1.view_as("pd", with_units=True)

    obj2 = lh5.read_as(
        "/data/struct/table", lh5_file, "pd", start_row=1, with_units=True
    )
    assert obj1.equals(obj2)

    obj2 = lh5.read_as("/data/struct/table", [lh5_file], "ak")


def test_read_multiple_files(lh5_file):
    lh5_obj = lh5.read("/data/struct/array", [lh5_file, lh5_file, lh5_file])
    assert isinstance(lh5_obj, lgdo.Array)
    assert len(lh5_obj) == 9
    assert (lh5_obj.nda == np.array([2, 3, 4] * 3)).all()

    lh5_obj = lh5.read(
        "/data/struct/array", [lh5_file, lh5_file, lh5_file], idx=[1, 3, 5, 7]
    )
    assert len(lh5_obj) == 4
    assert (lh5_obj.nda == np.array([3, 2, 4, 3])).all()
