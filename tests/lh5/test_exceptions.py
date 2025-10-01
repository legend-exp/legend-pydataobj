from __future__ import annotations

import logging
import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest

from lgdo import lh5, types
from lgdo.lh5.exceptions import LH5DecodeError, LH5EncodeError


def test_tools(tmptestdir):
    thefile = tmptestdir / "already-open-file-tools.lh5"
    lh5.write(types.Struct({}), "/empty", thefile)

    for mode in ("r", "r+", "w", "a"):
        f = h5py.File(thefile, mode)
        with pytest.raises(LH5DecodeError):
            lh5.ls(thefile)
        with pytest.raises(LH5DecodeError):
            lh5.show(thefile)
        f.close()


def test_interact_with_already_open_files(tmptestdir):
    thefile = tmptestdir / "already-open-file.lh5"
    lh5.write(types.Struct({}), "/empty", thefile)

    assert thefile.exists()

    f = h5py.File(thefile, "r")
    with pytest.raises(LH5EncodeError):
        lh5.write(types.Struct({}), "/empty2", thefile)
    f.close()

    for mode in ("r", "r+", "w", "a"):
        f = h5py.File(thefile, mode)
        with pytest.raises(LH5DecodeError):
            lh5.read("/empty", thefile)
        f.close()


def test_write_safe(tmptestdir):
    # write_safe should create new file
    struct = types.Struct()
    struct.add_field("scalar", types.Scalar(value=10, attrs={"sth": 1}))
    lh5.write(
        struct,
        "struct",
        f"{tmptestdir}/tmp-pygama-write_safe_exception.lh5",
        group="/data",
        start_row=1,
        n_rows=3,
        wo_mode="w",
    )

    # write_safe should not allow writing to existing dataset
    with pytest.raises(lh5.exceptions.LH5EncodeError):
        lh5.write(
            struct,
            "struct",
            f"{tmptestdir}/tmp-pygama-write_safe_exception.lh5",
            group="/data",
            start_row=1,
            n_rows=3,
            wo_mode="w",
        )


def test_open_non_existing_file(tmptestdir):
    with pytest.raises(LH5DecodeError):
        lh5.read("boh", tmptestdir / "beh.lh5")


def test_open_non_existing_dataset(tmptestdir):
    thefile = tmptestdir / "dummy-file.lh5"
    lh5.write(types.Struct({}), "/empty", thefile)

    with pytest.raises(LH5DecodeError):
        lh5.read("/boh", thefile)


def test_write_cleanup_on_error(tmptestdir):
    outfile = tmptestdir / "cleanup_error.lh5"
    h = types.Histogram(np.array([[1]]), (np.array([0, 1]), np.array([0, 1])))

    lh5.write(h, "hist", outfile, wo_mode="overwrite_file")
    with pytest.raises(lh5.exceptions.LH5EncodeError):
        lh5.write(h, "hist", outfile, wo_mode="append")

    # file should be closed so we can open it again
    assert lh5.read("hist", outfile) is not None


def test_write_append_struct(tmptestdir):
    outfile = tmptestdir / "test-write-append-struct.lh5"
    st = types.Struct({"arr1": types.Table({"a": types.Array([1, 2, 3])})})
    lh5.write(st, "struct", outfile, wo_mode="of")
    st2 = types.Struct({"arr2": types.Table({"a": types.Array([1, 2, 3])})})
    lh5.write(st2, "struct", outfile, wo_mode="ac")

    # test error when appending existing field
    with pytest.raises(lh5.exceptions.LH5EncodeError):
        lh5.write(
            types.Struct({"arr2": types.Array([4, 5, 6])}),
            "struct",
            outfile,
            wo_mode="ac",
        )

    # error if appending to object of different type
    with pytest.raises(lh5.exceptions.LH5EncodeError):
        lh5.write(
            types.Struct({"arr2": types.Array([4, 5, 6])}),
            "struct",
            outfile,
            wo_mode="ac",
        )

    lh5.write(
        types.Table({"arr1": types.Array([1, 2, 3])}), "struct", outfile, wo_mode="of"
    )

    # error if appending to object of different type
    with pytest.raises(lh5.exceptions.LH5EncodeError):
        lh5.write(
            types.Table({"arr2": types.Array([4, 5, 6, 7])}),
            "struct",
            outfile,
            wo_mode="ac",
        )


# Test that when we try to overwrite an existing column in a table we fail
def test_write_object_append_column(tmptestdir):
    # Try to append an array to a table
    if Path(f"{tmptestdir}/write_object_append_column_test.lh5").exists():
        Path(f"{tmptestdir}/write_object_append_column_test.lh5").unlink()

    array1 = types.Array(np.zeros(10))
    tb1 = types.Table(col_dict={"dset1`": types.Array(np.ones(10))})
    store = lh5.LH5Store()
    store.write(array1, "my_table", f"{tmptestdir}/write_object_append_column_test.lh5")
    with pytest.raises(lh5.exceptions.LH5EncodeError):
        store.write(
            tb1,
            "my_table",
            f"{tmptestdir}/write_object_append_column_test.lh5",
            wo_mode="append_column",
        )  # Now, try to append a column to an array

    # Try to append a table that has a same key as the old table
    if Path(f"{tmptestdir}/write_object_append_column_test.lh5").exists():
        Path(f"{tmptestdir}/write_object_append_column_test.lh5").unlink()

    tb1 = types.Table(
        col_dict={
            "dset1": types.Array(np.zeros(10)),
            "dset2": types.Array(np.zeros(10)),
        }
    )
    tb2 = types.Table(
        col_dict={"dset2": types.Array(np.ones(10))}
    )  # Same field name, different values
    store = lh5.LH5Store()
    store.write(tb1, "my_table", f"{tmptestdir}/write_object_append_column_test.lh5")
    with pytest.raises(lh5.exceptions.LH5EncodeError):
        store.write(
            tb2,
            "my_table",
            f"{tmptestdir}/write_object_append_column_test.lh5",
            wo_mode="append_column",
        )  # Now, try to append a column with a same field

    # try appending a column that is larger than one that exists
    if Path(f"{tmptestdir}/write_object_append_column_test.lh5").exists():
        Path(f"{tmptestdir}/write_object_append_column_test.lh5").unlink()

    tb1 = types.Table(col_dict={"dset1": types.Array(np.zeros(10))})
    tb2 = types.Table(
        col_dict={"dset2": types.Array(np.ones(20))}
    )  # different field name, different size
    store = lh5.LH5Store()
    store.write(tb1, "my_table", f"{tmptestdir}/write_object_append_column_test.lh5")
    with pytest.raises(lh5.exceptions.LH5EncodeError):
        store.write(
            tb2,
            "my_table",
            f"{tmptestdir}/write_object_append_column_test.lh5",
            wo_mode="append_column",
        )  # Now, try to append a column with a different field size


# Test writing and reading histograms.
def test_write_histogram(caplog, tmptestdir):
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    # Start with an types.Histogram
    if Path(f"{tmptestdir}/write_histogram_test.lh5").exists():
        Path(f"{tmptestdir}/write_histogram_test.lh5").unlink()

    h1 = types.Histogram(
        np.array([[1, 1], [1, 1]]), (np.array([0, 1, 2]), np.array([2.1, 2.2, 2.3]))
    )
    h2 = types.Histogram(
        np.array([[10, 10], [10, 10]]),
        (np.array([2, 3, 4]), np.array([5, 6, 7])),
        isdensity=True,
    )
    h2.binning[0]["binedges"].attrs["units"] = "ns"

    # Same field name, different values
    store = lh5.LH5Store()
    # "appending" to a non-existing histogram should work.
    store.write(
        h1,
        "my_histogram",
        f"{tmptestdir}/write_histogram_test.lh5",
        group="my_group",
        wo_mode="append",
    )
    with pytest.raises(lh5.exceptions.LH5EncodeError):
        # appending to an existing histogram should not work.
        store.write(
            h1,
            "my_histogram",
            f"{tmptestdir}/write_histogram_test.lh5",
            group="my_group",
            wo_mode="append",
        )
    store.write(
        h2,
        "my_histogram",
        f"{tmptestdir}/write_histogram_test.lh5",
        wo_mode="overwrite",
        group="my_group",
    )

    # Now, check that writing with other modes throws.
    for disallowed_wo_mode in ["append", "append_column"]:
        with pytest.raises(lh5.exceptions.LH5EncodeError):
            store.write(
                h2,
                "my_histogram",
                f"{tmptestdir}/write_histogram_test.lh5",
                wo_mode=disallowed_wo_mode,
                group="my_group",
            )
    with pytest.raises(lh5.exceptions.LH5EncodeError):
        store.write(
            h2,
            "my_histogram",
            f"{tmptestdir}/write_histogram_test.lh5",
            wo_mode="overwrite",
            write_start=1,
            group="my_group",
        )


def test_pickle():
    # test (un-)pickling of LH5 exceptions; e.g. for multiprocessing use.

    ex = LH5EncodeError("message", "file", "group", "name")
    ex = pickle.loads(pickle.dumps(ex))
    assert isinstance(ex, LH5EncodeError)

    ex = LH5DecodeError("message", "file", "obj")
    ex = pickle.loads(pickle.dumps(ex))
    assert isinstance(ex, LH5DecodeError)
