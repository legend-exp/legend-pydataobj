from __future__ import annotations

import logging
import os

import awkward as ak
import h5py
import numpy as np
import pytest

from lgdo import lh5, types


def test_write_compressed_lgnd_waveform_table(enc_lgnd_file):  # noqa: ARG001
    pass


def test_write_with_hdf5_compression(lgnd_file, tmptestdir):
    store = lh5.LH5Store()
    wft, n_rows = store.read("/geds/raw/waveform", lgnd_file)
    store.write(
        wft,
        "/geds/raw/waveform",
        f"{tmptestdir}/tmp-pygama-hdf5-compressed-wfs.lh5",
        wo_mode="overwrite_file",
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    with h5py.File(f"{tmptestdir}/tmp-pygama-hdf5-compressed-wfs.lh5") as h5f:
        assert h5f["/geds/raw/waveform/values"].compression == "gzip"
        assert h5f["/geds/raw/waveform/values"].compression_opts == 9
        assert h5f["/geds/raw/waveform/values"].shuffle is True

    store.write(
        wft,
        "/geds/raw/waveform",
        f"{tmptestdir}/tmp-pygama-hdf5-compressed-wfs.lh5",
        wo_mode="overwrite_file",
        compression=None,
        shuffle=False,
    )
    with h5py.File(f"{tmptestdir}/tmp-pygama-hdf5-compressed-wfs.lh5") as h5f:
        assert h5f["/geds/raw/waveform/values"].compression is None
        assert h5f["/geds/raw/waveform/values"].shuffle is False


def test_write_empty_vov(tmptestdir):
    vov = types.VectorOfVectors(flattened_data=[], cumulative_length=[])
    store = lh5.LH5Store()
    store.write(
        vov,
        "vov",
        f"{tmptestdir}/tmp-pygama-lgdo-empty-vov.lh5",
        group="/data",
    )

    obj, _ = store.read("/data/vov", f"{tmptestdir}/tmp-pygama-lgdo-empty-vov.lh5")
    assert obj == vov


def test_write_append_array(tmptestdir):
    arr = types.Array([1, 2, 3, 4])
    arr_bis = types.Array([11, 34, 55, 57, 16])

    outfile = f"{tmptestdir}/write_append_array.lh5"
    lh5.write(arr, "arr", outfile, wo_mode="of")
    lh5.write(arr_bis, "arr", outfile, wo_mode="append")

    v = lh5.read("arr", outfile)
    assert v == types.Array([1, 2, 3, 4, 11, 34, 55, 57, 16])


def test_write_append_vov(tmptestdir):
    vov = types.VectorOfVectors([[1, 2, 3], [4], [5, 6], [7, 9, 8]])
    vov_bis = types.VectorOfVectors([[11], [34, 55], [57, 16], [28]])

    outfile = f"{tmptestdir}/write_append_array.lh5"
    lh5.write(vov, "vov", outfile, wo_mode="of")
    lh5.write(vov_bis, "vov", outfile, wo_mode="append")

    v = lh5.read("vov", outfile)
    assert ak.is_valid(v.view_as("ak"))
    assert v == types.VectorOfVectors(
        [[1, 2, 3], [4], [5, 6], [7, 9, 8], [11], [34, 55], [57, 16], [28]]
    )

    vov = types.VectorOfVectors([[[1, 2, 3], [4]], [[5, 6], [7, 9, 8]]])
    vov_bis = types.VectorOfVectors([[[11], [34, 55]], [[57, 16], [28]]])

    outfile = f"{tmptestdir}/write_append_array.lh5"
    lh5.write(vov, "vov", outfile, wo_mode="of")
    lh5.write(vov_bis, "vov", outfile, wo_mode="append")

    v = lh5.read("vov", outfile)
    assert ak.is_valid(v.view_as("ak"))


# First test that we can overwrite a table with the same name without deleting the original field
def test_write_object_overwrite_table_no_deletion(caplog, tmptestdir):
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    if os.path.exists(f"{tmptestdir}/write_object_overwrite_test.lh5"):
        os.remove(f"{tmptestdir}/write_object_overwrite_test.lh5")

    tb1 = types.Table(col_dict={"dset1": types.Array(np.zeros(10))})
    tb2 = types.Table(
        col_dict={"dset1": types.Array(np.ones(10))}
    )  # Same field name, different values
    store = lh5.LH5Store()
    store.write(tb1, "my_group", f"{tmptestdir}/write_object_overwrite_test.lh5")
    store.write(
        tb2,
        "my_group",
        f"{tmptestdir}/write_object_overwrite_test.lh5",
        wo_mode="overwrite",
    )  # Now, try to overwrite the same field

    # If the old field is deleted from the file before writing the new field, then we would get an extra debug statement
    assert "dset1 is not present in new table, deleting field" not in [
        rec.message for rec in caplog.records
    ]

    # Now, check that the data were overwritten
    tb_dat, _ = store.read("my_group", f"{tmptestdir}/write_object_overwrite_test.lh5")
    assert np.array_equal(tb_dat["dset1"].nda, np.ones(10))


# Second: test that when we overwrite a table with fields with a different name, we delete the original field
def test_write_object_overwrite_table_with_deletion(caplog, tmptestdir):
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    if os.path.exists(f"{tmptestdir}/write_object_overwrite_test.lh5"):
        os.remove(f"{tmptestdir}/write_object_overwrite_test.lh5")

    tb1 = types.Table(col_dict={"dset1": types.Array(np.zeros(10))})
    tb2 = types.Table(
        col_dict={"dset2": types.Array(np.ones(10))}
    )  # Same field name, different values
    store = lh5.LH5Store()
    store.write(tb1, "my_group", f"{tmptestdir}/write_object_overwrite_test.lh5")
    store.write(
        tb2,
        "my_group",
        f"{tmptestdir}/write_object_overwrite_test.lh5",
        wo_mode="overwrite",
    )  # Now, try to overwrite with a different field

    # Now, check that the data were overwritten
    tb_dat, _ = store.read("my_group", f"{tmptestdir}/write_object_overwrite_test.lh5")
    assert np.array_equal(tb_dat["dset2"].nda, np.ones(10))

    # Also make sure that the first table's fields aren't lurking around the lh5 file!
    with h5py.File(f"{tmptestdir}/write_object_overwrite_test.lh5", "r") as lh5file:
        assert "dset1" not in list(lh5file["my_group"].keys())

    # Make sure the same behavior happens when we nest the table in a group
    if os.path.exists(f"{tmptestdir}/write_object_overwrite_test.lh5"):
        os.remove(f"{tmptestdir}/write_object_overwrite_test.lh5")

    tb1 = types.Table(col_dict={"dset1": types.Array(np.zeros(10))})
    tb2 = types.Table(
        col_dict={"dset2": types.Array(np.ones(10))}
    )  # Same field name, different values
    store = lh5.LH5Store()
    store.write(
        tb1,
        "my_table",
        f"{tmptestdir}/write_object_overwrite_test.lh5",
        group="my_group",
    )
    store.write(
        tb2,
        "my_table",
        f"{tmptestdir}/write_object_overwrite_test.lh5",
        group="my_group",
        wo_mode="overwrite",
    )  # Now, try to overwrite with a different field

    # Now, check that the data were overwritten
    tb_dat, _ = store.read(
        "my_group/my_table", f"{tmptestdir}/write_object_overwrite_test.lh5"
    )
    assert np.array_equal(tb_dat["dset2"].nda, np.ones(10))

    # Also make sure that the first table's fields aren't lurking around the lh5 file!
    with h5py.File(f"{tmptestdir}/write_object_overwrite_test.lh5", "r") as lh5file:
        assert "dset1" not in list(lh5file["my_group/my_table"].keys())


# Third: test that when we overwrite other LGDO classes
def test_write_object_overwrite_lgdo(caplog, tmptestdir):
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    # Start with an types.WaveformTable
    if os.path.exists(f"{tmptestdir}/write_object_overwrite_test.lh5"):
        os.remove(f"{tmptestdir}/write_object_overwrite_test.lh5")

    tb1 = types.WaveformTable(
        t0=np.zeros(10),
        t0_units="ns",
        dt=np.zeros(10),
        dt_units="ns",
        values=np.zeros((10, 10)),
        values_units="ADC",
    )
    tb2 = types.WaveformTable(
        t0=np.ones(10),
        t0_units="ns",
        dt=np.ones(10),
        dt_units="ns",
        values=np.ones((10, 10)),
        values_units="ADC",
    )  # Same field name, different values
    store = lh5.LH5Store()
    store.write(
        tb1,
        "my_table",
        f"{tmptestdir}/write_object_overwrite_test.lh5",
        group="my_group",
    )
    store.write(
        tb2,
        "my_table",
        f"{tmptestdir}/write_object_overwrite_test.lh5",
        wo_mode="overwrite",
        group="my_group",
    )

    # If the old field is deleted from the file before writing the new field, then we would get a debug statement
    assert "my_table is not present in new table, deleting field" not in [
        rec.message for rec in caplog.records
    ]

    # Now, check that the data were overwritten
    tb_dat, _ = store.read(
        "my_group/my_table", f"{tmptestdir}/write_object_overwrite_test.lh5"
    )
    assert np.array_equal(tb_dat["values"].nda, np.ones((10, 10)))

    # Now try overwriting an array, and test the write_start argument
    array1 = types.Array(nda=np.zeros(10))
    array2 = types.Array(nda=np.ones(20))
    store = lh5.LH5Store()
    store.write(array1, "my_array", f"{tmptestdir}/write_object_overwrite_test.lh5")
    store.write(
        array2,
        "my_array",
        f"{tmptestdir}/write_object_overwrite_test.lh5",
        wo_mode="overwrite",
        write_start=5,
    )

    # Now, check that the data were overwritten
    array_dat, _ = store.read(
        "my_array", f"{tmptestdir}/write_object_overwrite_test.lh5"
    )
    expected_out_array = np.append(np.zeros(5), np.ones(20))

    assert np.array_equal(array_dat.nda, expected_out_array)

    # Now try overwriting a scalar
    scalar1 = types.Scalar(0)
    scalar2 = types.Scalar(1)
    store = lh5.LH5Store()
    store.write(scalar1, "my_scalar", f"{tmptestdir}/write_object_overwrite_test.lh5")
    store.write(
        scalar2,
        "my_scalar",
        f"{tmptestdir}/write_object_overwrite_test.lh5",
        wo_mode="overwrite",
    )

    # Now, check that the data were overwritten
    scalar_dat, _ = store.read(
        "my_scalar", f"{tmptestdir}/write_object_overwrite_test.lh5"
    )

    assert scalar_dat.value == 1

    # Finally, try overwriting a vector of vectors
    vov1 = types.VectorOfVectors([np.zeros(1), np.ones(2), np.zeros(3)])
    vov2 = types.VectorOfVectors([np.ones(1), np.zeros(2), np.ones(3)])
    store = lh5.LH5Store()
    store.write(vov1, "my_vector", f"{tmptestdir}/write_object_overwrite_test.lh5")
    store.write(
        vov2,
        "my_vector",
        f"{tmptestdir}/write_object_overwrite_test.lh5",
        wo_mode="overwrite",
        write_start=1,
    )  # start overwriting the second list of lists

    vector_dat, _ = store.read(
        "my_vector", f"{tmptestdir}/write_object_overwrite_test.lh5"
    )

    assert np.array_equal(vector_dat.cumulative_length.nda, [1, 2, 4, 7])
    assert np.array_equal(vector_dat.flattened_data.nda, [0, 1, 0, 0, 1, 1, 1])


# Test that when we try to overwrite an existing column in a table we fail
def test_write_object_append_column(tmptestdir):
    # Try to append an array to a table
    if os.path.exists(f"{tmptestdir}/write_object_append_column_test.lh5"):
        os.remove(f"{tmptestdir}/write_object_append_column_test.lh5")

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
    if os.path.exists(f"{tmptestdir}/write_object_append_column_test.lh5"):
        os.remove(f"{tmptestdir}/write_object_append_column_test.lh5")

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
    if os.path.exists(f"{tmptestdir}/write_object_append_column_test.lh5"):
        os.remove(f"{tmptestdir}/write_object_append_column_test.lh5")

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

    # Finally successfully append a column
    if os.path.exists(f"{tmptestdir}/write_object_append_column_test.lh5"):
        os.remove(f"{tmptestdir}/write_object_append_column_test.lh5")

    tb1 = types.Table(col_dict={"dset1": types.Array(np.zeros(10))})
    tb2 = types.Table(
        col_dict={"dset2": types.Array(np.ones(10))}
    )  # different field name, different size
    store = lh5.LH5Store()
    store.write(
        tb1,
        "my_table",
        f"{tmptestdir}/write_object_append_column_test.lh5",
        group="my_group",
    )
    store.write(
        tb2,
        "my_table",
        f"{tmptestdir}/write_object_append_column_test.lh5",
        group="my_group",
        wo_mode="append_column",
    )

    # Now, check that the data were appended
    tb_dat, _ = store.read(
        "my_group/my_table", f"{tmptestdir}/write_object_append_column_test.lh5"
    )
    assert isinstance(tb_dat, types.Table)
    assert np.array_equal(tb_dat["dset1"].nda, np.zeros(10))
    assert np.array_equal(tb_dat["dset2"].nda, np.ones(10))
