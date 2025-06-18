from __future__ import annotations

import logging
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import pytest

from lgdo import lh5, types


def test_write_compressed_lgnd_waveform_table(enc_lgnd_file):
    pass


def test_write_with_hdf5_compression_global(tmptestdir):
    data = types.Table(
        size=1000,
        col_dict={
            "col1": types.Array(np.arange(0, 100, 0.1)),
            "col2": types.Array(np.random.default_rng().random(1000)),
        },
    )
    outfile = f"{tmptestdir}/write_hdf5_data_global_var.lh5"

    lh5.settings.DEFAULT_HDF5_SETTINGS["shuffle"] = False
    lh5.settings.DEFAULT_HDF5_SETTINGS["compression"] = "lzf"

    lh5.write(data, "data", outfile, wo_mode="of")

    with h5py.File(outfile) as h5f:
        assert h5f["/data/col1"].shuffle is False
        assert h5f["/data/col1"].compression == "lzf"

    lh5.settings.DEFAULT_HDF5_SETTINGS = lh5.settings.default_hdf5_settings()
    assert lh5.settings.DEFAULT_HDF5_SETTINGS["shuffle"] is True
    assert lh5.settings.DEFAULT_HDF5_SETTINGS["compression"] == "gzip"


def test_write_with_hdf5_compression(lgnd_file, tmptestdir):
    store = lh5.LH5Store()
    wft = store.read("/geds/raw/waveform", lgnd_file)
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

    store.write(
        wft.values,
        "/geds/raw/waveform/values",
        f"{tmptestdir}/tmp-pygama-hdf5-compressed-wfs.lh5",
        wo_mode="overwrite_file",
        chunks=[1, 10],
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

    obj = store.read("/data/vov", f"{tmptestdir}/tmp-pygama-lgdo-empty-vov.lh5")
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

    if Path(f"{tmptestdir}/write_object_overwrite_test.lh5").exists():
        Path(f"{tmptestdir}/write_object_overwrite_test.lh5").unlink()

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
    tb_dat = store.read("my_group", f"{tmptestdir}/write_object_overwrite_test.lh5")
    assert np.array_equal(tb_dat["dset1"].nda, np.ones(10))


# Second: test that when we overwrite a table with fields with a different name, we delete the original field
def test_write_object_overwrite_table_with_deletion(caplog, tmptestdir):
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    if Path(f"{tmptestdir}/write_object_overwrite_test.lh5").exists():
        Path(f"{tmptestdir}/write_object_overwrite_test.lh5").unlink()

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
    tb_dat = store.read("my_group", f"{tmptestdir}/write_object_overwrite_test.lh5")
    assert np.array_equal(tb_dat["dset2"].nda, np.ones(10))

    # Also make sure that the first table's fields aren't lurking around the lh5 file!
    with h5py.File(f"{tmptestdir}/write_object_overwrite_test.lh5", "r") as lh5file:
        assert "dset1" not in list(lh5file["my_group"].keys())

    # Make sure the same behavior happens when we nest the table in a group
    if Path(f"{tmptestdir}/write_object_overwrite_test.lh5").exists():
        Path(f"{tmptestdir}/write_object_overwrite_test.lh5").unlink()

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
    tb_dat = store.read(
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
    if Path(f"{tmptestdir}/write_object_overwrite_test.lh5").exists():
        Path(f"{tmptestdir}/write_object_overwrite_test.lh5").unlink()

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
    tb_dat = store.read(
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
    array_dat = store.read("my_array", f"{tmptestdir}/write_object_overwrite_test.lh5")
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
    scalar_dat = store.read(
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

    vector_dat = store.read(
        "my_vector", f"{tmptestdir}/write_object_overwrite_test.lh5"
    )

    assert np.array_equal(vector_dat.cumulative_length.nda, [1, 2, 4, 7])
    assert np.array_equal(vector_dat.flattened_data.nda, [0, 1, 0, 0, 1, 1, 1])


def test_write_object_append_column(tmptestdir):
    # Try to append an array to a table
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
        wo_mode="of",
    )
    store.write(
        tb2,
        "my_table",
        f"{tmptestdir}/write_object_append_column_test.lh5",
        group="my_group",
        wo_mode="append_column",
    )

    # Now, check that the data were appended
    tb_dat = store.read(
        "my_group/my_table", f"{tmptestdir}/write_object_append_column_test.lh5"
    )
    assert isinstance(tb_dat, types.Table)
    assert np.array_equal(tb_dat["dset1"].nda, np.zeros(10))
    assert np.array_equal(tb_dat["dset2"].nda, np.ones(10))


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
    store.write(
        h2,
        "my_histogram",
        f"{tmptestdir}/write_histogram_test.lh5",
        wo_mode="overwrite",
        group="my_group",
    )

    # Now, check that the data were overwritten
    h3 = store.read("my_group/my_histogram", f"{tmptestdir}/write_histogram_test.lh5")
    assert isinstance(h3, types.Histogram)
    assert np.array_equal(h3.weights.nda, np.array([[10, 10], [10, 10]]))
    assert h3.binning[0].edges[0] == 2
    assert h3.binning[1].edges[-1] == 7
    assert h3.isdensity
    assert h3.binning[0].get_binedgeattrs() == {"units": "ns"}


def test_write_histogram_variable(caplog, tmptestdir):
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    # Start with an types.Histogram
    if Path(f"{tmptestdir}/write_histogram_test.lh5").exists():
        Path(f"{tmptestdir}/write_histogram_test.lh5").unlink()

    h1 = types.Histogram(
        np.array([[1, 1], [1, 1]]), (np.array([0, 1.2, 2]), np.array([2.1, 2.5, 2.3]))
    )
    h2 = types.Histogram(
        np.array([[10, 10], [10, 10]]),
        (np.array([2, 3.5, 4]), np.array([5, 6.5, 7])),
        isdensity=True,
        attrs={"testattr": "test"},
    )
    h2["weights"].attrs["weightattr"] = "testweight"
    h2.binning[0].attrs["binningattr"] = "testbinning"

    # Same field name, different values
    store = lh5.LH5Store()
    store.write(
        h1,
        "my_histogram",
        f"{tmptestdir}/write_histogram_test.lh5",
        group="my_group",
        wo_mode="write_safe",
    )
    store.write(
        h2,
        "my_histogram",
        f"{tmptestdir}/write_histogram_test.lh5",
        wo_mode="overwrite",
        group="my_group",
    )

    # Now, check that the data were overwritten
    h3 = store.read("my_group/my_histogram", f"{tmptestdir}/write_histogram_test.lh5")
    assert isinstance(h3, types.Histogram)
    assert np.array_equal(h3.weights.nda, np.array([[10, 10], [10, 10]]))
    assert np.array_equal(h3.binning[0].edges, np.array([2, 3.5, 4]))
    with pytest.raises(TypeError):
        x = h3.binning[0].first
    with pytest.raises(TypeError):
        x = h3.binning[1].last  # noqa: F841
    assert not h3.binning[0].is_range
    assert h3.isdensity

    # ensure that reading back attrs not only on binedges works.
    assert h3.attrs["testattr"] == "test"
    assert h3["weights"].attrs["weightattr"] == "testweight"
    assert h3.binning[0].attrs["binningattr"] == "testbinning"


def test_write_append_struct(tmptestdir):
    outfile = str(tmptestdir / "test-write-append-struct.lh5")
    st = types.Struct({"arr1": types.Table({"a": types.Array([1, 2, 3])})})
    lh5.write(st, "struct", outfile, wo_mode="of")
    st2 = types.Struct({"arr2": types.Table({"a": types.Array([1, 2, 3])})})
    lh5.write(st2, "struct", outfile, wo_mode="ac")

    result = lh5.read("struct", outfile)
    assert list(result.keys()) == ["arr1", "arr2"]
    assert len(result.arr1) == len(st.arr1)
    assert len(result.arr2) == len(st2.arr2)

    # append to empty struct
    outfile = str(tmptestdir / "test-write-append-struct.lh5")
    lh5.write(types.Struct({}), "struct", outfile, wo_mode="of")
    st2 = types.Struct({"arr2": types.Table({"a": types.Array([1, 2, 3])})})
    lh5.write(st2, "struct", outfile, wo_mode="ac")

    result = lh5.read("struct", outfile)
    assert list(result.keys()) == ["arr2"]
    assert len(result.arr2) == len(st2.arr2)


def test_write_structs_not_groups(tmptestdir):
    outfile = str(tmptestdir / "test-write-structs-not-groups2.lh5")

    scalar = types.Scalar("made with legend-pydataobj!")
    array = types.Array([1, 2, 3])
    array2 = types.Array([4, 5, 6])
    lh5.write(scalar, name="message", lh5_file=outfile, wo_mode="overwrite_file")
    lh5.write(array, name="numbers", group="closet", lh5_file=outfile)
    lh5.write(array2, name="numbers2", group="closet", lh5_file=outfile)
    result = lh5.read("/", outfile)
    assert isinstance(result, types.Struct)
    assert result.attrs["datatype"] == "struct{closet,message}"

    outfile = str(tmptestdir / "test-write-structs-not-groups.lh5")
    tb = types.Table({"a": types.Array([1, 2, 3])})
    lh5.write(tb, "test/table", outfile)
    print(lh5.show(outfile))
    tb2 = types.Table({"a": types.Array([4, 5, 6])})
    lh5.write(tb2, "test/table2", outfile)
    print(lh5.show(outfile))

    result = lh5.read("test", outfile)
    assert isinstance(result, types.Struct)
    assert result.attrs["datatype"] == "struct{table,table2}"
    assert result.table.a == tb.a
    assert result.table2.a == tb2.a
