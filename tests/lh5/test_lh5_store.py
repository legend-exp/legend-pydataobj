# ruff: noqa: ARG001

from __future__ import annotations

import awkward as ak
import h5py
import numpy as np
import pytest

import lgdo
from lgdo import lh5, types
from lgdo.lh5 import DEFAULT_HDF5_SETTINGS


def test_init():
    lh5.LH5Store()


def test_gimme_file(lgnd_file):
    store = lh5.LH5Store(keep_open=True)

    f = store.gimme_file(lgnd_file)
    assert isinstance(f, h5py.File)
    assert store.files[lgnd_file] == f

    with pytest.raises(FileNotFoundError):
        store.gimme_file("non-existent-file")


def test_gimme_group(lgnd_file, tmptestdir):
    f = h5py.File(lgnd_file)
    store = lh5.LH5Store()
    g = store.gimme_group("/geds", f)
    assert isinstance(g, h5py.Group)

    f = h5py.File(f"{tmptestdir}/testfile.lh5", mode="w")
    g = store.gimme_group("/geds", f, grp_attrs={"attr1": 1}, overwrite=True)
    assert isinstance(g, h5py.Group)


def test_write_objects(lh5_file):
    pass


def test_read_n_rows(lh5_file):
    store = lh5.LH5Store()
    assert store.read_n_rows("/data/struct_full/aoesa", lh5_file) == 5
    assert store.read_n_rows("/data/struct_full/array", lh5_file) == 5
    assert store.read_n_rows("/data/struct_full/scalar", lh5_file) is None
    assert store.read_n_rows("/data/struct_full/table", lh5_file) == 4
    assert store.read_n_rows("/data/struct_full/voev", lh5_file) == 5
    assert store.read_n_rows("/data/struct_full/vov", lh5_file) == 5
    assert store.read_n_rows("/data/struct_full/vov3d", lh5_file) == 5
    assert store.read_n_rows("/data/struct_full/wftable", lh5_file) == 10
    assert store.read_n_rows("/data/struct_full/wftable_enc/values", lh5_file) == 10


def test_get_buffer(lh5_file):
    store = lh5.LH5Store()
    buf = store.get_buffer("/data/struct_full/wftable_enc", lh5_file)
    assert isinstance(buf.values, types.ArrayOfEqualSizedArrays)


def test_read_scalar(lh5_file):
    store = lh5.LH5Store()
    lh5_obj, n_rows = store.read("/data/struct/scalar", lh5_file)
    assert isinstance(lh5_obj, lgdo.Scalar)
    assert lh5_obj.value == 10
    assert n_rows == 1
    assert lh5_obj.attrs["sth"] == 1
    with h5py.File(lh5_file) as h5f:
        assert h5f["/data/struct/scalar"].compression is None


def test_read_array(lh5_file):
    store = lh5.LH5Store()
    lh5_obj, n_rows = store.read("/data/struct/array", lh5_file)
    assert isinstance(lh5_obj, types.Array)
    assert (lh5_obj.nda == np.array([2, 3, 4])).all()
    assert n_rows == 3
    with h5py.File(lh5_file) as h5f:
        assert (
            h5f["/data/struct/array"].compression
            is DEFAULT_HDF5_SETTINGS["compression"]
        )

    lh5_obj, n_rows = store.read("/data/struct_full/array2d", lh5_file)
    assert isinstance(lh5_obj, types.Array)
    assert lh5_obj == types.Array(shape=(23, 56), fill_val=69, dtype=int)


def test_read_array_slice(lh5_file):
    store = lh5.LH5Store()
    lh5_obj, n_rows = store.read(
        "/data/struct_full/array", lh5_file, start_row=1, n_rows=3
    )
    assert isinstance(lh5_obj, types.Array)
    assert n_rows == 3
    assert lh5_obj == lgdo.Array([2, 3, 4])

    lh5_obj, n_rows = store.read(
        "/data/struct_full/array", [lh5_file, lh5_file], start_row=1, n_rows=6
    )
    assert isinstance(lh5_obj, types.Array)
    assert n_rows == 6
    assert lh5_obj == lgdo.Array([2, 3, 4, 5, 1, 2])


def test_read_array_fancy_idx(lh5_file):
    store = lh5.LH5Store()
    lh5_obj, n_rows = store.read("/data/struct_full/array", lh5_file, idx=[0, 3, 4])
    assert isinstance(lh5_obj, types.Array)
    assert n_rows == 3
    assert lh5_obj == lgdo.Array([1, 4, 5])

    lh5_obj, n_rows = store.read(
        "/data/struct_full/array", [lh5_file, lh5_file], idx=[[0, 3, 4], [0, 3, 4]]
    )
    assert isinstance(lh5_obj, types.Array)
    assert n_rows == 6
    assert lh5_obj == lgdo.Array([1, 4, 5, 1, 4, 5])


def test_read_vov(lh5_file):
    store = lh5.LH5Store()
    lh5_obj, n_rows = store.read("/data/struct/vov", lh5_file)
    assert isinstance(lh5_obj, types.VectorOfVectors)

    assert lh5_obj == lgdo.VectorOfVectors(
        [[3, 4, 5], [2], [4, 8, 9, 7]], attrs={"myattr": 2}
    )

    assert n_rows == 3
    assert lh5_obj.attrs["myattr"] == 2

    with h5py.File(lh5_file) as h5f:
        assert (
            h5f["/data/struct/vov/cumulative_length"].compression
            is DEFAULT_HDF5_SETTINGS["compression"]
        )
        assert (
            h5f["/data/struct/vov/flattened_data"].compression
            is DEFAULT_HDF5_SETTINGS["compression"]
        )

    lh5_obj, n_rows = store.read("/data/struct/vov3d", lh5_file)
    assert isinstance(lh5_obj, types.VectorOfVectors)

    assert ak.all(
        lh5_obj.view_as("ak") == ak.Array([[[2], [4, 8, 9, 7]], [[5, 3, 1]], [[3], []]])
    )


def test_read_vov_fancy_idx(lh5_file):
    store = lh5.LH5Store()

    lh5_obj, n_rows = store.read("/data/struct_full/vov", lh5_file, idx=[0], n_rows=1)
    assert isinstance(lh5_obj, types.VectorOfVectors)

    lh5_obj, n_rows = store.read("/data/struct_full/vov", lh5_file, idx=[0, 2])
    assert isinstance(lh5_obj, types.VectorOfVectors)

    assert lh5_obj == types.VectorOfVectors([[1, 2], [2]], attrs={"myattr": 2})
    assert n_rows == 2

    lh5_obj, n_rows = store.read("/data/struct_full/vov3d", lh5_file, idx=[0, 2])
    assert isinstance(lh5_obj, types.VectorOfVectors)

    print(lh5_obj)
    assert lh5_obj == types.VectorOfVectors([[[1, 2], [3, 4, 5]], [[5, 3, 1]]])
    assert n_rows == 2


def test_read_voev(lh5_file):
    store = lh5.LH5Store()
    lh5_obj, n_rows = store.read("/data/struct/voev", lh5_file, decompress=False)
    assert isinstance(lh5_obj, types.VectorOfEncodedVectors)

    desired = [np.array([3, 4, 5]), np.array([2]), np.array([4, 8, 9, 7])]

    for i in range(len(desired)):
        assert (desired[i] == lh5_obj[i][0]).all()

    assert n_rows == 3

    lh5_obj, n_rows = store.read(
        "/data/struct/voev", [lh5_file, lh5_file], decompress=False
    )
    assert isinstance(lh5_obj, types.VectorOfEncodedVectors)
    assert n_rows == 6

    with h5py.File(lh5_file) as h5f:
        assert h5f["/data/struct/voev/encoded_data/flattened_data"].compression is None
        assert (
            h5f["/data/struct/voev/encoded_data/cumulative_length"].compression
            is DEFAULT_HDF5_SETTINGS["compression"]
        )
        assert (
            h5f["/data/struct/voev/decoded_size"].compression
            is DEFAULT_HDF5_SETTINGS["compression"]
        )


def test_read_voev_fancy_idx(lh5_file):
    store = lh5.LH5Store()
    lh5_obj, n_rows = store.read(
        "/data/struct_full/voev", lh5_file, idx=[0, 2], decompress=False
    )
    assert isinstance(lh5_obj, types.VectorOfEncodedVectors)

    desired = [np.array([1, 2]), np.array([2])]

    for i in range(len(desired)):
        assert (desired[i] == lh5_obj[i][0]).all()

    assert n_rows == 2


def test_read_aoesa(lh5_file):
    store = lh5.LH5Store()
    lh5_obj, n_rows = store.read("/data/struct/aoesa", lh5_file)
    assert isinstance(lh5_obj, types.ArrayOfEqualSizedArrays)
    assert (lh5_obj.nda == np.full((3, 5), fill_value=42)).all()


def test_read_table(lh5_file):
    store = lh5.LH5Store()
    lh5_obj, n_rows = store.read("/data/struct/table", lh5_file)
    assert isinstance(lh5_obj, types.Table)
    assert n_rows == 3

    lh5_obj, n_rows = store.read("/data/struct/table", [lh5_file, lh5_file])
    assert n_rows == 6
    assert lh5_obj.attrs["stuff"] == 5
    assert lh5_obj["a"].attrs["attr"] == 9


def test_read_empty_struct(lh5_file):
    store = lh5.LH5Store()
    lh5_obj, n_rows = store.read("/data/struct/empty_struct", lh5_file)
    assert isinstance(lh5_obj, types.Struct)
    assert list(lh5_obj.keys()) == []


def test_read_hdf5_compressed_data(lh5_file):
    store = lh5.LH5Store()
    lh5_obj, n_rows = store.read("/data/struct/table", lh5_file)

    assert "compression" not in lh5_obj["b"].attrs
    with h5py.File(lh5_file) as h5f:
        assert (
            h5f["/data/struct/table/a"].compression
            is DEFAULT_HDF5_SETTINGS["compression"]
        )
        assert h5f["/data/struct/table/b"].compression == "gzip"
        assert h5f["/data/struct/table/c"].compression == "gzip"
        assert h5f["/data/struct/table/c"].compression_opts == 9
        assert h5f["/data/struct/table/d"].compression is None


def test_read_wftable(lh5_file):
    store = lh5.LH5Store()
    lh5_obj, n_rows = store.read("/data/struct/wftable", lh5_file)
    assert isinstance(lh5_obj, types.WaveformTable)
    assert n_rows == 3

    lh5_obj, n_rows = store.read("/data/struct/wftable", [lh5_file, lh5_file])
    assert n_rows == 6
    assert lh5_obj.values.attrs["custom"] == 8

    with h5py.File(lh5_file) as h5f:
        assert (
            h5f["/data/struct/wftable/values"].compression
            is DEFAULT_HDF5_SETTINGS["compression"]
        )
        assert (
            h5f["/data/struct/wftable/t0"].compression
            is DEFAULT_HDF5_SETTINGS["compression"]
        )
        assert (
            h5f["/data/struct/wftable/dt"].compression
            is DEFAULT_HDF5_SETTINGS["compression"]
        )


def test_read_wftable_encoded(lh5_file):
    store = lh5.LH5Store()
    lh5_obj, n_rows = store.read("/data/struct/wftable_enc", lh5_file, decompress=False)
    assert isinstance(lh5_obj, types.WaveformTable)
    assert isinstance(lh5_obj.values, types.ArrayOfEncodedEqualSizedArrays)
    assert n_rows == 3
    assert lh5_obj.values.attrs["codec"] == "radware_sigcompress"
    assert "codec_shift" in lh5_obj.values.attrs

    lh5_obj, n_rows = store.read("/data/struct/wftable_enc/values", lh5_file)
    assert isinstance(lh5_obj, lgdo.ArrayOfEqualSizedArrays)
    assert n_rows == 3

    lh5_obj, n_rows = store.read("/data/struct/wftable_enc", lh5_file)
    assert isinstance(lh5_obj, lgdo.WaveformTable)
    assert isinstance(lh5_obj.values, lgdo.ArrayOfEqualSizedArrays)
    assert n_rows == 3

    lh5_obj_chain, n_rows = store.read(
        "/data/struct/wftable_enc", [lh5_file, lh5_file], decompress=False
    )
    assert n_rows == 6
    assert isinstance(lh5_obj_chain.values, lgdo.ArrayOfEncodedEqualSizedArrays)

    lh5_obj_chain, n_rows = store.read(
        "/data/struct/wftable_enc", [lh5_file, lh5_file], decompress=True
    )
    assert isinstance(lh5_obj_chain.values, lgdo.ArrayOfEqualSizedArrays)
    assert np.array_equal(lh5_obj_chain.values[:3], lh5_obj.values)
    assert np.array_equal(lh5_obj_chain.values[3:], lh5_obj.values)
    assert n_rows == 6

    with h5py.File(lh5_file) as h5f:
        assert (
            h5f[
                "/data/struct/wftable_enc/values/encoded_data/flattened_data"
            ].compression
            is None
        )
        assert h5f["/data/struct/wftable_enc/values/decoded_size"].compression is None
        assert (
            h5f["/data/struct/wftable_enc/t0"].compression
            is DEFAULT_HDF5_SETTINGS["compression"]
        )
        assert (
            h5f["/data/struct/wftable_enc/dt"].compression
            is DEFAULT_HDF5_SETTINGS["compression"]
        )


def test_read_with_field_mask(lh5_file):
    store = lh5.LH5Store()

    lh5_obj, n_rows = store.read("/data/struct_full", lh5_file, field_mask=["array"])
    assert list(lh5_obj.keys()) == ["array"]

    lh5_obj, n_rows = store.read(
        "/data/struct_full", lh5_file, field_mask=("array", "table")
    )
    assert sorted(lh5_obj.keys()) == ["array", "table"]

    lh5_obj, n_rows = store.read(
        "/data/struct_full", lh5_file, field_mask={"array": True}
    )
    assert list(lh5_obj.keys()) == ["array"]

    lh5_obj, n_rows = store.read(
        "/data/struct_full", lh5_file, field_mask={"vov": False, "voev": False}
    )
    assert sorted(lh5_obj.keys()) == [
        "aoesa",
        "array",
        "array2d",
        "empty_struct",
        "scalar",
        "table",
        "vov3d",
        "wftable",
        "wftable_enc",
    ]


def test_read_lgnd_array(lgnd_file):
    store = lh5.LH5Store()

    lh5_obj, n_rows = store.read("/geds/raw/baseline", lgnd_file)
    assert isinstance(lh5_obj, types.Array)
    assert n_rows == 100
    assert len(lh5_obj) == 100

    lh5_obj, n_rows = store.read("/geds/raw/waveform/values", lgnd_file)
    assert isinstance(lh5_obj, types.ArrayOfEqualSizedArrays)


def test_read_lgnd_array_fancy_idx(lgnd_file):
    store = lh5.LH5Store()

    lh5_obj, n_rows = store.read(
        "/geds/raw/baseline", lgnd_file, idx=[2, 4, 6, 9, 11, 16, 68]
    )
    assert isinstance(lh5_obj, types.Array)
    assert n_rows == 7
    assert len(lh5_obj) == 7
    assert (lh5_obj.nda == [13508, 14353, 14525, 14341, 15079, 11675, 13995]).all()


def test_read_lgnd_vov(lgnd_file):
    store = lh5.LH5Store()

    lh5_obj, n_rows = store.read("/geds/raw/tracelist", lgnd_file)
    assert isinstance(lh5_obj, types.VectorOfVectors)
    assert n_rows == 100
    assert len(lh5_obj) == 100


def test_read_lgnd_vov_fancy_idx(lgnd_file):
    store = lh5.LH5Store()

    lh5_obj, n_rows = store.read(
        "/geds/raw/tracelist", lgnd_file, idx=[2, 4, 6, 9, 11, 16, 68]
    )
    assert isinstance(lh5_obj, types.VectorOfVectors)
    assert n_rows == 7
    assert len(lh5_obj) == 7
    assert (lh5_obj.cumulative_length.nda == [1, 2, 3, 4, 5, 6, 7]).all()
    assert (lh5_obj.flattened_data.nda == [40, 60, 64, 60, 64, 28, 60]).all()


def test_read_array_concatenation(lgnd_file):
    store = lh5.LH5Store()
    lh5_obj, n_rows = store.read("/geds/raw/baseline", [lgnd_file, lgnd_file])
    assert isinstance(lh5_obj, types.Array)
    assert n_rows == 200
    assert len(lh5_obj) == 200


def test_read_lgnd_waveform_table(lgnd_file):
    store = lh5.LH5Store()

    lh5_obj, n_rows = store.read("/geds/raw/waveform", lgnd_file)
    assert isinstance(lh5_obj, types.WaveformTable)

    lh5_obj, n_rows = store.read(
        "/geds/raw/waveform",
        lgnd_file,
        start_row=10,
        n_rows=10,
        field_mask=["t0", "dt"],
    )

    assert isinstance(lh5_obj, types.Table)
    assert list(lh5_obj.keys()) == ["t0", "dt"]
    assert len(lh5_obj) == 10


def test_read_lgnd_waveform_table_fancy_idx(lgnd_file):
    store = lh5.LH5Store()

    lh5_obj, n_rows = store.read(
        "/geds/raw/waveform",
        lgnd_file,
        idx=[7, 9, 25, 27, 33, 38, 46, 52, 57, 59, 67, 71, 72, 82, 90, 92, 93, 94, 97],
    )
    assert isinstance(lh5_obj, types.WaveformTable)
    assert len(lh5_obj) == 19


def test_read_compressed_lgnd_waveform_table(lgnd_file, enc_lgnd_file):
    store = lh5.LH5Store()
    wft, _ = store.read("/geds/raw/waveform", enc_lgnd_file)
    assert isinstance(wft.values, types.ArrayOfEqualSizedArrays)
    assert "compression" not in wft.values.attrs
