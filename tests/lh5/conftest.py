from __future__ import annotations

import numpy as np
import pytest

import lgdo
from lgdo import compression, lh5, types


@pytest.fixture(scope="module")
def lgnd_file(lgnd_test_data):
    return lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5")


@pytest.fixture(scope="module")
def lh5_file(tmptestdir):
    store = lh5.LH5Store()

    struct = lgdo.Struct()
    struct.add_field("scalar", lgdo.Scalar(value=10, attrs={"sth": 1}))
    struct.add_field("array", types.Array(nda=np.array([1, 2, 3, 4, 5])))
    struct.add_field("array2d", types.Array(shape=(23, 56), fill_val=69, dtype=int))
    struct.add_field(
        "aoesa",
        types.ArrayOfEqualSizedArrays(shape=(5, 5), dtype=np.float32, fill_val=42),
    )
    struct.add_field(
        "vov",
        types.VectorOfVectors(
            flattened_data=types.Array(
                nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1])
            ),
            cumulative_length=types.Array(nda=np.array([2, 5, 6, 10, 13])),
            attrs={"myattr": 2},
        ),
    )
    struct.add_field(
        "vov3d",
        types.VectorOfVectors(
            [[[1, 2], [3, 4, 5]], [[2], [4, 8, 9, 7]], [[5, 3, 1]], [[3], []], [[3, 4]]]
        ),
    )

    struct.add_field(
        "voev",
        types.VectorOfEncodedVectors(
            encoded_data=types.VectorOfVectors(
                flattened_data=types.Array(
                    nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1])
                ),
                cumulative_length=types.Array(nda=np.array([2, 5, 6, 10, 13])),
            ),
            decoded_size=types.Array(shape=5, fill_val=6),
        ),
    )

    col_dict = {
        "a": lgdo.Array(nda=np.array([1, 2, 3, 4]), attrs={"attr": 9}),
        "b": lgdo.Array(nda=np.array([5, 6, 7, 8]), attrs={"compression": "gzip"}),
        "c": lgdo.Array(
            nda=np.array([5, 6, 7, 8]),
            attrs={"compression": {"compression": "gzip", "compression_opts": 9}},
        ),
        "d": lgdo.Array(
            nda=np.array([5, 6, 7, 8]),
            attrs={"compression": None},
        ),
    }

    struct.add_field("table", types.Table(col_dict=col_dict, attrs={"stuff": 5}))

    struct.add_field(
        "wftable",
        types.WaveformTable(
            t0=types.Array(np.zeros(10)),
            dt=types.Array(np.full(10, fill_value=1)),
            values=types.ArrayOfEqualSizedArrays(
                shape=(10, 1000), dtype=np.uint16, fill_val=100, attrs={"custom": 8}
            ),
        ),
    )

    struct.add_field(
        "wftable_enc",
        types.WaveformTable(
            t0=types.Array(np.zeros(10)),
            dt=types.Array(np.full(10, fill_value=1)),
            values=compression.encode(
                struct["wftable"].values,
                codec=compression.RadwareSigcompress(codec_shift=-32768),
            ),
        ),
    )

    struct.add_field("empty_struct", types.Struct())

    store.write(
        struct,
        "struct",
        f"{tmptestdir}/tmp-pygama-lgdo-types.lh5",
        group="/data",
        start_row=1,
        n_rows=3,
        wo_mode="overwrite_file",
    )

    store.write(
        struct,
        "struct_full",
        f"{tmptestdir}/tmp-pygama-lgdo-types.lh5",
        group="/data",
        wo_mode="append",
    )

    assert struct["table"]["b"].attrs["compression"] == "gzip"

    return f"{tmptestdir}/tmp-pygama-lgdo-types.lh5"


@pytest.fixture(scope="module")
def enc_lgnd_file(lgnd_file, tmptestdir):
    store = lh5.LH5Store()
    wft, n_rows = store.read("/geds/raw/waveform", lgnd_file)
    wft.values.attrs["compression"] = compression.RadwareSigcompress(codec_shift=-32768)
    store.write(
        wft,
        "/geds/raw/waveform",
        f"{tmptestdir}/tmp-pygama-compressed-wfs.lh5",
        wo_mode="overwrite_file",
    )
    return f"{tmptestdir}/tmp-pygama-compressed-wfs.lh5"
