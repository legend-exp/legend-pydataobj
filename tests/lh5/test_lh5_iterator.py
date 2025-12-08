from __future__ import annotations

import shutil
from copy import deepcopy

import awkward as ak
import numpy as np
import pandas as pd
import pytest
from hist import axis

import lgdo
from lgdo import Table, lh5


@pytest.fixture(scope="module")
def lgnd_file(lgnd_test_data):
    return lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5")


def test_basics(lgnd_file):
    lh5_it = lh5.LH5Iterator(
        lgnd_file,
        "/geds/raw",
        entry_list=range(100),
        field_mask=["baseline"],
        buffer_len=5,
    )

    lh5_obj = lh5_it.read(4)
    assert len(lh5_obj) == 5
    assert isinstance(lh5_obj, lgdo.Table)
    assert list(lh5_obj.keys()) == ["baseline"]
    assert (
        lh5_obj["baseline"].nda == np.array([14353, 14254, 14525, 11656, 13576])
    ).all()

    for lh5_obj in lh5_it:
        entry = lh5_it.current_i_entry
        assert len(lh5_obj) == 5
        assert entry % 5 == 0
        assert all(lh5_it.current_local_entries == np.arange(entry, entry + 5))
        assert all(lh5_it.current_global_entries == np.arange(entry, entry + 5))
        assert all(lh5_it.current_files == [lgnd_file] * 5)
        assert all(lh5_it.current_groups == ["/geds/raw"] * 5)


def test_errors(lgnd_file):
    with pytest.raises(RuntimeError):
        lh5.LH5Iterator("non-existent-file.lh5", "random-group")

    with pytest.raises(ValueError):
        lh5.LH5Iterator(1, 2)

    with pytest.raises(ValueError):
        lh5.LH5Iterator(
            lgnd_file,
            "/geds/raw",
            entry_list=range(100),
            entry_mask=np.ones(100, "bool"),
        )


def test_lgnd_waveform_table_fancy_idx(lgnd_file):
    lh5_it = lh5.LH5Iterator(
        lgnd_file,
        "geds/raw/waveform",
        entry_list=[
            7,
            9,
            25,
            27,
            33,
            38,
            46,
            52,
            57,
            59,
            67,
            71,
            72,
            82,
            90,
            92,
            93,
            94,
            97,
        ],
        buffer_len=5,
    )

    lh5_obj = lh5_it.read(0)
    assert isinstance(lh5_obj, lgdo.WaveformTable)
    assert len(lh5_obj) == 5


@pytest.fixture(scope="module")
def more_lgnd_files(lgnd_test_data):
    return [
        [
            lgnd_test_data.get_path(
                "lh5/prod-ref-l200/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_raw.lh5"
            ),
            lgnd_test_data.get_path(
                "lh5/prod-ref-l200/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_raw.lh5"
            ),
        ],
        [
            lgnd_test_data.get_path(
                "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_dsp.lh5"
            ),
            lgnd_test_data.get_path(
                "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_dsp.lh5"
            ),
        ],
        [
            lgnd_test_data.get_path(
                "lh5/prod-ref-l200/generated/tier/hit/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_hit.lh5"
            ),
            lgnd_test_data.get_path(
                "lh5/prod-ref-l200/generated/tier/hit/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_hit.lh5"
            ),
        ],
    ]


def test_friend(more_lgnd_files):
    lh5_raw_it = lh5.LH5Iterator(
        more_lgnd_files[0],
        "ch1084803/raw",
        field_mask=["waveform", "baseline"],
        buffer_len=5,
    )
    lh5_it = lh5.LH5Iterator(
        more_lgnd_files[2],
        "ch1084803/hit",
        field_mask=["is_valid_0vbb"],
        buffer_len=5,
        friend=lh5_raw_it,
    )

    lh5_obj = lh5_it.read(0)

    assert len(lh5_obj) == 5
    assert isinstance(lh5_obj, lgdo.Table)
    assert set(lh5_obj.keys()) == {"waveform", "baseline", "is_valid_0vbb"}


def test_friend_conflict(more_lgnd_files):
    lh5_raw_it = lh5.LH5Iterator(
        more_lgnd_files[0],
        "ch1084803/raw",
        field_mask=["waveform", "baseline"],
        buffer_len=5,
    )

    lh5_it = lh5.LH5Iterator(
        more_lgnd_files[1],
        "ch1084803/dsp",
        field_mask=["baseline", "wf_max"],
        buffer_len=5,
        friend=lh5_raw_it,
    )
    lh5_obj = lh5_it.read(0)
    assert set(lh5_obj.keys()) == {"waveform", "baseline", "wf_max"}
    assert lh5_obj["waveform"] == lh5.read(
        "ch1084803/raw/waveform", more_lgnd_files[0], n_rows=5
    )
    assert lh5_obj["baseline"] == lh5.read(
        "ch1084803/dsp/baseline", more_lgnd_files[1], n_rows=5
    )
    assert lh5_obj["wf_max"] == lh5.read(
        "ch1084803/dsp/wf_max", more_lgnd_files[1], n_rows=5
    )

    lh5_it = lh5.LH5Iterator(
        more_lgnd_files[1],
        "ch1084803/dsp",
        field_mask=["baseline", "wf_max"],
        buffer_len=5,
        friend=lh5_raw_it,
        friend_prefix="raw_",
    )
    lh5_obj = lh5_it.read(0)
    assert set(lh5_obj.keys()) == {"raw_waveform", "raw_baseline", "baseline", "wf_max"}
    assert lh5_obj["raw_waveform"] == lh5.read(
        "ch1084803/raw/waveform", more_lgnd_files[0], n_rows=5
    )
    assert lh5_obj["raw_baseline"] == lh5.read(
        "ch1084803/raw/baseline", more_lgnd_files[0], n_rows=5
    )
    assert lh5_obj["baseline"] == lh5.read(
        "ch1084803/dsp/baseline", more_lgnd_files[1], n_rows=5
    )
    assert lh5_obj["wf_max"] == lh5.read(
        "ch1084803/dsp/wf_max", more_lgnd_files[1], n_rows=5
    )

    lh5_it = lh5.LH5Iterator(
        more_lgnd_files[1],
        "ch1084803/dsp",
        field_mask=["baseline", "wf_max"],
        buffer_len=5,
        friend=lh5_raw_it,
        friend_suffix="_raw",
    )
    lh5_obj = lh5_it.read(0)
    assert set(lh5_obj.keys()) == {"waveform_raw", "baseline_raw", "baseline", "wf_max"}
    assert lh5_obj["waveform_raw"] == lh5.read(
        "ch1084803/raw/waveform", more_lgnd_files[0], n_rows=5
    )
    assert lh5_obj["baseline_raw"] == lh5.read(
        "ch1084803/raw/baseline", more_lgnd_files[0], n_rows=5
    )
    assert lh5_obj["baseline"] == lh5.read(
        "ch1084803/dsp/baseline", more_lgnd_files[1], n_rows=5
    )
    assert lh5_obj["wf_max"] == lh5.read(
        "ch1084803/dsp/wf_max", more_lgnd_files[1], n_rows=5
    )

    lh5_it.reset_field_mask(["waveform_raw", "baseline", "wf_max"])
    lh5_obj = lh5_it.read(0)
    assert set(lh5_obj.keys()) == {"waveform_raw", "baseline", "wf_max"}
    assert lh5_obj["waveform_raw"] == lh5.read(
        "ch1084803/raw/waveform", more_lgnd_files[0], n_rows=5
    )
    assert lh5_obj["baseline"] == lh5.read(
        "ch1084803/dsp/baseline", more_lgnd_files[1], n_rows=5
    )
    assert lh5_obj["wf_max"] == lh5.read(
        "ch1084803/dsp/wf_max", more_lgnd_files[1], n_rows=5
    )

    lh5_it.reset_field_mask({"baseline_raw": False, "wf_max": False})
    lh5_obj = lh5_it.read(0)
    assert {"waveform_raw", "baseline"}.issubset(set(lh5_obj.keys()))
    assert {"baseline_raw", "wf_max"}.isdisjoint(set(lh5_obj.keys()))
    assert lh5_obj["waveform_raw"] == lh5.read(
        "ch1084803/raw/waveform", more_lgnd_files[0], n_rows=5
    )
    assert lh5_obj["baseline"] == lh5.read(
        "ch1084803/dsp/baseline", more_lgnd_files[1], n_rows=5
    )


def test_iterate(more_lgnd_files):
    # iterate through all hit groups in all files; there are 10 entries in
    # each group/file
    lh5_it = lh5.LH5Iterator(
        more_lgnd_files[2],
        ["ch1084803/hit", "ch1084804/hit", "ch1121600/hit"],
        field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
        entry_list=[0, 1, 2, 3, 5, 8, 13, 21, 34, 55],
        buffer_len=5,
    )

    exp_loc_entries = [[0, 1, 2, 3, 5], [8, 3, 1, 4, 5]]
    exp_glob_entries = [[0, 1, 2, 3, 5], [8, 13, 21, 34, 55]]
    exp_files = [
        [more_lgnd_files[2][0]] * 5,
        [more_lgnd_files[2][0]] * 3 + [more_lgnd_files[2][1]] * 2,
    ]
    exp_groups = [
        ["ch1084803/hit"] * 5,
        [
            "ch1084803/hit",
            "ch1084804/hit",
            "ch1121600/hit",
            "ch1084803/hit",
            "ch1121600/hit",
        ],
    ]

    for lh5_out in lh5_it:
        assert set(lh5_out.keys()) == {"is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"}
        entry = lh5_it.current_i_entry
        assert entry % 5 == 0
        assert len(lh5_out) == 5
        assert all(lh5_it.current_local_entries == exp_loc_entries[entry // 5])
        assert all(lh5_it.current_global_entries == exp_glob_entries[entry // 5])
        assert all(lh5_it.current_files == exp_files[entry // 5])
        assert all(lh5_it.current_groups == exp_groups[entry // 5])

    lh5_it = lh5.LH5Iterator(
        more_lgnd_files[2],
        ["ch1084803/hit", "ch1084804/hit", "ch1121600/hit"],
        field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
        entry_list=[[0, 1, 2, 3, 5, 8], [3], [1], [4], [], [5]],
        buffer_len=5,
    )

    for lh5_out in lh5_it:
        assert set(lh5_out.keys()) == {"is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"}
        assert lh5_it.current_i_entry % 5 == 0
        assert len(lh5_out) == 5

    assert all(
        i == j
        for i, j in zip(
            lh5_it.get_global_entrylist(), [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        )
    )

    lh5_it = lh5.LH5Iterator(
        more_lgnd_files[2],
        [["ch1084803/hit", "ch1084804/hit"], ["ch1084803/hit", "ch1121600/hit"]],
        field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
        buffer_len=5,
    )

    for lh5_out in lh5_it:
        assert set(lh5_out.keys()) == {"is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"}
        assert lh5_it.current_i_entry % 5 == 0
        assert len(lh5_out) == 5

    # different number of file sets and group sets
    with pytest.raises(ValueError):
        lh5.LH5Iterator(
            more_lgnd_files[2],
            [["ch1084803/hit", "ch1084804/hit"]],
            field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
            buffer_len=5,
        )

    with pytest.raises(ValueError):
        lh5.LH5Iterator(
            more_lgnd_files[2],
            [["ch1084803/hit"], ["ch1084804/hit"], ["ch1084803/hit"]],
            field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
            buffer_len=5,
        )


def test_group_data(more_lgnd_files):
    # test provision of metadata about groups
    lh5_it = lh5.LH5Iterator(
        more_lgnd_files[2],
        ["ch1084803/hit", "ch1084804/hit", "ch1121600/hit"],
        field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
        buffer_len=5,
        group_data={"chan": [1084803, 1084804, 1121600]},
    )

    exp_chan = [
        [1084803] * 5,
        [1084803] * 5,
        [1084804] * 5,
        [1084804] * 5,
        [1121600] * 5,
        [1121600] * 5,
        [1084803] * 5,
        [1084803] * 5,
        [1084804] * 5,
        [1084804] * 5,
        [1121600] * 5,
        [1121600] * 5,
    ]
    for tb, ec in zip(lh5_it, exp_chan):
        assert set(tb.keys()) == {
            "is_valid_0vbb",
            "timestamp",
            "zacEmax_ctc_cal",
            "chan",
        }
        assert all(tb.chan.nda == ec)

    # group_data must be same shape as field_mask
    with pytest.raises(ValueError):
        lh5_it = lh5.LH5Iterator(
            more_lgnd_files[2],
            ["ch1084803/hit", "ch1084804/hit", "ch1121600/hit"],
            field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
            buffer_len=5,
            group_data={"chan": [1084803, 1084804, 1121600, 1234567]},
        )

    # group_data provided using pandas dataframe
    lh5_it = lh5.LH5Iterator(
        more_lgnd_files[2],
        ["ch1084803/hit", "ch1084804/hit", "ch1121600/hit"],
        field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
        buffer_len=5,
        group_data=pd.DataFrame({"chan": [1084803, 1084804, 1121600]}),
    )
    for tb, ec in zip(lh5_it, exp_chan):
        assert set(tb.keys()) == {
            "is_valid_0vbb",
            "timestamp",
            "zacEmax_ctc_cal",
            "chan",
        }
        assert all(tb.chan.nda == ec)

    # group_data provided using awkward array as array of records
    lh5_it = lh5.LH5Iterator(
        more_lgnd_files[2],
        ["ch1084803/hit", "ch1084804/hit", "ch1121600/hit"],
        field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
        buffer_len=5,
        group_data=pd.DataFrame(
            [{"chan": 1084803}, {"chan": 1084804}, {"chan": 1121600}]
        ),
    )
    for tb, ec in zip(lh5_it, exp_chan):
        assert set(tb.keys()) == {
            "is_valid_0vbb",
            "timestamp",
            "zacEmax_ctc_cal",
            "chan",
        }
        assert all(tb.chan.nda == ec)


def test_range(lgnd_file):
    lh5_it = lh5.LH5Iterator(
        lgnd_file,
        "/geds/raw",
        field_mask=["baseline"],
        buffer_len=5,
        i_start=7,
        n_entries=13,
    )

    # Test error when n_entries > buffer_len
    with pytest.raises(ValueError):
        lh5_obj = lh5_it.read(4, n_entries=7)

    lh5_obj = lh5_it.read(4, n_entries=3)
    assert len(lh5_obj) == 3
    assert isinstance(lh5_obj, lgdo.Table)
    assert list(lh5_obj.keys()) == ["baseline"]
    assert (lh5_obj["baseline"].nda == np.array([14353, 14254, 14525])).all()

    exp_i_entries = [7, 12, 17]
    exp_lens = [5, 5, 3]
    for lh5_obj, exp_i, exp_len in zip(lh5_it, exp_i_entries, exp_lens):
        entry = lh5_it.current_i_entry
        assert len(lh5_obj) == exp_len
        assert entry == exp_i
        assert all(lh5_it.current_local_entries == np.arange(entry, entry + exp_len))
        assert all(lh5_it.current_global_entries == np.arange(entry, entry + exp_len))
        assert all(lh5_it.current_files == [lgnd_file] * exp_len)
        assert all(lh5_it.current_groups == ["/geds/raw"] * exp_len)


# function used in next test...
def return_tb(tb, _):
    return tb


def test_map(more_lgnd_files):
    # iterate through all hit groups in all files; there are 10 entries in
    # each group/file
    lh5_it = lh5.LH5Iterator(
        more_lgnd_files[2],
        ["ch1084803/hit", "ch1084804/hit", "ch1121600/hit"],
        field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
        buffer_len=5,
    )

    # test that output is same with map as looping over iterator
    exp_out = []
    for tb_out in lh5_it:
        exp_out += [deepcopy(tb_out)]

    map_out = lh5_it.map(return_tb)

    assert all(tb_out == tb_map for tb_out, tb_map in zip(exp_out, map_out))

    # test use of append as aggregator
    tb_exp = exp_out[0]
    for tb in exp_out[1:]:
        tb_exp.append(tb)

    tb_map = lh5_it.map(return_tb, aggregate=Table.append)
    assert tb_exp == tb_map

    # test use of aggregate and init to perform reduction
    sum_exp = np.sum(tb_exp["zacEmax_ctc_cal"].nda)

    def sumE(tb, _):
        return np.sum(tb["zacEmax_ctc_cal"].nda)

    sum_map = lh5_it.map(sumE, aggregate=np.add, init=-10.0)
    assert sum_exp - 10.0 == sum_map

    # test multiprocessing; note this only works as done here if
    # buffer_len evenly divides iterator length for each file!
    map_mp = lh5_it.map(return_tb, processes=2)

    assert all(tb == tb_mp for tb, tb_mp in zip(lh5_it, map_mp))


def query_lgdo(tb, _):
    mask = tb["zacEmax_ctc_cal"].nda > 200
    for field in tb:
        tb[field].nda = tb[field].nda[mask]
    tb.resize(np.sum(mask))
    return tb


def query_pd(tb, _):
    mask = tb["zacEmax_ctc_cal"].nda > 200
    return tb.view_as("pd").loc[mask]


def query_ak(tb, _):
    mask = tb["zacEmax_ctc_cal"].nda > 200
    return tb.view_as("ak")[mask]


def query_np(tb, _):
    mask = tb["zacEmax_ctc_cal"].nda > 200
    return tb["zacEmax_ctc_cal"].nda[mask]


def test_query(more_lgnd_files):
    lh5_it = lh5.LH5Iterator(
        more_lgnd_files[2],
        ["ch1084803/hit", "ch1084804/hit", "ch1121600/hit"],
        field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
        buffer_len=5,
    )

    # get a table with all entries and then apply our filter
    tb_exp = None
    for tb in lh5_it:
        if tb_exp is None:
            tb_exp = deepcopy(tb)
        else:
            tb_exp.append(tb)
    tb_exp = query_lgdo(tb_exp, None)

    # filter returning Table
    tb_lgdo = lh5_it.query(query_lgdo)
    assert tb_lgdo == tb_exp
    tb_lgdo_mp = lh5_it.query(query_lgdo, processes=3)
    assert tb_lgdo_mp == tb_exp

    pd_out = lh5_it.query(query_pd)
    assert pd_out.equals(tb_exp.view_as("pd"))
    pd_out_mp = lh5_it.query(query_pd, processes=3)
    assert pd_out_mp.equals(tb_exp.view_as("pd"))

    ak_out = lh5_it.query(query_ak)
    assert ak.array_equal(ak_out, tb_exp.view_as("ak"))
    ak_out_mp = lh5_it.query(query_ak, processes=3)
    assert ak.array_equal(ak_out_mp, tb_exp.view_as("ak"))

    np_out = lh5_it.query(query_np)
    assert np.all(np_out == tb_exp["zacEmax_ctc_cal"])
    np_out_mp = lh5_it.query(query_np, processes=3)
    assert np.all(np_out_mp == tb_exp["zacEmax_ctc_cal"])

    pd_out_str = lh5_it.query("zacEmax_ctc_cal>200", library="pd")
    assert pd_out_str.equals(tb_exp.view_as("pd"))
    pd_out_str_mp = lh5_it.query("zacEmax_ctc_cal>200", processes=3, library="pd")
    assert pd_out_str_mp.equals(tb_exp.view_as("pd"))


def test_hist(more_lgnd_files):
    lh5_it = lh5.LH5Iterator(
        more_lgnd_files[2],
        ["ch1084803/hit", "ch1084804/hit", "ch1121600/hit"],
        field_mask=["timestamp", "zacEmax_ctc_cal"],
        buffer_len=5,
    )

    # get a table with all entries and then apply our filter
    tb_exp = None
    for tb in lh5_it:
        if tb_exp is None:
            tb_exp = deepcopy(tb)
        else:
            tb_exp.append(tb)
    tb_exp = query_lgdo(tb_exp, None)

    h_exp, xedges, yedges = np.histogram2d(
        tb_exp["timestamp"], tb_exp["zacEmax_ctc_cal"]
    )
    # otherwise some entries end up in the overflow/underflow bins
    xedges[0] *= 0.99
    xedges[-1] *= 1.01
    yedges[0] *= 0.99
    yedges[-1] *= 1.01

    h_lgdo = lh5_it.hist(
        [axis.Variable(xedges), axis.Variable(yedges)],
        where=query_lgdo,
        keys=["timestamp", "zacEmax_ctc_cal"],
    )
    assert np.all(np.array(h_lgdo) == h_exp)
    h_lgdo_mp = lh5_it.hist(
        [axis.Variable(xedges), axis.Variable(yedges)],
        where=query_lgdo,
        keys=["timestamp", "zacEmax_ctc_cal"],
        processes=2,
    )
    assert np.all(np.array(h_lgdo_mp) == h_exp)

    h_pd = lh5_it.hist(
        [axis.Variable(xedges), axis.Variable(yedges)],
        where=query_pd,
        keys=["timestamp", "zacEmax_ctc_cal"],
    )
    assert np.all(np.array(h_pd) == h_exp)
    h_pd_mp = lh5_it.hist(
        [axis.Variable(xedges), axis.Variable(yedges)],
        where=query_pd,
        keys=["timestamp", "zacEmax_ctc_cal"],
        processes=2,
    )
    assert np.all(np.array(h_pd_mp) == h_exp)

    h_ak = lh5_it.hist(
        [axis.Variable(xedges), axis.Variable(yedges)],
        where=query_ak,
        keys=["timestamp", "zacEmax_ctc_cal"],
    )
    assert np.all(np.array(h_ak) == h_exp)
    h_ak_mp = lh5_it.hist(
        [axis.Variable(xedges), axis.Variable(yedges)],
        where=query_ak,
        keys=["timestamp", "zacEmax_ctc_cal"],
        processes=2,
    )
    assert np.all(np.array(h_ak_mp) == h_exp)

    h_np = lh5_it.hist(
        [axis.Variable(yedges)],
        where=query_np,
        keys=["zacEmax_ctc_cal"],
    )
    assert np.all(np.array(h_np) == np.sum(h_exp, axis=0))
    h_np_mp = lh5_it.hist(
        [axis.Variable(yedges)],
        where=query_np,
        keys=["zacEmax_ctc_cal"],
        processes=2,
    )
    assert np.all(np.array(h_np_mp) == np.sum(h_exp, axis=0))

    h_pd_str = lh5_it.hist(
        [axis.Variable(xedges), axis.Variable(yedges)],
        where="zacEmax_ctc_cal>200",
        keys=["timestamp", "zacEmax_ctc_cal"],
    )
    assert np.all(np.array(h_pd_str) == h_exp)
    h_pd_str_mp = lh5_it.hist(
        [axis.Variable(xedges), axis.Variable(yedges)],
        where="zacEmax_ctc_cal>200",
        keys=["timestamp", "zacEmax_ctc_cal"],
        processes=2,
    )
    assert np.all(np.array(h_pd_str_mp) == h_exp)


def test_iterator_wo_mode_write(tmp_path, lh5_file):
    dst = tmp_path / "rw.lh5"
    shutil.copy(lh5_file, dst)
    it = lh5.LH5Iterator(
        dst.as_posix(), "/data/struct_full/array", h5py_open_mode="append"
    )
    store = it.lh5_st
    store.write(
        lgdo.Array(nda=np.array([0], dtype=int)),
        "dummy",
        dst.as_posix(),
        group="/data",
        wo_mode="append",
    )
    assert len(store.read("/data/dummy", dst.as_posix())) == 1
    assert len(it.read(0)) > 0
