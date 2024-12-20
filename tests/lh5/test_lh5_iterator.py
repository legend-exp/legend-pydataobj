# ruff: noqa: ARG001

from __future__ import annotations

import numpy as np
import pytest

import lgdo
from lgdo import lh5


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
        more_lgnd_files[1],
        "ch1084803/hit",
        field_mask=["is_valid_0vbb"],
        buffer_len=5,
        friend=lh5_raw_it,
    )

    lh5_obj = lh5_it.read(0)

    assert len(lh5_obj) == 5
    assert isinstance(lh5_obj, lgdo.Table)
    assert set(lh5_obj.keys()) == {"waveform", "baseline", "is_valid_0vbb"}


def test_iterate(more_lgnd_files):
    # iterate through all hit groups in all files; there are 10 entries in
    # each group/file
    lh5_it = lh5.LH5Iterator(
        more_lgnd_files[1],
        ["ch1084803/hit", "ch1084804/hit", "ch1121600/hit"],
        field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
        entry_list=[0, 1, 2, 3, 5, 8, 13, 21, 34, 55],
        buffer_len=5,
    )

    exp_loc_entries = [[0, 1, 2, 3, 5], [8, 3, 1, 4, 5]]
    exp_glob_entries = [[0, 1, 2, 3, 5], [8, 13, 21, 34, 55]]
    exp_files = [
        [more_lgnd_files[1][0]] * 5,
        [more_lgnd_files[1][0]] * 3 + [more_lgnd_files[1][1]] * 2,
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
        more_lgnd_files[1],
        ["ch1084803/hit", "ch1084804/hit", "ch1121600/hit"],
        field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
        entry_list=[[0, 1, 2, 3, 5, 8], [3], [1], [4], [], [5]],
        buffer_len=5,
    )

    for lh5_out in lh5_it:
        assert set(lh5_out.keys()) == {"is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"}
        assert lh5_it.current_i_entry % 5 == 0
        assert len(lh5_out) == 5
    print(lh5_it.get_global_entrylist())
    assert all(
        i == j
        for i, j in zip(
            lh5_it.get_global_entrylist(), [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        )
    )

    lh5_it = lh5.LH5Iterator(
        more_lgnd_files[1],
        [["ch1084803/hit", "ch1084804/hit"], ["ch1084803/hit", "ch1121600/hit"]],
        field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
        buffer_len=5,
    )

    for lh5_out in lh5_it:
        assert set(lh5_out.keys()) == {"is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"}
        assert lh5_it.current_i_entry % 5 == 0
        assert len(lh5_out) == 5

    with pytest.raises(ValueError):
        lh5.LH5Iterator(
            more_lgnd_files[1],
            [["ch1084803/hit", "ch1084804/hit"], "ch1084803/hit"],
            field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
            buffer_len=5,
        )

    with pytest.raises(ValueError):
        lh5.LH5Iterator(
            more_lgnd_files[1],
            [["ch1084803/hit"], ["ch1084804/hit"], ["ch1084803/hit"]],
            field_mask=["is_valid_0vbb", "timestamp", "zacEmax_ctc_cal"],
            buffer_len=5,
        )
