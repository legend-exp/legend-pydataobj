from __future__ import annotations

from lgdo import lh5


def test_ls(lgnd_test_data):
    lgnd_file = lgnd_test_data.get_path(
        "lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"
    )
    assert lh5.ls(lgnd_file) == ["geds"]
    assert lh5.ls(lgnd_file, "/*/raw") == ["geds/raw"]
    assert lh5.ls(lgnd_file, "geds/raw/") == [
        "geds/raw/baseline",
        "geds/raw/channel",
        "geds/raw/energy",
        "geds/raw/ievt",
        "geds/raw/numtraces",
        "geds/raw/packet_id",
        "geds/raw/timestamp",
        "geds/raw/tracelist",
        "geds/raw/waveform",
        "geds/raw/wf_max",
        "geds/raw/wf_std",
    ]
    assert lh5.ls(lgnd_file, recursive=True) == [
        "geds",
        "geds/raw",
        "geds/raw/baseline",
        "geds/raw/channel",
        "geds/raw/energy",
        "geds/raw/ievt",
        "geds/raw/numtraces",
        "geds/raw/packet_id",
        "geds/raw/timestamp",
        "geds/raw/tracelist",
        "geds/raw/waveform",
        "geds/raw/wf_max",
        "geds/raw/wf_std",
        "geds/raw/tracelist/cumulative_length",
        "geds/raw/tracelist/flattened_data",
        "geds/raw/waveform/dt",
        "geds/raw/waveform/t0",
        "geds/raw/waveform/values",
    ]

    lgnd_file = lgnd_test_data.get_path("lh5/L200-comm-20211130-phy-spms.lh5")
    assert lh5.ls(lgnd_file, "ch5/raw/", recursive=True) == [
        "ch5/raw/abs_delta_mu_usec",
        "ch5/raw/baseline",
        "ch5/raw/channel",
        "ch5/raw/daqenergy",
        "ch5/raw/deadtime",
        "ch5/raw/delta_mu_usec",
        "ch5/raw/dr_maxticks",
        "ch5/raw/dr_start_pps",
        "ch5/raw/dr_start_ticks",
        "ch5/raw/dr_stop_pps",
        "ch5/raw/dr_stop_ticks",
        "ch5/raw/eventnumber",
        "ch5/raw/mu_offset_sec",
        "ch5/raw/mu_offset_usec",
        "ch5/raw/numtraces",
        "ch5/raw/packet_id",
        "ch5/raw/runtime",
        "ch5/raw/timestamp",
        "ch5/raw/to_master_sec",
        "ch5/raw/to_start_sec",
        "ch5/raw/to_start_usec",
        "ch5/raw/tracelist",
        "ch5/raw/ts_maxticks",
        "ch5/raw/ts_pps",
        "ch5/raw/ts_ticks",
        "ch5/raw/waveform",
        "ch5/raw/tracelist/cumulative_length",
        "ch5/raw/tracelist/flattened_data",
        "ch5/raw/waveform/dt",
        "ch5/raw/waveform/t0",
        "ch5/raw/waveform/values",
    ]


def test_show(lgnd_file):
    lh5.show(lgnd_file)
    lh5.show(lgnd_file, "/geds/raw")
    lh5.show(lgnd_file, "geds/raw")
