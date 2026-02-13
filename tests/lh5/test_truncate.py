
from __future__ import annotations

from pathlib import Path

import numpy as np
import awkward as ak
import pytest

from lgdo import lh5, types, read_as
from lgdo.lh5.truncate import truncate
from lgdo.lh5 import concat


def test_truncate_tcm(lgnd_test_data, tmptestdir):
    # truncate the TCM, which is event-ordered
    infile = lgnd_test_data.get_path(
        "lh5/l200-truncated/generated/tier/tcm/ath/p13/r001/l200-p13-r001-ath-20241210T230220Z-tier_tcm.lh5"
    )
    # ath -> sparse SiPM data
    outfile = f"{tmptestdir}/tcm_trunc.lh5"

    # force evt-style truncation (slicing) even on a raw file
    truncate(infile=infile, outfile=outfile, length_or_slice=5, overwrite=True)

    store = lh5.LH5Store()
    assert lh5.ls(outfile) == [
        "hardware_tcm_1"
    ]
    assert lh5.ls(outfile, "hardware_tcm_1/") == [
        "hardware_tcm_1/row_in_table",
        "hardware_tcm_1/table_key"
    ]

    # read first array-like object and assert its length equals requested truncation
    obj = store.read("hardware_tcm_1/row_in_table", outfile)
    assert len(obj) == 5
    obj = store.read("hardware_tcm_1/table_key", outfile)
    assert len(obj) == 5

    truncate(infile=infile, outfile=outfile, length_or_slice=slice(2,6), overwrite=True)
    obj = store.read("hardware_tcm_1/row_in_table", outfile)
    assert len(obj) == 4
    obj = store.read("hardware_tcm_1/table_key", outfile)
    assert len(obj) == 4

    # no slicing -> stays the same
    table_key_orig = read_as("hardware_tcm_1/table_key", infile, "ak")
    truncate(infile=infile, outfile=outfile, length_or_slice=len(table_key_orig), overwrite=True)
    assert ak.all(read_as("hardware_tcm_1/table_key", outfile, "ak") == table_key_orig)


channellist_sorted = sorted([
        "ch1052803", 
        "ch1052804", 
        "ch1054401", 
        "ch1054402", 
        "ch1054403", 
        "ch1054404", 
        "ch1054405", 
        "ch1056000", 
        "ch1056001", 
        "ch1056002", 
        "ch1056003", 
        "ch1056004", 
        "ch1056005", 
        "ch1057600", 
        "ch1057601", 
        "ch1057602", 
        "ch1057603", 
        "ch1057604", 
        "ch1057605", 
        "ch1059200", 
        "ch1059201", 
        "ch1059202", 
        "ch1059204", 
        "ch1060801", 
        "ch1060802", 
        "ch1060803", 
        "ch1060804", 
        "ch1060805", 
        "ch1062400", 
        "ch1062401", 
        "ch1062402", 
        "ch1062403", 
        "ch1062404", 
        "ch1064000", 
        "ch1064001", 
        "ch1064002", 
        "ch1064003", 
        "ch1064004", 
        "ch1065600", 
        "ch1065601", 
        "ch1065602", 
        "ch1065603", 
        "ch1065604", 
        "ch1065605", 
        "ch1067200", 
        "ch1067201", 
        "ch1067202", 
        "ch1067203", 
        "ch1067204", 
    ])

def test_truncate_hit(lgnd_test_data, tmptestdir):
    hit_file = lgnd_test_data.get_path(
        "lh5/l200-truncated/generated/tier/hit/ath/p13/r001/l200-p13-r001-ath-20241210T230220Z-tier_hit.lh5"
    )
    tcm_file = lgnd_test_data.get_path(
        "lh5/l200-truncated/generated/tier/tcm/ath/p13/r001/l200-p13-r001-ath-20241210T230220Z-tier_tcm.lh5"
    )

    outfile1 = f"{tmptestdir}/hit_trunc1.lh5"
    outfile2 = f"{tmptestdir}/hit_trunc2.lh5"
    outfile_concat = f"{tmptestdir}/hit_trunc_concat.lh5"

    table_key_orig = read_as("hardware_tcm_1/table_key", tcm_file, "ak")
    total_len = len(table_key_orig)
    assert total_len > 25

    #two pieces, which we stitch together again later...
    truncate(infile=hit_file, outfile=outfile1, length_or_slice=25, overwrite=True, tcm_file=tcm_file, include_list=["ch*"])
    truncate(infile=hit_file, outfile=outfile2, length_or_slice=slice(25,total_len), overwrite=True, tcm_file=tcm_file, include_list=["ch*"])

    assert sorted(lh5.ls(outfile1)) == channellist_sorted

    assert sorted(lh5.ls(outfile1, "ch1052803/hit/")) == sorted([
        "ch1052803/hit/energy_in_pe",
        "ch1052803/hit/is_valid_hit",
        "ch1052803/hit/energy_in_pe_dplms",
        "ch1052803/hit/is_valid_hit_dplms",
        "ch1052803/hit/timestamp",
        "ch1052803/hit/trigger_pos",
        "ch1052803/hit/trigger_pos_dplms",
        "ch1052803/hit/has_any_noise"
    ])

    # this will fail if outfile1 or outfile2 are too small, probably because there can be 
    # channels with 0 entries.
    concat.lh5concat(output=outfile_concat, lh5_files=[outfile1, outfile2], overwrite=True)

    assert ak.all(read_as("ch1052803/hit/energy_in_pe", hit_file, "ak") == read_as("ch1052803/hit/energy_in_pe", outfile_concat, "ak"))
    assert ak.all(read_as("ch1052803/hit/timestamp", hit_file, "ak") == read_as("ch1052803/hit/timestamp", outfile_concat, "ak"))
    assert ak.all(read_as("ch1052803/hit/has_any_noise", hit_file, "ak") == read_as("ch1052803/hit/has_any_noise", outfile_concat, "ak"))
    assert ak.all(read_as("ch1057605/hit/energy_in_pe", hit_file, "ak") == read_as("ch1057605/hit/energy_in_pe", outfile_concat, "ak"))
    assert len(read_as("ch1052803/hit/energy_in_pe", hit_file, "ak")) > len(read_as("ch1052803/hit/energy_in_pe", outfile1, "ak"))
    assert len(read_as("ch1052803/hit/energy_in_pe", hit_file, "ak")) >  len(read_as("ch1052803/hit/energy_in_pe", outfile2, "ak"))


def test_truncate_dsp(lgnd_test_data, tmptestdir):
    dsp_file = lgnd_test_data.get_path(
        "lh5/l200-truncated/generated/tier/dsp/ath/p13/r001/l200-p13-r001-ath-20241210T230220Z-tier_dsp.lh5"
    )
    tcm_file = lgnd_test_data.get_path(
        "lh5/l200-truncated/generated/tier/tcm/ath/p13/r001/l200-p13-r001-ath-20241210T230220Z-tier_tcm.lh5"
    )

    outfile1 = f"{tmptestdir}/dsp_trunc1.lh5"
    outfile2 = f"{tmptestdir}/dsp_trunc2.lh5"
    outfile_concat = f"{tmptestdir}/dsp_trunc_concat.lh5"

    table_key_orig = read_as("hardware_tcm_1/table_key", tcm_file, "ak")
    total_len = len(table_key_orig)

    #two pieces, which we stitch together again later...
    truncate(infile=dsp_file, outfile=outfile1, length_or_slice=25, overwrite=True, tcm_file=tcm_file, include_list=["ch*"])
    truncate(infile=dsp_file, outfile=outfile2, length_or_slice=slice(25,total_len), overwrite=True, tcm_file=tcm_file, include_list=["ch*"])

    assert sorted(lh5.ls(outfile1)) == channellist_sorted

    assert sorted(lh5.ls(outfile1, "ch1052803/dsp/")) == sorted([
        "ch1052803/dsp/curr_fwhm",
        "ch1052803/dsp/energy",
        "ch1052803/dsp/energy_dplms",
        "ch1052803/dsp/timestamp",
        "ch1052803/dsp/trigger_pos",
        "ch1052803/dsp/trigger_pos_dplms",
        "ch1052803/dsp/wf_lower_hwhm",
        "ch1052803/dsp/wf_min",
        "ch1052803/dsp/wf_mode"
    ])

    assert len(read_as("ch1052803/dsp/timestamp", dsp_file, "ak")) <= 25

    # this will fail if outfile1 or outfile2 are too small, probably because there can be 
    # channels with 0 entries.
    concat.lh5concat(output=outfile_concat, lh5_files=[outfile1, outfile2], overwrite=True)

    assert ak.all(read_as("ch1052803/dsp/energy", dsp_file, "ak") == read_as("ch1052803/dsp/energy", outfile_concat, "ak"))
    assert ak.all(read_as("ch1052803/dsp/timestamp", dsp_file, "ak") == read_as("ch1052803/dsp/timestamp", outfile_concat, "ak"))
    assert ak.all(read_as("ch1052803/dsp/trigger_pos", dsp_file, "ak") == read_as("ch1052803/dsp/trigger_pos", outfile_concat, "ak"))
    assert ak.all(read_as("ch1052803/dsp/wf_min", dsp_file, "ak") == read_as("ch1052803/dsp/wf_min", outfile_concat, "ak"))
    assert ak.all(read_as("ch1057605/dsp/energy", dsp_file, "ak") == read_as("ch1057605/dsp/energy", outfile_concat, "ak"))

def test_truncate_raw(lgnd_test_data, tmptestdir):
    raw_file = lgnd_test_data.get_path(
        "lh5/l200-truncated/generated/tier/raw/ath/p13/r001/l200-p13-r001-ath-20241210T230220Z-tier_raw.lh5"
    )
    tcm_file = lgnd_test_data.get_path(
        "lh5/l200-truncated/generated/tier/tcm/ath/p13/r001/l200-p13-r001-ath-20241210T230220Z-tier_tcm.lh5"
    )

    outfile1 = f"{tmptestdir}/raw_trunc1.lh5"
    outfile2 = f"{tmptestdir}/raw_trunc2.lh5"
    outfile_concat = f"{tmptestdir}/raw_trunc_concat.lh5"

    table_key_orig = read_as("hardware_tcm_1/table_key", tcm_file, "ak")
    total_len = len(table_key_orig)

    #two pieces, which we stitch together again later...
    truncate(infile=raw_file, outfile=outfile1, length_or_slice=25, overwrite=True, tcm_file=tcm_file, include_list=["ch*"])
    truncate(infile=raw_file, outfile=outfile2, length_or_slice=slice(25,total_len), overwrite=True, tcm_file=tcm_file, include_list=["ch*"])

    assert sorted(lh5.ls(outfile1)) == channellist_sorted

    assert sorted(lh5.ls(outfile1, "ch1052803/raw/")) == sorted([
        "ch1052803/raw/abs_delta_mu_usec",
        "ch1052803/raw/baseline",
        "ch1052803/raw/board_id",
        "ch1052803/raw/channel",
        "ch1052803/raw/daqenergy",
        "ch1052803/raw/deadinterval_nsec",
        "ch1052803/raw/deadtime",
        "ch1052803/raw/delta_mu_usec",
        "ch1052803/raw/dr_ch_idx",
        "ch1052803/raw/dr_ch_len",
        "ch1052803/raw/dr_maxticks",
        "ch1052803/raw/dr_start_pps",
        "ch1052803/raw/dr_start_ticks",
        "ch1052803/raw/dr_stop_pps",
        "ch1052803/raw/dr_stop_ticks",
        "ch1052803/raw/event_type",
        "ch1052803/raw/eventnumber",
        "ch1052803/raw/fc_input",
        "ch1052803/raw/fcid",
        "ch1052803/raw/lifetime",
        "ch1052803/raw/mu_offset_sec",
        "ch1052803/raw/mu_offset_usec",
        "ch1052803/raw/numtraces",
        "ch1052803/raw/packet_id",
        "ch1052803/raw/runtime",
        "ch1052803/raw/timestamp",
        "ch1052803/raw/to_master_sec",
        "ch1052803/raw/to_start_sec",
        "ch1052803/raw/to_start_usec",
        "ch1052803/raw/ts_maxticks",
        "ch1052803/raw/ts_pps",
        "ch1052803/raw/ts_ticks",
        "ch1052803/raw/waveform_bit_drop"
    ])

    # this will fail if outfile1 or outfile2 are too small, probably because there can be 
    # channels with 0 entries.
    concat.lh5concat(output=outfile_concat, lh5_files=[outfile1, outfile2], overwrite=True)

    assert ak.all(read_as("ch1052803/raw/baseline", raw_file, "ak") == read_as("ch1052803/raw/baseline", outfile_concat, "ak"))
    assert ak.all(read_as("ch1052803/raw/timestamp", raw_file, "ak") == read_as("ch1052803/raw/timestamp", outfile_concat, "ak"))
    assert ak.all(read_as("ch1052803/raw/waveform_bit_drop/t0", raw_file, "ak") == read_as("ch1052803/raw/waveform_bit_drop/t0", outfile_concat, "ak"))
    assert ak.all(read_as("ch1052803/raw/waveform_bit_drop/values", raw_file, "ak") == read_as("ch1052803/raw/waveform_bit_drop/values", outfile_concat, "ak"))
    assert ak.all(read_as("ch1057605/raw/waveform_bit_drop/values", raw_file, "ak") == read_as("ch1057605/raw/waveform_bit_drop/values", outfile_concat, "ak"))


