from __future__ import annotations

import os
from pathlib import Path

import pytest

from lgdo.lh5 import utils


@pytest.fixture(scope="module")
def lgnd_file(lgnd_test_data):
    return lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5")


def test_expand_vars():
    # Check env variable expansion
    os.environ["PYGAMATESTBASEDIR"] = "a_random_string"
    assert utils.expand_vars("$PYGAMATESTBASEDIR/blah") == "a_random_string/blah"

    # Check user variable expansion
    assert (
        utils.expand_vars(
            "$PYGAMATESTBASEDIR2/blah",
            substitute={"PYGAMATESTBASEDIR2": "a_random_string"},
        )
        == "a_random_string/blah"
    )


def test_expand_path(lgnd_test_data):
    files = [
        lgnd_test_data.get_path(
            "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_dsp.lh5"
        ),
        lgnd_test_data.get_path(
            "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r001/l200-p03-r001-cal-20230318T012228Z-tier_dsp.lh5"
        ),
    ]
    base_dir = Path(files[0]).parent

    assert utils.expand_path(f"{base_dir}/*20230318T012144Z*") == files[0]

    # Should fail if file not found
    with pytest.raises(FileNotFoundError):
        utils.expand_path(f"{base_dir}/not_a_real_file.lh5")

    # Should fail if multiple files found
    with pytest.raises(FileNotFoundError):
        utils.expand_path(f"{base_dir}/*.lh5")

    # Check if it finds a list of files correctly
    assert sorted(utils.expand_path(f"{base_dir}/*.lh5", list=True)) == sorted(files)

    # check with base_path specified
    base_path = base_dir.parent
    assert (
        utils.expand_path(f"{base_dir.name}/*20230318T012144Z*", base_path=base_path)
        == Path(files[0]).relative_to(base_path).as_posix()
    )
