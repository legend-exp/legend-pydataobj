from __future__ import annotations

import os

import pytest

from lgdo.lh5 import utils


@pytest.fixture(scope="module")
def lgnd_file(lgnd_test_data):
    return lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5")


def test_parse_datatype():
    datatypes = [
        ("real", ("scalar", None, "real")),
        ("array<1>{bool}", ("array", (1,), "bool")),
        ("fixedsizearray<2>{real}", ("fixedsizearray", (2,), "real")),
        (
            "arrayofequalsizedarrays<3,4>{complex}",
            ("arrayofequalsizedarrays", (3, 4), "complex"),
        ),
        ("array<1>{array<1>{blob}}", ("array", (1,), "array<1>{blob}")),
        (
            "struct{field1,field2,fieldn}",
            ("struct", None, ["field1", "field2", "fieldn"]),
        ),
        ("table{col1,col2,coln}", ("table", None, ["col1", "col2", "coln"])),
    ]

    for string, dt_tuple in datatypes:
        pd_dt_tuple = utils.parse_datatype(string)
        assert pd_dt_tuple == dt_tuple


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
    base_dir = os.path.dirname(files[0])

    assert utils.expand_path(f"{base_dir}/*20230318T012144Z*") == files[0]

    # Should fail if file not found
    with pytest.raises(FileNotFoundError):
        utils.expand_path(f"{base_dir}/not_a_real_file.lh5")

    # Should fail if multiple files found
    with pytest.raises(FileNotFoundError):
        utils.expand_path(f"{base_dir}/*.lh5")

    # Check if it finds a list of files correctly
    assert sorted(utils.expand_path(f"{base_dir}/*.lh5", list=True)) == sorted(files)
