from __future__ import annotations

from lgdo import cli, lh5


def test_lh5ls(lgnd_test_data):
    cli.lh5ls(
        [
            "-a",
            lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
            "geds/raw",
        ]
    )


def test_lh5concat(lgnd_test_data, tmptestdir):
    infile1 = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/raw/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_raw.lh5"
    )
    infile2 = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/raw/phy/p03/r001/l200-p03-r001-phy-20230322T170202Z-tier_raw.lh5"
    )
    outfile = f"{tmptestdir}/out.lh5"

    cli.lh5concat_cli(
        ["--output", outfile, "--debug", "--overwrite", "--", infile1, infile2]
    )

    assert lh5.ls(outfile) == [
        "ch1057600",
        "ch1059201",
        "ch1062405",
        "ch1084803",
        "ch1084804",
        "ch1121600",
    ]
