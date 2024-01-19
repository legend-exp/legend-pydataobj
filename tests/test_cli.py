from __future__ import annotations

import subprocess


def test_cli(lgnd_test_data):
    subprocess.check_call(["lh5ls", "--help"])
    subprocess.check_call(
        [
            "lh5ls",
            "-a",
            lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
            "geds/raw",
        ]
    )
