from __future__ import annotations

import pytest

from lgdo import lh5


@pytest.fixture()
def wftable(lgnd_test_data):
    store = lh5.LH5Store()
    wft, _ = store.read(
        "/geds/raw/waveform",
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
    )
    return wft
