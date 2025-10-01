from __future__ import annotations

import shutil
import uuid
from getpass import getuser
from pathlib import Path
from tempfile import gettempdir

import pytest
from legendtestdata import LegendTestData

_tmptestdir = Path(gettempdir()) / f"lgdo-tests-{getuser()}-{uuid.uuid4()!s}"


@pytest.fixture(scope="session")
def tmptestdir():
    Path(_tmptestdir).mkdir(parents=True, exist_ok=True)
    return _tmptestdir


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    if exitstatus == 0:
        shutil.rmtree(_tmptestdir, ignore_errors=True)


@pytest.fixture(scope="session")
def lgnd_test_data():
    ldata = LegendTestData()
    ldata.checkout("df10dde")
    return ldata
