import os
import shutil
import uuid
from getpass import getuser
from tempfile import gettempdir

import pytest
from legendtestdata import LegendTestData

_tmptestdir = os.path.join(gettempdir(), f"lgdo-tests-{getuser()}-{str(uuid.uuid4())}")


@pytest.fixture(scope="session")
def tmptestdir():
    os.mkdir(_tmptestdir)
    return _tmptestdir


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 0:
        shutil.rmtree(_tmptestdir, ignore_errors=True)


@pytest.fixture(scope="session")
def lgnd_test_data():
    ldata = LegendTestData()
    ldata.checkout("c089a59")
    return ldata
