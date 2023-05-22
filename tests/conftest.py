import copy
import inspect
import os
import re
import shutil
import uuid
from getpass import getuser
from tempfile import gettempdir

import pytest
from legendtestdata import LegendTestData

_tmptestdir = os.path.join(
    gettempdir(), "pygama-tests-" + getuser() + str(uuid.uuid4())
)


@pytest.fixture(scope="session")
def tmptestdir():
    os.mkdir(_tmptestdir)
    yield _tmptestdir
    shutil.rmtree(_tmptestdir)


def pytest_sessionfinish(session, exitstatus):
    if exitstatus is True:
        os.rmdir(_tmptestdir)


@pytest.fixture(scope="session")
def lgnd_test_data():
    ldata = LegendTestData()
    ldata.checkout("c089a59")
    return ldata


# @pytest.fixture(scope="session")
# def multich_raw_file(lgnd_test_data):
#     out_file = "/tmp/L200-comm-20211130-phy-spms.lh5"
#     out_spec = {
#         "FCEventDecoder": {
#             "ch{key}": {
#                 "key_list": [[0, 6]],
#                 "out_stream": out_file + ":{name}",
#                 "out_name": "raw",
#             }
#         }
#     }

#     build_raw(
#         in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
#         out_spec=out_spec,
#         overwrite=True,
#     )
#     assert os.path.exists(out_file)

#     return out_file
