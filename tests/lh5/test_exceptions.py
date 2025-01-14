from __future__ import annotations

import pickle

from lgdo.lh5.exceptions import LH5DecodeError, LH5EncodeError


def test_pickle():
    # test (un-)pickling of LH5 exceptions; e.g. for multiprocessing use.

    ex = LH5EncodeError("message", "file", "group", "name")
    ex = pickle.loads(pickle.dumps(ex))
    assert isinstance(ex, LH5EncodeError)

    ex = LH5DecodeError("message", "file", "obj")
    ex = pickle.loads(pickle.dumps(ex))
    assert isinstance(ex, LH5DecodeError)
