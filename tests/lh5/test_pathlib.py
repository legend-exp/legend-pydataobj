from __future__ import annotations

from pathlib import Path

import numpy as np

from lgdo import lh5, types
from lgdo.lh5 import utils


def test_expand_path_with_path_objects(tmp_path):
    f = tmp_path / "file.lh5"
    f.touch()
    # direct Path
    assert utils.expand_path(f) == str(f)
    # Path with base_path as Path
    assert utils.expand_path(f.name, base_path=tmp_path) == f.name
    assert utils.expand_path(Path(f.name), base_path=Path(tmp_path)) == f.name
    # wildcard with Path
    assert utils.expand_path(tmp_path / "*.lh5", list=True) == [str(f)]


def test_read_write_with_path_objects(tmp_path):
    arr = types.Array(np.arange(5))
    out = tmp_path / "out.lh5"
    lh5.write(arr, "arr", out, group="/data", wo_mode="overwrite_file")
    result = lh5.read("/data/arr", out)
    assert np.array_equal(result.nda, np.arange(5))


def test_store_with_path_objects(lh5_file, tmp_path):
    store = lh5.LH5Store(base_path=tmp_path)
    path_obj = Path(lh5_file)
    obj = store.read("/data/struct/scalar", path_obj)
    assert obj.value == 10
    out = tmp_path / "out_store.lh5"
    arr = types.Array(np.arange(3))
    store.write(arr, "arr", out, group="/data", wo_mode="overwrite_file")
    obj2 = store.read("/data/arr", out)
    assert np.array_equal(obj2.nda, np.arange(3))
