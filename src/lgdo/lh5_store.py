from __future__ import annotations

import sys
from typing import Iterator, Union
from warnings import warn

import h5py
import numpy as np
import pandas as pd

from . import lh5
from .types import Array  # noqa: F401
from .types import ArrayOfEncodedEqualSizedArrays  # noqa: F401
from .types import ArrayOfEqualSizedArrays  # noqa: F401
from .types import FixedSizeArray  # noqa: F401
from .types import Scalar  # noqa: F401
from .types import Struct  # noqa: F401
from .types import Table  # noqa: F401
from .types import VectorOfEncodedVectors  # noqa: F401
from .types import VectorOfVectors  # noqa: F401
from .types import WaveformTable  # noqa: F401

DEFAULT_HDF5_COMPRESSION = None
LGDO = Union[Array, Scalar, Struct, VectorOfVectors]
DEFAULT_HDF5_SETTINGS: dict[str, ...] = {"shuffle": True, "compression": "gzip"}


class LH5Iterator(lh5.LH5Iterator):
    def __init__(
        self,
        lh5_files: str | list[str],
        groups: str | list[str],
        base_path: str = "",
        entry_list: list[int] | list[list[int]] = None,
        entry_mask: list[bool] | list[list[bool]] = None,
        field_mask: dict[str, bool] | list[str] | tuple[str] = None,
        buffer_len: int = 3200,
        friend: Iterator = None,
    ) -> None:
        warn(
            "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Store andLH5Iterator."
            "We recommend using 'import lgdo.lh5 as lh5' and then creating anLH5Iterator instance via 'lh5.LH5Iterator()'."
            "lgdo.lh5_store will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            lh5_files,
            groups,
            base_path,
            entry_list,
            entry_mask,
            field_mask,
            buffer_len,
            friend,
        )

    def write_object(
        self,
        obj: LGDO,
        name: str,
        lh5_file: str | h5py.File,
        group: str | h5py.Group = "/",
        start_row: int = 0,
        n_rows: int = None,
        wo_mode: str = "append",
        write_start: int = 0,
        **h5py_kwargs,
    ) -> None:
        warn(
            "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Store andLH5Iterator. "
            "The object you are calling this function from uses the old LH5Iterator class."
            "We recommend using 'import lgdo.lh5 as lh5' and then creating an instance via 'lh5.LH5Iterator()'.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write(
            obj,
            name,
            lh5_file,
            group,
            start_row,
            n_rows,
            wo_mode,
            write_start,
            h5py_kwargs,
        )

    def read_object(
        self,
        name: str,
        lh5_file: str | h5py.File | list[str | h5py.File],
        start_row: int = 0,
        n_rows: int = sys.maxsize,
        idx: np.ndarray | list | tuple | list[np.ndarray | list | tuple] = None,
        field_mask: dict[str, bool] | list[str] | tuple[str] = None,
        obj_buf: LGDO = None,
        obj_buf_start: int = 0,
        decompress: bool = True,
    ) -> tuple[LGDO, int]:
        warn(
            "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Store andLH5Iterator. "
            "The object you are calling this function from uses the old LH5Iterator class."
            "We recommend using 'import lgdo.lh5 as lh5' and then creating a LH5Store instance via 'lh5.LH5Store()'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.read(
            name,
            lh5_file,
            start_row,
            n_rows,
            idx,
            field_mask,
            obj_buf,
            obj_buf_start,
            decompress,
        )


class LH5Store(lh5.LH5Store):
    def __init__(self, base_path: str = "", keep_open: bool = False):
        warn(
            "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Store andLH5Iterator. "
            "We recommend using 'import lgdo.lh5 as lh5' and then creating an instance via 'lh5.LH5Iterator()'.",
            "lgdo.lh5_store will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(base_path, keep_open)


def load_dfs(
    f_list: str | list[str],
    par_list: list[str],
    lh5_group: str = "",
    idx_list: list[np.ndarray | list | tuple] = None,
) -> pd.DataFrame:
    warn(
        "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Store andLH5Iterator. "
        "Please replace 'import lgdo.lh5_store' with 'import lgdo.lh5'. "
        "lgdo.lh5_store will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return lh5.load_dfs(f_list, par_list, lh5_group, idx_list)


def load_nda(
    f_list: str | list[str],
    par_list: list[str],
    lh5_group: str = "",
    idx_list: list[np.ndarray | list | tuple] = None,
) -> dict[str, np.ndarray]:
    warn(
        "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Store andLH5Iterator. "
        "Please replace 'import lgdo.lh5_store' with 'import lgdo.lh5'. "
        "lgdo.lh5_store will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return lh5.load_nda(f_list, par_list, lh5_group, idx_list)


def ls(lh5_file: str | h5py.Group, lh5_group: str = "") -> list[str]:
    warn(
        "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Store andLH5Iterator. "
        "Please replace 'import lgdo.lh5_store' with 'import lgdo.lh5'. "
        "lgdo.lh5_store will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return lh5.ls(lh5_file, lh5_group)


def show(
    lh5_file: str | h5py.Group,
    lh5_group: str = "/",
    attrs: bool = False,
    indent: str = "",
    header: bool = True,
) -> None:
    warn(
        "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Store andLH5Iterator. "
        "Please replace 'import lgdo.lh5_store' with 'import lgdo.lh5'. "
        "lgdo.lh5_store will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    lh5.show(lh5_file, lh5_group, attrs, indent, header)
