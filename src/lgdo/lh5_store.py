"""
.. warning::
    This subpackage is deprecated, use :mod:`lgdo.lh5`.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Union
from warnings import warn

import h5py
import numpy as np
import pandas as pd

from . import lh5
from .types import (
    Array,
    ArrayOfEncodedEqualSizedArrays,  # noqa: F401
    ArrayOfEqualSizedArrays,  # noqa: F401
    FixedSizeArray,  # noqa: F401
    Scalar,
    Struct,
    Table,  # noqa: F401
    VectorOfEncodedVectors,  # noqa: F401
    VectorOfVectors,
    WaveformTable,  # noqa: F401
)

LGDO = Union[Array, Scalar, Struct, VectorOfVectors]


class LH5Iterator(lh5.LH5Iterator):
    """
    .. warning::
        This class is deprecated, use :class:`lgdo.lh5.iterator.LH5Iterator`.

    """

    def __init__(
        self,
        lh5_files: str | list[str],
        groups: str | list[str],
        base_path: str = "",
        entry_list: list[int] | list[list[int]] | None = None,
        entry_mask: list[bool] | list[list[bool]] | None = None,
        field_mask: dict[str, bool] | list[str] | tuple[str] | None = None,
        buffer_len: int = 3200,
        friend: Iterator | None = None,
    ) -> None:
        warn(
            "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Iterator."
            "Please replace 'from lgdo.lh5_store import LH5Iterator' with 'from lgdo.lh5 import LH5Iterator'."
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
        n_rows: int | None = None,
        wo_mode: str = "append",
        write_start: int = 0,
        **h5py_kwargs,
    ) -> None:
        """
        .. warning::
            This method is deprecated, use :meth:`lgdo.lh5.iterator.LH5Iterator.write`.

        """
        warn(
            "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Iterator. "
            "The object you are calling this function from uses the old LH5Iterator class."
            "Please replace 'from lgdo.lh5_store import LH5Iterator' with 'from lgdo.lh5 import LH5Iterator'."
            "lgdo.lh5_store will be removed in a future release.",
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
        field_mask: dict[str, bool] | list[str] | tuple[str] | None = None,
        obj_buf: LGDO = None,
        obj_buf_start: int = 0,
        decompress: bool = True,
    ) -> tuple[LGDO, int]:
        """
        .. warning::
            This method is deprecated, use :meth:`lgdo.lh5.iterator.LH5Iterator.read`.

        """
        warn(
            "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Iterator. "
            "The object you are calling this function from uses the old LH5Iterator class."
            "Please replace 'from lgdo.lh5_store import LH5Iterator' with 'from lgdo.lh5 import LH5Iterator'."
            "lgdo.lh5_store will be removed in a future release.",
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
    """
    .. warning::
        This class is deprecated, use :class:`lgdo.lh5.iterator.LH5Store`.

    """

    def __init__(self, base_path: str = "", keep_open: bool = False):
        warn(
            "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Store. "
            "Please replace 'from lgdo.lh5_store import LH5Store' with 'from lgdo.lh5 import LH5Store'."
            "lgdo.lh5_store will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(base_path, keep_open)

    def read_object(
        self,
        name: str,
        lh5_file: str | h5py.File | list[str | h5py.File],
        **kwargs,
    ) -> tuple[LGDO, int]:
        """
        .. warning::
            This method is deprecated, use :meth:`lgdo.lh5.store.LH5Store.read`.

        """
        warn(
            "LH5Store.read_object() has been renamed to LH5Store.read(), "
            "Please update your code."
            "LH5Store.read_object() will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().read(self, name, lh5_file, **kwargs)

    def write_object(
        self,
        obj: LGDO,
        name: str,
        lh5_file: str | h5py.File,
        **kwargs,
    ) -> tuple[LGDO, int]:
        """
        .. warning::
            This method is deprecated, use :meth:`lgdo.lh5.store.LH5Store.write`.

        """
        warn(
            "LH5Store.write_object() has been renamed to LH5Store.write(), "
            "Please update your code."
            "LH5Store.write_object() will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().read(self, obj, name, lh5_file, **kwargs)


def load_dfs(
    f_list: str | list[str],
    par_list: list[str],
    lh5_group: str = "",
    idx_list: list[np.ndarray | list | tuple] | None = None,
) -> pd.DataFrame:
    """
    .. warning::
        This function is deprecated, use :meth:`lgdo.types.lgdo.LGDO.view_as` to
        view LGDO data as a Pandas data structure.

    """
    warn(
        "lgdo.lh5_store has moved to a subfolder lgdo.lh5. "
        "Please replace 'from lgdo.lh5_store import load_dfs' with 'from lgdo.lh5 import load_dfs'. "
        "lgdo.lh5_store will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return lh5.load_dfs(f_list, par_list, lh5_group, idx_list)


def load_nda(
    f_list: str | list[str],
    par_list: list[str],
    lh5_group: str = "",
    idx_list: list[np.ndarray | list | tuple] | None = None,
) -> dict[str, np.ndarray]:
    """
    .. warning::
        This function is deprecated, use :meth:`lgdo.types.lgdo.LGDO.view_as` to
        view LGDO data as a NumPy data structure.

    """
    warn(
        "lgdo.lh5_store has moved to a subfolder lgdo.lh5. "
        "Please replace 'from lgdo.lh5_store import load_nda' with 'from lgdo.lh5 import load_nda'. "
        "lgdo.lh5_store will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return lh5.load_nda(f_list, par_list, lh5_group, idx_list)


def ls(lh5_file: str | h5py.Group, lh5_group: str = "") -> list[str]:
    """
    .. warning::
        This function is deprecated, import :func:`lgdo.lh5.tools.ls`.

    """
    warn(
        "lgdo.lh5_store has moved to a subfolder lgdo.lh5. "
        "Please replace 'from lgdo.lh5_store import ls' with 'from lgdo.lh5 import ls'. "
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
    """
    .. warning::
        This function is deprecated, import :func:`lgdo.lh5.tools.show`.

    """
    warn(
        "lgdo.lh5_store has moved to a subfolder lgdo.lh5. "
        "Please replace 'from lgdo.lh5_store import show' with 'from lgdo.lh5 import show'. "
        "lgdo.lh5_store will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    lh5.show(lh5_file, lh5_group, attrs, indent, header)
