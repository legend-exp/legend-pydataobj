from __future__ import annotations

from warnings import warn

import numpy as np

from . import types as lgdo
from .lh5 import utils


def copy(obj: lgdo.LGDO, dtype: np.dtype = None) -> None:
    warn(
        "lgdo_utils.copy will soon be removed and will be replaced soon with copy member functions of each LGDO data type.",
        DeprecationWarning,
        stacklevel=2,
    )
    return utils.copy(obj, dtype)


def get_element_type(obj: object) -> str:
    warn(
        "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Store and LH5Iterator. "
        "Please replace 'import lgdo.lh5_store' with 'import lgdo.lh5'. "
        "lgdo.lh5_store will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return utils.get_element_type(obj)


def parse_datatype(datatype: str) -> tuple[str, tuple[int, ...], str | list[str]]:
    warn(
        "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Store and LH5Iterator. "
        "Please replace 'import lgdo.lh5_store' with 'import lgdo.lh5'. "
        "lgdo.lh5_store will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return utils.parse_datatype(datatype)


def expand_vars(expr: str, substitute: dict[str, str] = None) -> str:
    warn(
        "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Store and LH5Iterator. "
        "Please replace 'import lgdo.lh5_store' with 'import lgdo.lh5'. "
        "lgdo.lh5_store will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return utils.expand_vars(expr, substitute)


def expand_path(
    path: str,
    substitute: dict[str, str] = None,
    list: bool = False,
    base_path: str = None,
) -> str | list:
    warn(
        "lgdo.lh5_store has moved to a subfolder lgdo.lh5 containing LH5Store and LH5Iterator. "
        "Please replace 'import lgdo.lh5_store' with 'import lgdo.lh5'. "
        "lgdo.lh5_store will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return utils.expand_path(path, substitute, list, base_path)
