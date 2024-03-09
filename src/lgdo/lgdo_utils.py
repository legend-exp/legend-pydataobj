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
        "'lgdo.lgdo_utils' has been renamed to 'lgdo.utils'. "
        "Please replace either 'import lgdo.lgdo_utils as utils' with 'import lgdo.utils as utils' "
        "or 'from lgdo.lgdo_utils import get_element_type' with 'from lgdo.utils import get_element_type'."
        "'lgdo.lgdo_utils' will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return utils.get_element_type(obj)


def parse_datatype(datatype: str) -> tuple[str, tuple[int, ...], str | list[str]]:
    warn(
        "'lgdo.lgdo_utils' has been renamed to 'lgdo.utils'. "
        "Please replace either 'import lgdo.lgdo_utils as utils' with 'import lgdo.utils as utils' "
        "or 'from lgdo.lgdo_utils import parse_datatype' with 'from lgdo.utils import parse_datatype'."
        "'lgdo.lgdo_utils' will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return utils.parse_datatype(datatype)


def expand_vars(expr: str, substitute: dict[str, str] | None = None) -> str:
    warn(
        "'lgdo.lgdo_utils' has been renamed to 'lgdo.utils'. "
        "Please replace either 'import lgdo.lgdo_utils as utils' with 'import lgdo.utils as utils' "
        "or 'from lgdo.lgdo_utils import expand_vars' with 'from lgdo.utils import expand_vars'."
        "'lgdo.lgdo_utils' will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return utils.expand_vars(expr, substitute)


def expand_path(
    path: str,
    substitute: dict[str, str] | None = None,
    list: bool = False,
    base_path: str | None = None,
) -> str | list:
    warn(
        "'lgdo.lgdo_utils' has been renamed to 'lgdo.utils'. "
        "Please replace either 'import lgdo.lgdo_utils as utils' with 'import lgdo.utils as utils' "
        "or 'from lgdo.lgdo_utils import expand_path' with 'from lgdo.utils import expand_path'."
        "'lgdo.lgdo_utils' will be removed in a future release. ",
        DeprecationWarning,
        stacklevel=2,
    )
    return utils.expand_path(path, substitute, list, base_path)
