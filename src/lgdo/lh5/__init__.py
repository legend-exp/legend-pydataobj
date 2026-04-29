from __future__ import annotations

from warnings import warn

try:
    from lh5 import *  # noqa: F403
except ModuleNotFoundError as e:
    msg = (
        "lgdo.lh5 has moved to its own package, legend-lh5io. "
        "Please install it (e.g. 'pip install legend-lh5io') and "
        "replace 'import lgdo.lh5' with 'import lh5'."
    )
    raise ModuleNotFoundError(msg) from e

warn(
    "lgdo.lh5 has moved to its own package, legend-lh5io. "
    "Please replace 'import lgdo.lh5' with 'import lh5'. "
    "lgdo.lh5 will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
