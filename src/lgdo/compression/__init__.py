from __future__ import annotations

from warnings import warn

from lh5 import *  # noqa: F403

warn(
    "lgdo.compression has moved to its own package, legend-lh5io. "
    "Please replace 'import lgdo.compression' with 'import lh5.compression'. "
    "lgdo.compression will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
