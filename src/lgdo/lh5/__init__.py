from __future__ import annotations

from warnings import warn

from lh5 import *  # noqa: F403

warn(
    "lgdo.lh5 has moved to its own package, legend-lh5io. "
    "Please replace 'import lgdo.lh5' with 'import lh5'. "
    "lgdo.lh5 will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
