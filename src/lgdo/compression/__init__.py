from __future__ import annotations

import importlib
import sys
from warnings import warn

try:
    from lh5.compression import *  # noqa: F403
except ModuleNotFoundError as e:
    msg = (
        "lgdo.compression has moved to its own package, legend-lh5io. "
        "Please install it (e.g. 'pip install legend-lh5io') and "
        "replace 'import lgdo.compression' with 'import lh5.compression'."
    )
    raise ModuleNotFoundError(msg) from e

# Re-expose old submodule paths (lgdo.compression.<name> -> lh5.compression.<name>)
# so legacy imports like ``from lgdo.compression.base import WaveformCodec`` keep
# working after the split.
for _name in ("base", "generic", "radware", "utils", "varlen"):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(
        f"lh5.compression.{_name}"
    )
del _name

warn(
    "lgdo.compression has moved to its own package, legend-lh5io. "
    "Please replace 'import lgdo.compression' with 'import lh5.compression'. "
    "lgdo.compression will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
