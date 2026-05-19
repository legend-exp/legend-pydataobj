from __future__ import annotations

import importlib
import sys
from warnings import warn

try:
    import lh5  # noqa: F401
except ModuleNotFoundError as e:
    msg = (
        "lgdo.lh5 has moved to its own package, legend-lh5io. "
        "Please install it (e.g. 'pip install legend-lh5io') and "
        "replace 'import lgdo.lh5' with 'import lh5'."
    )
    raise ModuleNotFoundError(msg) from e

sys.modules[__name__] = sys.modules["lh5"]

# Re-expose old submodule paths (lgdo.lh5.<name> -> lh5.io.<name>) so legacy
# imports like ``import lgdo.lh5.exceptions`` keep working after the split.
for _name in (
    "concat",
    "core",
    "datatype",
    "exceptions",
    "iterator",
    "settings",
    "store",
    "tools",
    "utils",
    "_serializers",
):
    _submodule = importlib.import_module(f"lh5.io.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _submodule
    setattr(sys.modules[__name__], _name, _submodule)
del _name, _submodule

warn(
    "lgdo.lh5 has moved to its own package, legend-lh5io. "
    "Please replace 'import lgdo.lh5' with 'import lh5'. "
    "lgdo.lh5 will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
