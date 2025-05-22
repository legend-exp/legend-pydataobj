from __future__ import annotations

from typing import Any


def default_hdf5_settings() -> dict[str, Any]:
    """Returns the HDF5 settings for writing data to disk to the pydataobj defaults.

    Examples
    --------
    >>> from lgdo import lh5
    >>> lh5.DEFAULT_HDF5_SETTINGS["compression"] = "lzf"
    >>> lh5.write(data, "data", "file.lh5")  # compressed with LZF
    >>> lh5.DEFAULT_HDF5_SETTINGS = lh5.default_hdf5_settings()
    >>> lh5.write(data, "data", "file.lh5", "of")  # compressed with default settings (GZIP)
    """

    return {
        "shuffle": True,
        "compression": "gzip",
    }


DEFAULT_HDF5_SETTINGS: dict[str, ...] = default_hdf5_settings()
"""Global dictionary storing the default HDF5 settings for writing data to disk.

Modify this global variable before writing data to disk with this package.

Examples
--------
>>> from lgdo import lh5
>>> lh5.DEFAULT_HDF5_SETTINGS["compression"] = "lzf"
>>> lh5.write(data, "data", "file.lh5")  # compressed with LZF
"""
