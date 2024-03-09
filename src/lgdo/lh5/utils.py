"""Implements utilities for LEGEND Data Objects."""
from __future__ import annotations

import glob
import logging
import os
import string

log = logging.getLogger(__name__)


def parse_datatype(datatype: str) -> tuple[str, tuple[int, ...], str | list[str]]:
    """Parse datatype string and return type, dimensions and elements.

    Parameters
    ----------
    datatype
        a LGDO-formatted datatype string.

    Returns
    -------
    element_type
        the datatype name dims if not ``None``, a tuple of dimensions for the
        LGDO. Note this is not the same as the NumPy shape of the underlying
        data object. See the LGDO specification for more information. Also see
        :class:`~.types.ArrayOfEqualSizedArrays` and
        :meth:`.lh5_store.LH5Store.read` for example code elements for
        numeric objects, the element type for struct-like  objects, the list of
        fields in the struct.
    """
    if "{" not in datatype:
        return "scalar", None, datatype

    # for other datatypes, need to parse the datatype string
    from parse import parse

    datatype, element_description = parse("{}{{{}}}", datatype)
    if datatype.endswith(">"):
        datatype, dims = parse("{}<{}>", datatype)
        dims = [int(i) for i in dims.split(",")]
        return datatype, tuple(dims), element_description

    return datatype, None, element_description.split(",")


def expand_vars(expr: str, substitute: dict[str, str] | None = None) -> str:
    """Expand (environment) variables.

    Note
    ----
    Malformed variable names and references to non-existing variables are left
    unchanged.

    Parameters
    ----------
    expr
        string expression, which may include (environment) variables prefixed by
        ``$``.
    substitute
        use this dictionary to substitute variables. Takes precedence over
        environment variables.
    """
    if substitute is None:
        substitute = {}

    # use provided mapping
    # then expand env variables
    return os.path.expandvars(string.Template(expr).safe_substitute(substitute))


def expand_path(
    path: str,
    substitute: dict[str, str] | None = None,
    list: bool = False,
    base_path: str | None = None,
) -> str | list:
    """Expand (environment) variables and wildcards to return absolute paths.

    Parameters
    ----------
    path
        name of path, which may include environment variables and wildcards.
    list
        if ``True``, return a list. If ``False``, return a string; if ``False``
        and a unique file is not found, raise an exception.
    substitute
        use this dictionary to substitute variables. Environment variables take
        precedence.
    base_path
        name of base path. Returned paths will be relative to base.

    Returns
    -------
    path or list of paths
        Unique absolute path, or list of all absolute paths
    """
    if base_path is not None and base_path != "":
        base_path = os.path.expanduser(os.path.expandvars(base_path))
        path = os.path.join(base_path, path)

    # first expand variables
    _path = expand_vars(path, substitute)

    # then expand wildcards
    paths = sorted(glob.glob(os.path.expanduser(_path)))

    if base_path is not None and base_path != "":
        paths = [os.path.relpath(p, base_path) for p in paths]

    if not list:
        if len(paths) == 0:
            msg = f"could not find path matching {path}"
            raise FileNotFoundError(msg)
        if len(paths) > 1:
            msg = f"found multiple paths matching {path}"
            raise FileNotFoundError(msg)

        return paths[0]

    return paths
