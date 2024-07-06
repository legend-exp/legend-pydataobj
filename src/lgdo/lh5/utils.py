"""Implements utilities for LEGEND Data Objects."""

from __future__ import annotations

import glob
import logging
import os
import string
from collections.abc import Mapping, Sequence
from typing import Any

import h5py

from .. import types
from . import _serializers, datatype
from .exceptions import LH5DecodeError

log = logging.getLogger(__name__)


def get_buffer(
    name: str,
    lh5_file: str | h5py.File | Sequence[str | h5py.File],
    size: int | None = None,
    field_mask: Mapping[str, bool] | Sequence[str] | None = None,
) -> types.LGDO:
    """Returns an LGDO appropriate for use as a pre-allocated buffer.

    Sets size to `size` if object has a size.
    """
    obj, n_rows = _serializers._h5_read_lgdo(
        lh5_file[name], n_rows=0, field_mask=field_mask
    )

    if hasattr(obj, "resize") and size is not None:
        obj.resize(new_size=size)

    return obj


def read_n_rows(name: str, h5f: str | h5py.File) -> int | None:
    """Look up the number of rows in an Array-like LGDO object on disk.

    Return ``None`` if `name` is a :class:`.Scalar` or a :class:`.Struct`.
    """
    if not isinstance(h5f, h5py.File):
        h5f = h5py.File(h5f, "r")

    try:
        attrs = h5f[name].attrs
    except KeyError as e:
        msg = "not found"
        raise LH5DecodeError(msg, h5f, name) from e
    except AttributeError as e:
        msg = "missing 'datatype' attribute"
        raise LH5DecodeError(msg, h5f, name) from e

    lgdotype = datatype.datatype(attrs["datatype"])

    # scalars are dim-0 datasets
    if lgdotype is types.Scalar:
        return None

    # structs don't have rows
    if lgdotype is types.Struct:
        return None

    # tables should have elements with all the same length
    if lgdotype is types.Table:
        # read out each of the fields
        rows_read = None
        for field in datatype.get_struct_fields(attrs["datatype"]):
            n_rows_read = read_n_rows(name + "/" + field, h5f)
            if not rows_read:
                rows_read = n_rows_read
            elif rows_read != n_rows_read:
                log.warning(
                    f"'{field}' field in table '{name}' has {rows_read} rows, "
                    f"{n_rows_read} was expected"
                )
        return rows_read

    # length of vector of vectors is the length of its cumulative_length
    if lgdotype is types.VectorOfVectors:
        return read_n_rows(f"{name}/cumulative_length", h5f)

    # length of vector of encoded vectors is the length of its decoded_size
    if lgdotype in (types.VectorOfEncodedVectors, types.ArrayOfEncodedEqualSizedArrays):
        return read_n_rows(f"{name}/encoded_data", h5f)

    # return array length (without reading the array!)
    if issubclass(lgdotype, types.Array):
        # compute the number of rows to read
        return h5f[name].shape[0]

    msg = f"don't know how to read rows of LGDO {lgdotype.__name__}"
    raise LH5DecodeError(msg, h5f, name)


def get_h5_group(
    group: str | h5py.Group,
    base_group: h5py.Group,
    grp_attrs: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> h5py.Group:
    """
    Returns an existing :mod:`h5py` group from a base group or creates a
    new one. Can also set (or replace) group attributes.

    Parameters
    ----------
    group
        name of the HDF5 group.
    base_group
        HDF5 group to be used as a base.
    grp_attrs
        HDF5 group attributes.
    overwrite
        whether overwrite group attributes, ignored if `grp_attrs` is
        ``None``.
    """
    if not isinstance(group, h5py.Group):
        if group in base_group:
            group = base_group[group]
        else:
            group = base_group.create_group(group)
            if grp_attrs is not None:
                group.attrs.update(grp_attrs)
            return group
    if (
        grp_attrs is not None
        and len(set(grp_attrs.items()) ^ set(group.attrs.items())) > 0
    ):
        if not overwrite:
            msg = (
                f"Provided {grp_attrs=} are different from "
                f"existing ones {dict(group.attrs)=} but overwrite flag is not set"
            )
            raise RuntimeError(msg)

        log.debug(f"overwriting {group}.attrs...")
        for key in group.attrs:
            group.attrs.pop(key)
        group.attrs.update(grp_attrs)

    return group


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


# https://stackoverflow.com/a/1094933
def fmtbytes(num, suffix="B"):
    """Returns formatted f-string for printing human-readable number of bytes."""
    for unit in ("", "k", "M", "G", "T", "P", "E", "Z"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Y{suffix}"
