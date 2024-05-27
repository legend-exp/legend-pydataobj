"""Implements utilities for LEGEND Data Objects."""

from __future__ import annotations

from functools import reduce
import glob
import json
import logging
import operator
import os
import string
import sys
from collections.abc import Mapping, Sequence
from typing import Any

# https://stackoverflow.com/a/30316760
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType

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
        name, lh5_file, n_rows=0, field_mask=field_mask
    )

    if hasattr(obj, "resize") and size is not None:
        obj.resize(new_size=size)

    return obj


def read_n_rows(
    name: str, 
    h5f: str | h5py.File,
    metadata: dict = None,
) -> int | None:
    """Look up the number of rows in an Array-like LGDO object on disk.

    Return ``None`` if `name` is a :class:`.Scalar` or a :class:`.Struct`.
    """
    if not isinstance(h5f, h5py.File):
        h5f = h5py.File(h5f, "r")

    # this needs to be done for the requested object
    if metadata is not None:
        try:
            attrs = metadata['attrs']
            lgdotype = datatype.datatype(attrs["datatype"])
            log.debug(
                f"{name}.attrs.datatype found in metadata"
            )
        except KeyError as e:
            log.debug(
                f"metadata key error in {h5f.filename}: {e} - will attempt to use file directly instead"
            )
            metadata = None
    
    if metadata is None:
        try:
            attrs = h5f[name].attrs
        except KeyError as e:
            msg = "not found in file"
            raise LH5DecodeError(msg, h5f, name) from e
        except AttributeError as e:
            msg = "missing 'datatype' attribute in file"
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
            n_rows_read = read_n_rows(name + "/" + field, h5f, 
                                      metadata=metadata[field] if metadata is not None else None)
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
        return read_n_rows(f"{name}/cumulative_length", h5f, 
                           metadata=metadata["cumulative_length"] if metadata is not None else None)

    # length of vector of encoded vectors is the length of its decoded_size
    if lgdotype in (types.VectorOfEncodedVectors, types.ArrayOfEncodedEqualSizedArrays):
        return read_n_rows(f"{name}/encoded_data", h5f, 
                           metadata=metadata["encoded_data"] if metadata is not None else None)

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

# https://stackoverflow.com/a/14692747
def getFromDict(dataDict, mapList):
    if not mapList:
        return dataDict
    return reduce(operator.getitem, mapList, dataDict)

# https://stackoverflow.com/a/30316760
def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size

# function is recursive
def get_metadata(
    lh5_file: str | h5py.Group | h5py.File,
    build: bool = False,
    force: bool = False,
    metadata: dict = {}, 
    base: str = "/", 
    recursing: bool = False,
) -> dict:
    """Get metadata from an LH5 file.

    The `"metadata"` `Dataset` contains a `dict` (stored as a JSON string) of the `Attributes` of all `Datasets` 
    and `Groups` in the  `LH5` file. The structure of the `dict` matches the structure of the `LH5` file. It is used
    for much faster loading of attributes. 
    
    If the `"metadata"` `Dataset` is not found in the file, then the `dict` is generated from the `LH5` file itself by
    default, controlled by the `build` flag. 

    If the `"metadata"` `Dataset` is not found in the file and the metadata is not built, then `None` is returned.

    Parameters
    ----------
    lh5_file
        path to an `LH5` file
    build
        whether to build the metadata from the file if the `"metadata"` `Dataset` is not found; default is `False`
    force
        whether to ignore the `"metadata"` `Dataset` and build a `dict` from the file instead; default is `False`. 
        Ignores the `build` flag.
    """

    # open file
    if isinstance(lh5_file, str):
        # expand_path gives an error if file does not exist
        try: 
            fullpath = expand_path(lh5_file)
        except FileNotFoundError:
            log.debug(
                f"{lh5_file} does not exist, metadata is None"
            )
            return None
        lh5_file = h5py.File(fullpath, "r")

    # looks for "metadata" dataset and uses it if it exists
    # or you can force it to loop over the file to build it instead
    if not recursing and not force and 'metadata' in lh5_file:
        log.debug(
            f"metadata found in {lh5_file.filename}"
        )
        return json.loads(lh5_file['metadata'][()])
    elif build or force: # this is the recursive bit
        if not recursing:
            log.debug(
                f"metadata not found in {lh5_file.filename}, building it instead"
            )
        for obj in lh5_file:
            # if "metadata" actually was in the file and was missed due to forcing a rebuild, then the metadata
            # from the old file could be dragged along and updated and have outdated stuff in it
            # not 100% sure this is needed
            if obj == 'metadata':
                pass

            metadata[obj] = {}

            metadata[obj]['attrs'] = {}
            for attr in lh5_file[base+obj].attrs:
                metadata[obj]['attrs'][attr] = lh5_file[base+obj].attrs[attr]

            if isinstance(lh5_file[obj], h5py.Group):
                get_metadata(lh5_file[base+obj], metadata=metadata[obj], base=base+obj+'/', build=True, recursing=True)
    else:
        log.debug(
            f"metadata not found in {lh5_file.filename} and did not build it -> metadata is None"
        )
        return None
    
    # know thyself!
    if not recursing:
        metadata["metadata"] = {'attrs': {'datatype': 'JSON'}}
        
    return metadata