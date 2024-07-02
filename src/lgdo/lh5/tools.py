from __future__ import annotations

import fnmatch
import glob
import logging
import os
from copy import copy
from warnings import warn

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import utils
from .store import LH5Store

log = logging.getLogger(__name__)


def ls(
    lh5_file: str | h5py.Group,
    lh5_group: str = "",
    recursive: bool = False,
) -> list[str]:
    """Return a list of LH5 groups in the input file and group, similar
    to ``ls`` or ``h5ls``. Supports wildcards in group names.


    Parameters
    ----------
    lh5_file
        name of file.
    lh5_group
        group to search. add a ``/`` to the end of the group name if you want to
        list all objects inside that group.
    recursive
        if ``True``, recurse into subgroups.
    """

    log.debug(
        f"Listing objects in '{lh5_file}'"
        + ("" if lh5_group == "" else f" (and group {lh5_group})")
    )

    lh5_st = LH5Store()
    # To use recursively, make lh5_file a h5group instead of a string
    if isinstance(lh5_file, str):
        lh5_file = lh5_st.gimme_file(lh5_file, "r")
        if lh5_group.startswith("/"):
            lh5_group = lh5_group[1:]

    if lh5_group == "":
        lh5_group = "*"

    # get the first group in the group path
    splitpath = lh5_group.split("/", 1)
    # filter out objects that don't match lh5_group pattern
    matchingkeys = fnmatch.filter(lh5_file.keys(), splitpath[0])

    ret = []
    # if there were no "/" in lh5_group just return the result
    if len(splitpath) == 1:
        ret = matchingkeys

    else:
        for key in matchingkeys:
            ret.extend([f"{key}/{path}" for path in ls(lh5_file[key], splitpath[1])])

    if recursive:
        rec_ret = copy(ret)
        for obj in ret:
            try:
                rec_ret += ls(lh5_file, lh5_group=f"{obj}/", recursive=True)
            except AttributeError:
                continue

        return rec_ret

    return ret


def show(
    lh5_file: str | h5py.Group,
    lh5_group: str = "/",
    attrs: bool = False,
    indent: str = "",
    header: bool = True,
    depth: int | None = None,
    detail: bool = False,
) -> None:
    """Print a tree of LH5 file contents with LGDO datatype.

    Parameters
    ----------
    lh5_file
        the LH5 file.
    lh5_group
        print only contents of this HDF5 group.
    attrs
        print the HDF5 attributes too.
    indent
        indent the diagram with this string.
    header
        print `lh5_group` at the top of the diagram.
    depth
        maximum tree depth of groups to print
    detail
        whether to print additional information about how the data is stored

    Examples
    --------
    >>> from lgdo import show
    >>> show("file.lh5", "/geds/raw")
    /geds/raw
    ├── channel · array<1>{real}
    ├── energy · array<1>{real}
    ├── timestamp · array<1>{real}
    ├── waveform · table{t0,dt,values}
    │   ├── dt · array<1>{real}
    │   ├── t0 · array<1>{real}
    │   └── values · array_of_equalsized_arrays<1,1>{real}
    └── wf_std · array<1>{real}
    """
    # check tree depth if we are using it
    if depth is not None and depth <= 0:
        return

    # open file
    if isinstance(lh5_file, str):
        lh5_file = h5py.File(utils.expand_path(lh5_file), "r")

    # go to group
    if lh5_group != "/":
        lh5_file = lh5_file[lh5_group]

    if header:
        print(f"\033[1m{lh5_group}\033[0m")  # noqa: T201

    # get an iterator over the keys in the group
    it = iter(lh5_file)
    key = None

    # make sure there is actually something in this file/group
    try:
        key = next(it)  # get first key
    except StopIteration:
        print(f"{indent}└──  empty")  # noqa: T201
        return

    # loop over keys
    while True:
        val = lh5_file[key]
        # we want to print the LGDO datatype
        dtype = val.attrs.get("datatype", default="no datatype")
        if dtype == "no datatype" and isinstance(val, h5py.Group):
            dtype = "HDF5 group"

        _attrs = ""
        if attrs:
            attrs_d = dict(val.attrs)
            attrs_d.pop("datatype", "")
            _attrs = "── " + str(attrs_d) if attrs_d else ""

        # is this the last key?
        killme = False
        try:
            k_new = next(it)  # get next key
        except StopIteration:
            char = "└──"
            killme = True  # we'll have to kill this loop later
        else:
            char = "├──"

        print(f"{indent}{char} \033[1m{key}\033[0m · {dtype} {_attrs}")  # noqa: T201

        if detail and isinstance(val, h5py.Dataset):
            char = "|       "
            if killme:
                char = "        "
            toprint = f"{indent}{char}"
            try:
                toprint += f"\033[3mdtype\033[0m={val.dtype}"
                toprint += f", \033[3mshape\033[0m={val.shape}"
                toprint += f", \033[3mnbytes\033[0m={utils.fmtbytes(val.nbytes)}"
                if (chunkshape := val.chunks) is None:
                    toprint += ", \033[3mnumchunks\033[0m=contiguous"
                else:
                    toprint += f", \033[3mnumchunks\033[0m={val.id.get_num_chunks()}"
                    toprint += f", \033[3mchunkshape\033[0m={chunkshape}"
                toprint += ", \033[3mfilters\033[0m="

                numfilters = val.id.get_create_plist().get_nfilters()
                if numfilters == 0:
                    toprint += "None"
                else:
                    toprint += "("
                    for i in range(numfilters):
                        thisfilter = val.id.get_create_plist().get_filter(i)[3].decode()
                        if "lz4" in thisfilter:
                            thisfilter = "lz4"
                        toprint += f"{thisfilter},"
                    toprint += ")"

            except TypeError:
                toprint += "(scalar)"

            print(toprint)  # noqa: T201

        # if it's a group, call this function recursively
        if isinstance(val, h5py.Group):
            show(
                val,
                indent=indent + ("    " if killme else "│   "),
                header=False,
                attrs=attrs,
                depth=depth - 1 if depth else None,
                detail=detail,
            )

        # break or move to next key
        if killme:
            break

        key = k_new


def load_nda(
    f_list: str | list[str],
    par_list: list[str],
    lh5_group: str = "",
    idx_list: list[NDArray | list | tuple] | None = None,
) -> dict[str, NDArray]:
    r"""Build a dictionary of :class:`numpy.ndarray`\ s from LH5 data.

    Given a list of files, a list of LH5 table parameters, and an optional
    group path, return a NumPy array with all values for each parameter.

    Parameters
    ----------
    f_list
        A list of files. Can contain wildcards.
    par_list
        A list of parameters to read from each file.
    lh5_group
        group path within which to find the specified parameters.
    idx_list
        for fancy-indexed reads. Must be one index array for each file in
        `f_list`.

    Returns
    -------
    par_data
        A dictionary of the parameter data keyed by the elements of `par_list`.
        Each entry contains the data for the specified parameter concatenated
        over all files in `f_list`.
    """
    warn(
        "load_nda() is deprecated. "
        "Please replace it with LH5Store.read(...).view_as('np'), "
        "or just read_as(..., 'np'). "
        "load_nda() will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(f_list, str):
        f_list = [f_list]
        if idx_list is not None:
            idx_list = [idx_list]
    if idx_list is not None and len(f_list) != len(idx_list):
        msg = f"f_list length ({len(f_list)}) != idx_list length ({len(idx_list)})!"
        raise ValueError(msg)

    # Expand wildcards
    f_list = [f for f_wc in f_list for f in sorted(glob.glob(os.path.expandvars(f_wc)))]

    sto = LH5Store()
    par_data = {par: [] for par in par_list}
    for ii, ff in enumerate(f_list):
        f = sto.gimme_file(ff, "r")
        for par in par_list:
            if f"{lh5_group}/{par}" not in f:
                msg = f"'{lh5_group}/{par}' not in file {ff}"
                raise RuntimeError(msg)

            if idx_list is None:
                data, _ = sto.read(f"{lh5_group}/{par}", f)
            else:
                data, _ = sto.read(f"{lh5_group}/{par}", f, idx=idx_list[ii])
            if not data:
                continue
            par_data[par].append(data.nda)
    return {par: np.concatenate(par_data[par]) for par in par_list}


def load_dfs(
    f_list: str | list[str],
    par_list: list[str],
    lh5_group: str = "",
    idx_list: list[NDArray | list | tuple] | None = None,
) -> pd.DataFrame:
    """Build a :class:`pandas.DataFrame` from LH5 data.

    Given a list of files (can use wildcards), a list of LH5 columns, and
    optionally the group path, return a :class:`pandas.DataFrame` with all
    values for each parameter.

    See Also
    --------
    :func:`load_nda`

    Returns
    -------
    dataframe
        contains columns for each parameter in `par_list`, and rows containing
        all data for the associated parameters concatenated over all files in
        `f_list`.
    """
    warn(
        "load_dfs() is deprecated. "
        "Please replace it with LH5Store.read(...).view_as('pd'), "
        "or just read_as(..., 'pd'). "
        "load_dfs() will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return pd.DataFrame(
        load_nda(f_list, par_list, lh5_group=lh5_group, idx_list=idx_list)
    )
