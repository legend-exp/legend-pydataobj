from __future__ import annotations

import fnmatch
import logging

from lgdo.lh5 import LH5Iterator

from .. import Array, Scalar, Struct, Table, VectorOfVectors, lh5

log = logging.getLogger(__name__)


def _get_obj_list(
    lh5_files: list, include_list: list | None = None, exclude_list: list | None = None
) -> list[str]:
    """Extract a list of lh5 objects to concatenate.

    Parameters
    ----------
    lh5_files
        list of input files to concatenate.
    include_list
        patterns for tables to include.
    exclude_list
        patterns for tables to exclude.

    """
    file0 = lh5_files[0]
    obj_list_full = set(lh5.ls(file0, recursive=True))

    # let's remove objects with nested LGDOs inside
    to_remove = set()
    for name in obj_list_full:
        if len(fnmatch.filter(obj_list_full, f"{name}/*")) > 1:
            to_remove.add(name)
    obj_list_full -= to_remove

    obj_list = set()
    # now first remove excluded stuff
    if exclude_list is not None:
        for exc in exclude_list:
            obj_list_full -= set(fnmatch.filter(obj_list_full, exc.strip("/")))

    # then make list of included, based on latest list
    if include_list is not None:
        for inc in include_list:
            obj_list |= set(fnmatch.filter(obj_list_full, inc.strip("/")))
    else:
        obj_list = obj_list_full

    # sort
    return sorted(obj_list)


def _get_lgdos(file, obj_list):
    """Get name of LGDO objects."""

    store = lh5.LH5Store()
    h5f0 = store.gimme_file(file)

    lgdos = []
    lgdo_structs = {}

    # loop over object list in the first file
    for name in obj_list:
        # now loop over groups starting from root
        current = ""
        for item in name.split("/"):
            current = f"{current}/{item}".strip("/")

            if current in lgdos:
                break

            # not even an LGDO (i.e. a plain HDF5 group)!
            if "datatype" not in h5f0[current].attrs:
                continue

            # read as little as possible
            obj = store.read(current, h5f0, n_rows=1)
            if isinstance(obj, (Table, Array, VectorOfVectors)):
                lgdos.append(current)

            elif isinstance(obj, Struct):
                # structs might be used in a "group-like" fashion (i.e. they might only
                # contain array-like objects).
                # note: handle after handling tables, as tables also satisfy this check.
                lgdo_structs[current] = obj.attrs["datatype"]
                continue

            elif isinstance(obj, Scalar):
                msg = f"cannot concat scalar field {current}"
                log.warning(msg)

            break

    msg = f"first-level, array-like objects: {lgdos}"
    log.info(msg)

    msg = f"nested structs: {lgdo_structs}"
    log.info(msg)

    h5f0.close()

    if lgdos == []:
        msg = "did not find any field to concatenate, exit"
        raise RuntimeError(msg)

    return lgdos, lgdo_structs


def _inplace_table_filter(name, table, obj_list):
    """filter objects nested in this LGDO"""
    skm = fnmatch.filter(obj_list, f"{name}/*")
    kept = {it.removeprefix(name).strip("/").split("/")[0] for it in skm}

    # now remove fields
    for k in list(table.keys()):
        if k not in kept:
            table.remove_column(k)

    msg = f"fields left in table '{name}': {table.keys()}"
    log.debug(msg)

    # recurse!
    for k2, v2 in table.items():
        if not isinstance(v2, Table):
            continue

        _inplace_table_filter(f"{name}/{k2}", v2, obj_list)


def _remove_nested_fields(lgdos: dict, obj_list: list):
    """Remove (nested) table fields based on obj_list."""

    for key, val in lgdos.items():
        if not isinstance(val, Table):
            continue

        _inplace_table_filter(key, val, obj_list)


def lh5concat(
    lh5_files: list,
    output: str,
    overwrite: bool = False,
    *,
    include_list: list | None = None,
    exclude_list: list | None = None,
) -> None:
    """Concatenate LGDO Arrays, VectorOfVectors and Tables in LH5 files.

    Parameters
    ----------
    lh5_files
        list of input files to concatenate.
    output
        path to the output file
    include_list
        patterns for tables to include.
    exclude_list
        patterns for tables to exclude.
    """

    if len(lh5_files) < 2:
        msg = "you must provide at least two input files"
        raise RuntimeError(msg)

    # determine list of objects by recursively ls'ing first file
    obj_list = _get_obj_list(
        lh5_files, include_list=include_list, exclude_list=exclude_list
    )

    msg = f"objects matching include patterns {include_list} in {lh5_files[0]}: {obj_list}"
    log.info(msg)

    lgdos, lgdo_structs = _get_lgdos(lh5_files[0], obj_list)
    first_done = False
    store = lh5.LH5Store()

    # loop over lgdo objects
    for lgdo in lgdos:
        # iterate over the files
        for lh5_obj in LH5Iterator(lh5_files, lgdo):
            data = {lgdo: lh5_obj}

            # remove the nested fields
            _remove_nested_fields(data, obj_list)

            if first_done is False:
                msg = f"creating output file {output}"
                log.info(msg)

                store.write(
                    data[lgdo],
                    lgdo,
                    output,
                    wo_mode="overwrite_file"
                    if (overwrite and not first_done)
                    else "write_safe",
                )
                first_done = True

            else:
                msg = f"appending to {output}"
                log.info(msg)

                if isinstance(data[lgdo], Table):
                    _inplace_table_filter(lgdo, data[lgdo], obj_list)

                store.write(data[lgdo], lgdo, output, wo_mode="append")

    if lgdo_structs != {}:
        output_file = store.gimme_file(output, mode="a")
        for struct, struct_dtype in lgdo_structs.items():
            msg = f"reset datatype of struct {struct} to {struct_dtype}"
            log.debug(msg)

            output_file[struct].attrs["datatype"] = struct_dtype
        output_file.close()
