from __future__ import annotations

import fnmatch
import logging

from .. import Array, Scalar, Struct, Table, VectorOfVectors, lh5

log = logging.getLogger(__name__)


def lh5concat(
    lh5_files: list,
    output: str,
    overwrite: bool,
    *,
    include_list: list | None,
    exclude_list: list | None,
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
    obj_list = sorted(obj_list)

    msg = f"objects matching include patterns {include_list} in {file0}: {obj_list}"
    log.debug(msg)

    # 1. read first valid lgdo from left to right
    store = lh5.LH5Store()
    h5f0 = store.gimme_file(file0)
    lgdos = {}
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
            obj, _ = store.read(current, h5f0, n_rows=1)
            if isinstance(obj, (Table, Array, VectorOfVectors)):
                # read all!
                obj, _ = store.read(current, h5f0)
                lgdos[current] = obj
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

    msg = f"first-level, array-like objects: {lgdos.keys()}"
    log.debug(msg)
    msg = f"nested structs: {lgdo_structs.keys()}"
    log.debug(msg)

    h5f0.close()

    if lgdos == {}:
        msg = "did not find any field to concatenate, exit"
        log.error(msg)
        return

    # 2. remove (nested) table fields based on obj_list

    def _inplace_table_filter(name, table, obj_list):
        # filter objects nested in this LGDO
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

    for key, val in lgdos.items():
        if not isinstance(val, Table):
            continue

        _inplace_table_filter(key, val, obj_list)

    # 3. write to output file
    msg = f"creating output file {output}"
    log.info(msg)

    first_done = False
    for name, obj in lgdos.items():
        store.write(
            obj,
            name,
            output,
            wo_mode="overwrite_file"
            if (overwrite and not first_done)
            else "write_safe",
        )

        first_done = True

    # 4. loop over rest of files/names and write-append

    for file in lh5_files[1:]:
        msg = f"appending file {file} to {output}"
        log.info(msg)

        for name in lgdos:
            obj, _ = store.read(name, file)
            # need to remove nested LGDOs from obj too before appending
            if isinstance(obj, Table):
                _inplace_table_filter(name, obj, obj_list)

            store.write(obj, name, output, wo_mode="append")

    # 5. reset datatypes of the "group-like" structs

    if lgdo_structs != {}:
        output_file = store.gimme_file(output, mode="a")
        for struct, struct_dtype in lgdo_structs.items():
            msg = f"reset datatype of struct {struct} to {struct_dtype}"
            log.debug(msg)

            output_file[struct].attrs["datatype"] = struct_dtype
        output_file.close()
