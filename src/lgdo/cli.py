"""legend-pydataobj's command line interface utilities."""

from __future__ import annotations

import argparse
import fnmatch
import logging
import sys

from . import Array, Table, VectorOfVectors, __version__, lh5
from . import logging as lgdogging  # eheheh

log = logging.getLogger(__name__)


def lh5ls(args=None):
    """:func:`.lh5.show` command line interface."""
    parser = argparse.ArgumentParser(
        prog="lh5ls", description="Inspect LEGEND HDF5 (LH5) file contents"
    )

    # global options
    parser.add_argument(
        "--version",
        action="store_true",
        help="""Print legend-pydataobj version and exit""",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="""Increase the program verbosity""",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="""Increase the program verbosity to maximum""",
    )

    parser.add_argument(
        "lh5_file",
        help="""Input LH5 file""",
    )
    parser.add_argument("lh5_group", nargs="?", help="""LH5 group.""", default="/")
    parser.add_argument(
        "--attributes", "-a", action="store_true", help="""Print HDF5 attributes too"""
    )
    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        default=None,
        help="""Maximum tree depth of groups to print""",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="""Print details about datasets""",
    )

    args = parser.parse_args(args)

    if args.verbose:
        lgdogging.setup(logging.DEBUG)
    elif args.debug:
        lgdogging.setup(logging.DEBUG, logging.root)
    else:
        lgdogging.setup()

    if args.version:
        print(__version__)  # noqa: T201
        sys.exit()

    lh5.show(
        args.lh5_file,
        args.lh5_group,
        attrs=args.attributes,
        depth=args.depth,
        detail=args.detail,
    )


def lh5concat(args=None):
    """Command line interface for concatenating array-like LGDOs in LH5 files."""
    parser = argparse.ArgumentParser(
        prog="lh5concat",
        description="""
Concatenate LGDO Arrays, VectorOfVectors and Tables in LH5 files.

Examples
--------

Concatenate all eligible objects in file{1,2}.lh5 into concat.lh5:

  $ lh5concat -o concat.lh5 file1.lh5 file2.lh5

Include only the /data/table1 Table:

  $ lh5concat -o concat.lh5 -i /data/table1/* file1.lh5 file2.lh5

Exclude the /data/table1/col1 Table column:

  $ lh5concat -o concat.lh5 -e /data/table1/col1 file1.lh5 file2.lh5
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # global options
    parser.add_argument(
        "--version",
        action="store_true",
        help="""Print legend-pydataobj version and exit""",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="""Increase the program verbosity""",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="""Increase the program verbosity to maximum""",
    )

    parser.add_argument(
        "lh5_file",
        nargs="+",
        help="""Input LH5 files""",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="""Output file""",
        default="lh5concat-output.lh5",
    )
    parser.add_argument(
        "--overwrite",
        "-w",
        action="store_true",
        help="""Overwrite output file""",
    )
    parser.add_argument(
        "--include",
        "-i",
        help="""Regular expression (fnmatch style) for object names that should
        be concatenated. To include full tables, you need to explicitly include
        all its columns with e.g. '/path/to/table/*'. The option can be passed
        multiple times to provide a list of patterns.
        """,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--exclude",
        "-e",
        help="""List of object names that should be excluded. Takes priority
        over --include. See --include help for more details.
        """,
        action="append",
        default=None,
    )

    args = parser.parse_args(args)

    if args.verbose:
        lgdogging.setup(logging.INFO, log)
    elif args.debug:
        lgdogging.setup(logging.DEBUG, logging.root)
    else:
        lgdogging.setup()

    if args.version:
        print(__version__)  # noqa: T201
        sys.exit()

    if len(args.lh5_file) < 2:
        msg = "you must provide at least two input files"
        raise RuntimeError(msg)

    # determine list of objects by recursively ls'ing first file
    file0 = args.lh5_file[0]
    obj_list_full = set(lh5.ls(file0, recursive=True))

    # let's remove objects with nested LGDOs inside
    to_remove = set()
    for name in obj_list_full:
        if len(fnmatch.filter(obj_list_full, f"{name}/*")) > 1:
            to_remove.add(name)
    obj_list_full -= to_remove

    obj_list = set()
    # now first remove excluded stuff
    if args.exclude is not None:
        for exc in args.exclude:
            obj_list_full -= set(fnmatch.filter(obj_list_full, exc.strip("/")))

    # then make list of included, based on latest list
    if args.include is not None:
        for inc in args.include:
            obj_list |= set(fnmatch.filter(obj_list_full, inc.strip("/")))
    else:
        obj_list = obj_list_full

    # sort
    obj_list = sorted(obj_list)

    msg = f"objects matching include patterns {args.include} in {file0}: {obj_list}"
    log.debug(msg)

    # 1. read first valid lgdo from left to right
    store = lh5.LH5Store()
    h5f0 = store.gimme_file(file0)
    lgdos = {}
    # loop over object list in the first file
    for name in obj_list:
        # now loop over groups starting from root
        current = ""
        for item in name.split("/"):
            current = f"{current}/{item}".strip("/")

            if current in lgdos:
                break

            # not even an LGDO!
            if "datatype" not in h5f0[current].attrs:
                continue

            # read as little as possible
            obj, _ = store.read(current, h5f0, n_rows=1)
            if isinstance(obj, (Table, Array, VectorOfVectors)):
                # read all!
                obj, _ = store.read(current, h5f0)
                lgdos[current] = obj

            break

    msg = f"first-level, array-like objects: {lgdos.keys()}"
    log.debug(msg)

    h5f0.close()

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
    msg = f"creating output file {args.output}"
    log.info(msg)

    first_done = False
    for name, obj in lgdos.items():
        store.write(
            obj,
            name,
            args.output,
            wo_mode="overwrite_file"
            if (args.overwrite and not first_done)
            else "write_safe",
        )

        first_done = True

    # 4. loop over rest of files/names and write-append

    for file in args.lh5_file[1:]:
        msg = f"appending file {file} to {args.output}"
        log.info(msg)

        for name in lgdos:
            obj, _ = store.read(name, file)
            # need to remove nested LGDOs from obj too before appending
            if isinstance(obj, Table):
                _inplace_table_filter(name, obj, obj_list)

            store.write(obj, name, args.output, wo_mode="append")
