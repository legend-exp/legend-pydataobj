"""legend-pydataobj's command line interface utilities."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__, lh5
from . import logging as lgdogging  # eheheh
from .lh5.concat import lh5concat

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
        lgdogging.setup(logging.DEBUG, logging.getLogger("lgdo"))
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


def lh5concat_cli(args=None):
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
        lgdogging.setup(logging.INFO, logging.getLogger("lgdo"))
    elif args.debug:
        lgdogging.setup(logging.DEBUG, logging.root)
    else:
        lgdogging.setup()

    if args.version:
        print(__version__)  # noqa: T201
        sys.exit()

    lh5concat(
        lh5_files=args.lh5_file,
        overwrite=args.overwrite,
        output=args.output,
        include_list=args.include,
        exclude_list=args.exclude,
    )
