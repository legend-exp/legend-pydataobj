"""legend-pydataobj's command line interface utilities."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__, lh5
from . import logging as lgdogging  # eheheh
from .lh5.concat import lh5concat
from .lh5.truncate import truncate

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


def lh5truncate_cli(args=None):
    """Command line interface for truncating LH5 files."""
    parser = argparse.ArgumentParser(
        prog="lh5truncate",
        description="""
Truncate LGDO Arrays, VectorOfVectors and Tables in LH5 files.

Examples
--------

Truncate to keep first 100 rows:

  $ lh5truncate infile.lh5 -o outfile.lh5 100

Truncate with a slice (rows 10 to 50) for hit-ordered data using a TCM file:

  $ lh5truncate infile.lh5 -o outfile.lh5 10:50 --tcm-file tcm.lh5 --file-type raw

Include only specific paths:

  $ lh5truncate infile.lh5 -o outfile.lh5 50 -i 'ch*'
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
        help="""Input LH5 file""",
    )
    parser.add_argument(
        "length_or_slice",
        help="""Number of rows to keep (e.g. 100) or slice notation (e.g. 10:50)""",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="""Output file""",
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        "-w",
        action="store_true",
        help="""Overwrite output file""",
    )
    parser.add_argument(
        "--tcm-file",
        help="""Path to TCM file (required for hit-ordered truncation)""",
        default=None,
    )
    parser.add_argument(
        "--file-type",
        help="""File type (auto-deduced if not provided). Options: evt, tcm, raw,
        dsp, hit, any-evt, any-hit""",
        default=None,
    )
    parser.add_argument(
        "--include",
        "-i",
        help="""fnmatch pattern for object names to include. The option can be
        passed multiple times to provide a list of patterns.
        """,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--exclude",
        "-e",
        help="""fnmatch pattern for object names to exclude. Takes priority over
        --include. Can be passed multiple times.
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

    # parse length_or_slice: could be an integer or a slice like "10:50"
    length_or_slice: int | slice
    if ":" in args.length_or_slice:
        parts = args.length_or_slice.split(":")
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
        step = int(parts[2]) if len(parts) > 2 and parts[2] else None
        length_or_slice = slice(start, stop, step)
    else:
        length_or_slice = int(args.length_or_slice)

    truncate(
        infile=args.lh5_file,
        outfile=args.output,
        length_or_slice=length_or_slice,
        overwrite=args.overwrite,
        tcm_file=args.tcm_file,
        file_type=args.file_type,
        include_list=args.include,
        exclude_list=args.exclude,
    )
