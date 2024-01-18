"""legend-pydataobj's command line interface utilities."""
from __future__ import annotations

import argparse
import logging
import sys

import lgdo
import lgdo.logging
from lgdo.lh5 import show


def lh5ls():
    """:func:`.lh5.show` command line interface."""
    parser = argparse.ArgumentParser(
        prog="lh5ls", description="Inspect LEGEND HDF5 (LH5) file contents"
    )

    # global options
    parser.add_argument(
        "--version", action="store_true", help="""Print lgdo version and exit"""
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
        help="""Input LH5 file.""",
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

    args = parser.parse_args()

    if args.verbose:
        lgdo.logging.setup(logging.DEBUG)
    elif args.debug:
        lgdo.logging.setup(logging.DEBUG, logging.root)
    else:
        lgdo.logging.setup()

    if args.version:
        print(lgdo.__version__)  # noqa: T201
        sys.exit()

    show(args.lh5_file, args.lh5_group, attrs=args.attributes, depth=args.depth)
