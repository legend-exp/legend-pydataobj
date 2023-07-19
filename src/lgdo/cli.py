"""Command line interface utilities."""
from __future__ import annotations

import argparse
import sys

from . import __version__, logging, show


def lh5ls():
    """``lh5ls`` console command.

    To learn more, have a look at the help section:

    .. code-block:: console

      $ lh5ls --help
    """

    parser = argparse.ArgumentParser(
        prog="lh5ls", description="Inspect LEGEND HDF5 (LH5) file contents"
    )

    # global options
    parser.add_argument(
        "--version", action="store_true", help="""Print the program version and exit"""
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="""Increase the program verbosity""",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="""Increase the program verbosity to maximum""",
    )

    # show options
    parser.add_argument(
        "lh5_file",
        help="""Input LH5 file.""",
    )
    parser.add_argument("lh5_group", nargs="?", help="""LH5 group.""", default="/")
    parser.add_argument(
        "--attributes",
        "-a",
        action="store_true",
        default=False,
        help="""Print HDF5 attributes too""",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.setup(logging.DEBUG)
    elif args.debug:
        logging.setup(logging.DEBUG, logging.root)
    else:
        logging.setup()

    if args.version:
        print(__version__)  # noqa: T201
        sys.exit()

    show(args.lh5_file, args.lh5_group, attrs=args.attributes)
