"""Lazy command routing keeps unrelated hardware and learning dependencies out."""

import argparse
import importlib
import sys

from .paths import ROOT


def dispatch(description, commands, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("command", choices=sorted(commands))
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    args = parser.parse_args(argv[:1])
    module, *prefix = commands[args.command]
    target = importlib.import_module(module)
    saved = sys.argv
    try:
        sys.argv = [module, *prefix, *argv[1:]]
        return target.main()
    finally:
        sys.argv = saved
