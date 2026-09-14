#!/usr/bin/env python3
"""Report signed per-axis maxima and minima of recorded KWR75 net values."""

import argparse
import csv
import sys
from pathlib import Path

if __package__:
    from scripts.force_sensor.plot_kwr75_csv import AXES, UNITS, load_csv
else:
    from scripts.force_sensor.plot_kwr75_csv import AXES, UNITS, load_csv


def summarize(path):
    _, _, data = load_csv(path, ("net",))
    values = data["net"]
    maxima = values.max(axis=0)
    minima = values.min(axis=0)
    return [
        {
            "axis": axis,
            "unit": unit,
            "max": float(maxima[channel]),
            "min": float(minima[channel]),
        }
        for channel, (axis, unit) in enumerate(zip(AXES, UNITS))
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="completed CSV from kwr75_reader.py")
    args = parser.parse_args(argv)
    try:
        result = summarize(args.csv)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"{'Axis':<5} {'Unit':<5} {'Max':>14} {'Min':>14}")
    for row in result:
        print(f"{row['axis']:<5} {row['unit']:<5} {row['max']:>+14.6f} {row['min']:>+14.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
