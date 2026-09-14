#!/usr/bin/env python3
"""Plot a completed KWR75 CSV without opening a serial port."""

import argparse
import csv
import sys
import textwrap
from pathlib import Path

import numpy as np

AXES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
UNITS = ("N", "N", "N", "Nm", "Nm", "Nm")
PANEL_ORDER = (0, 3, 1, 4, 2, 5)
COLORS = {"raw": "#bd572b", "net": "#137e79"}


def load_csv(path, modes):
    columns = [f"{mode}_{axis}_{unit}" for mode in modes for axis, unit in zip(AXES, UNITS)]
    times, indices, values = [], [], []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = ["frame_index", "elapsed_s", *columns]
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV header is missing or contains duplicate columns")
        missing = set(required) - set(reader.fieldnames)
        if missing:
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
        for row in reader:
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("incomplete or extra fields")
                index = int(row["frame_index"])
                t = float(row["elapsed_s"])
                w = [float(row[column]) for column in columns]
                if index < 1 or not np.isfinite([t, *w]).all() or t < 0:
                    raise ValueError("non-finite values, negative time, or invalid frame index")
                if times and (t < times[-1] or index <= indices[-1]):
                    raise ValueError("time must not decrease and frame_index must increase")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"CSV line {reader.line_num}: {exc}") from exc
            times.append(t)
            indices.append(index)
            values.append(w)
    if not times:
        raise ValueError("CSV has no data rows")
    array = np.asarray(values)
    return (
        np.asarray(times),
        np.asarray(indices),
        {mode: array[:, i * 6 : (i + 1) * 6] for i, mode in enumerate(modes)},
    )


def window_mean(times, values, window_s):
    """Trailing frame-weighted mean, evaluated once per recorded batch timestamp."""
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("Averaging window must be finite and positive")
    ends = np.unique(times)
    left = np.searchsorted(times, ends - window_s, side="left")
    right = np.searchsorted(times, ends, side="right")
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    return ends, (cumulative[right] - cumulative[left]) / (right - left)[:, None]


def broken_line(times, values, breaks):
    """Insert NaNs to avoid drawing continuous lines through missing data."""
    positions = np.flatnonzero(breaks) + 1
    return np.insert(times, positions, np.nan), np.insert(values, positions, np.nan, axis=0)


def make_figure(times, indices, data, name, window_ms=0):
    import matplotlib.pyplot as plt

    fig, panels = plt.subplots(3, 2, figsize=(12, 8), sharex=True, layout="constrained")
    gaps = (np.diff(times) > 0.05) | (np.diff(indices) > 1)
    for mode, values in data.items():
        x, y = broken_line(times, values, gaps)
        averaged = None
        if window_ms > 0:
            avg_t, avg = window_mean(times, values, window_ms / 1000)
            # Carry frame-index gaps through to the averaged curve as well.
            cut_times = times[1:][gaps]
            avg_gaps = np.isin(avg_t[1:], cut_times) | (np.diff(avg_t) > 0.05)
            averaged = broken_line(avg_t, avg, avg_gaps)
        for channel, ax in zip(PANEL_ORDER, panels.flat):
            ax.plot(
                x,
                y[:, channel],
                color=COLORS[mode],
                linewidth=0.65,
                alpha=0.3 if averaged is not None else 0.85,
                marker="." if len(times) == 1 else None,
                label=f"{mode.capitalize()} frames",
            )
            if averaged is not None:
                ax.plot(
                    averaged[0],
                    averaged[1][:, channel],
                    color=COLORS[mode],
                    linewidth=1.15,
                    marker="." if len(avg_t) == 1 else None,
                    label=f"{mode.capitalize()} {window_ms:g} ms mean",
                )
    for channel, ax in zip(PANEL_ORDER, panels.flat):
        ax.set_ylabel(f"{AXES[channel]} ({UNITS[channel]})")
        ax.axhline(0, color="#737373", linewidth=0.6, linestyle=":")
        ax.grid(alpha=0.18)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    for ax in panels[-1]:
        ax.set_xlabel("Elapsed time (s)")
    panels[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "KWR75 | "
        + "\n".join(textwrap.wrap(name, width=85))
        + f"\n{len(times):,} frames | {times[0]:.3f}-{times[-1]:.3f} s",
        fontsize=12,
    )
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="completed CSV from kwr75_reader.py")
    parser.add_argument("--mode", choices=("raw", "net", "both"), default="net")
    parser.add_argument(
        "--window-ms",
        type=float,
        default=0,
        help="overlay a trailing mean, e.g. 10; 0 disables averaging",
    )
    parser.add_argument("--start", type=float, default=0, help="first elapsed second to plot")
    parser.add_argument("--end", type=float, help="last elapsed second to plot (inclusive)")
    parser.add_argument("--output", type=Path, help="new PNG path; default beside the CSV")
    parser.add_argument("--show", action="store_true", help="also open an interactive plot window")
    args = parser.parse_args(argv)
    if not np.isfinite(args.window_ms) or args.window_ms < 0:
        parser.error("--window-ms must be finite and nonnegative")
    if not np.isfinite(args.start) or args.start < 0:
        parser.error("--start must be finite and nonnegative")
    if args.end is not None and (not np.isfinite(args.end) or args.end < args.start):
        parser.error("--end must be finite and >= --start")
    suffix = f"_{args.mode}"
    if args.window_ms:
        suffix += f"_{args.window_ms:g}ms"
    if args.start or args.end is not None:
        suffix += f"_{args.start:g}-{args.end if args.end is not None else 'end'}s"
    output = args.output or args.csv.with_name(args.csv.stem + suffix + ".png")
    if output.suffix.lower() != ".png":
        parser.error("--output must have a .png extension")
    if output.resolve() == args.csv.resolve() or output.exists():
        parser.error(f"Output already exists or is the input (will not overwrite): {output}")
    fig = None
    try:
        modes = ("raw", "net") if args.mode == "both" else (args.mode,)
        times, indices, data = load_csv(args.csv, modes)
        selected = times >= args.start
        if args.end is not None:
            selected &= times <= args.end
        if not selected.any():
            raise ValueError("No frames in the selected time range")
        times, indices = times[selected], indices[selected]
        data = {mode: values[selected] for mode, values in data.items()}
        import matplotlib

        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = make_figure(times, indices, data, args.csv.name, args.window_ms)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=160)
        print(f"Saved: {output}\nFrames: {len(times)} | elapsed: {times[0]:.6f}-{times[-1]:.6f} s")
        gaps = np.count_nonzero((np.diff(times) > 0.05) | (np.diff(indices) > 1))
        if gaps:
            print(f"Warning: {gaps} time/frame gaps; plotted lines are broken at gaps.")
        print("Timestamps are host batch times; repeated timestamps are preserved.")
        if args.show:
            if str(matplotlib.get_backend()).lower() == "agg":
                print(
                    "Warning: no interactive backend; PNG saved. Use a desktop session with Tk/Qt.",
                    file=sys.stderr,
                )
            else:
                plt.show()
        return 0
    except (OSError, ValueError, ImportError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt

            plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
