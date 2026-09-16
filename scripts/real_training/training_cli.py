"""Offline training stages dispatched by the real_training package CLI."""

import argparse
import json
import sys

from scripts.real_training.config import load_config, resolve, validate_config
from scripts.shared.paths import writable_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Offline real-demonstration training; never commands hardware"
    )
    parser.add_argument(
        "stage",
        choices=(
            "inspect",
            "prepare",
            "cross-validate",
            "train",
            "evaluate",
            "export",
            "smoke-test",
        ),
    )
    parser.add_argument("--config", help="Training defaults or a saved run_config.yaml")
    parser.add_argument("--demonstrations", "--session", dest="demonstrations",
                        help="Accepted demonstration session directory (or session.json)")
    parser.add_argument("--exploration", help="Completed real exploration JSONL")
    parser.add_argument("--raw-data", help="Already imported raw HDF5 (instead of recordings)")
    parser.add_argument("--encoder", help="Frozen simulation encoder checkpoint")
    parser.add_argument("--output-dir", help="Dedicated training output directory")
    parser.add_argument("--confirm-same-setup", action="store_true",
                        help="Confirm the manual demonstration and exploration installation match")
    parser.add_argument("--programmed-hold-last", action="store_true",
                        help="Explicitly allow derived 10s padding of programmed demonstrations")
    parser.add_argument("--calibration", help="Calibration record for sensor-calibrated import")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Restore exact optimizer/RNG state for train or cross-validate",
    )
    parser.add_argument(
        "--keep-smoke-artifacts",
        action="store_true",
        help="Keep labeled synthetic artifacts in a new temporary directory",
    )
    args = parser.parse_args(argv)
    if args.resume and args.stage not in ("train", "cross-validate"):
        parser.error("--resume is only valid with train or cross-validate")
    if args.keep_smoke_artifacts and args.stage != "smoke-test":
        parser.error("--keep-smoke-artifacts is only valid with smoke-test")
    recordings = bool(args.demonstrations or args.exploration)
    if recordings and not (args.demonstrations and args.exploration):
        parser.error("--demonstrations and --exploration must be supplied together")
    if recordings and (args.raw_data or args.resume or args.stage not in ("inspect", "prepare", "train")):
        parser.error("Recordings require inspect/prepare/train without --raw-data or --resume; resume with --config run_config.yaml")
    if not recordings and (args.confirm_same_setup or args.programmed_hold_last or args.calibration):
        parser.error("Recording import options require --demonstrations and --exploration")
    if args.resume and (args.raw_data or args.encoder or args.output_dir):
        parser.error("--resume uses the saved --config unchanged; path overrides are forbidden")
    if args.stage == "smoke-test" and (args.raw_data or args.encoder or args.output_dir):
        parser.error("smoke-test uses isolated synthetic paths; input/output overrides are forbidden")
    try:
        if recordings:
            from scripts.real_training.workflow import run_recordings

            print(json.dumps(run_recordings(args), indent=2, allow_nan=False))
            return 0
        cfg = load_config(args.config or "configs/real_training/real_training_paper.yaml")
        path_overrides = any((args.raw_data, args.encoder, args.output_dir))
        if args.raw_data:
            cfg.pop("recordings", None)
        for key in ("raw_data", "encoder", "output_dir"):
            if getattr(args, key):
                cfg[key] = str(writable_path(args.output_dir) if key == "output_dir"
                               else resolve(getattr(args, key)).resolve())
        if path_overrides:
            validate_config(cfg)
        if args.stage == "prepare":
            from scripts.shared.run_paths import new_run_config

            cfg = new_run_config(cfg)
        if args.stage in ("inspect", "prepare"):
            from scripts.real_training.data import inspect, prepare

            info = inspect(cfg)[1] if args.stage == "inspect" else prepare(cfg)
            if args.stage == "prepare" and path_overrides:
                import yaml

                snapshot = resolve(cfg["output_dir"]) / "run_config.yaml"
                with snapshot.open("x", encoding="utf-8") as stream:
                    yaml.safe_dump(cfg, stream, allow_unicode=True, sort_keys=False)
                print(f"Run config (use for subsequent stages): {snapshot}", file=sys.stderr, flush=True)
            result = {
                key: info[key]
                for key in (
                    "source_kind",
                    "raw_sha256",
                    "encoder_sha256",
                    "demo_ids",
                    "encoder_epoch",
                    "encoder_randomization",
                    "valid_windows_total",
                    "exploration_out_of_range_per_channel",
                    "warnings",
                )
            }
            if args.stage == "prepare" and path_overrides:
                result["run_config"] = str(snapshot)
        elif args.stage == "smoke-test":
            from scripts.real_training.testing import smoke_test

            result = smoke_test(cfg, keep=args.keep_smoke_artifacts)
        else:
            from scripts.real_training.training import (
                cross_validate,
                evaluate,
                export,
                train,
            )

            action = {
                "cross-validate": cross_validate,
                "train": train,
                "evaluate": evaluate,
                "export": export,
            }[args.stage]
            result = (
                action(cfg, resume=args.resume)
                if args.stage in ("train", "cross-validate")
                else action(cfg)
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
