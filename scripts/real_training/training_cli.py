"""Offline training stages dispatched by the real_training package CLI."""

import argparse
import json
import sys

from scripts.real_training.config import load_config


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
    parser.add_argument("--config", default="configs/real_training/real_training_paper.yaml")
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
    try:
        cfg = load_config(args.config)
        if args.stage in ("inspect", "prepare"):
            from scripts.real_training.data import inspect, prepare

            info = inspect(cfg)[1] if args.stage == "inspect" else prepare(cfg)
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
