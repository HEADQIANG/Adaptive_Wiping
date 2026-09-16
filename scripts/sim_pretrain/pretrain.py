"""Run with python -m scripts.sim_pretrain.pretrain from the project root."""

import argparse
import json
import sys

from scripts.shared.common import load_config


def main():
    parser = argparse.ArgumentParser(description="AIRBOT simulation / VAE pretraining")
    parser.add_argument("stage", choices=("sanity", "collect", "train", "evaluate", "export"))
    parser.add_argument("--config", default="configs/sim_pretrain/pretrain_paper.yaml")
    parser.add_argument(
        "--retry-failed", action="store_true", help="Retry the same failed collection assignments"
    )
    parser.add_argument(
        "--output", help="Fresh evaluation directory; required when evaluating an archived run"
    )
    args = parser.parse_args()
    if args.retry_failed and args.stage != "collect":
        parser.error("--retry-failed is only valid with collect")
    if args.output and args.stage != "evaluate":
        parser.error("--output is only valid with evaluate; other stages use config output_dir")
    try:
        cfg = load_config(args.config)
        from scripts.shared.run_paths import new_output, new_run_config

        if args.stage == "sanity":
            cfg = new_run_config(cfg)
        if args.output:
            args.output = new_output(args.output)
        if args.stage in ("sanity", "collect"):
            from scripts.sim_pretrain.collection import collect, sanity

            result = sanity(cfg) if args.stage == "sanity" else collect(cfg, args.retry_failed)
        else:
            from scripts.sim_pretrain.learning import evaluate, export, train

            result = (
                evaluate(cfg, output=args.output)
                if args.stage == "evaluate"
                else {"train": train, "export": export}[args.stage](cfg)
            )
        print(json.dumps(result, indent=2))
        if args.stage == "sanity" and not result["passed"]:
            return 2
        if args.stage == "collect" and not result["complete"]:
            return 2
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
