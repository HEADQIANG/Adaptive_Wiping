"""Path-driven, offline collection-to-policy workflow using the audited importers."""

import copy
import json
import sys
import tempfile
from pathlib import Path

import yaml

from scripts.real_training.config import load_config, resolve, validate_config
from scripts.shared.paths import writable_path
from scripts.shared.run_paths import new_output


def recording_config(args):
    session = resolve(args.demonstrations).resolve()
    if session.name == "session.json" and session.is_file():
        session = session.parent
    metadata = json.loads((session / "session.json").read_text(encoding="utf-8"))
    exploration = resolve(args.exploration).resolve()
    if not exploration.is_file():
        raise FileNotFoundError(f"Exploration log not found: {exploration}")
    collected = metadata["config"]
    programmed = (metadata.get("demonstration_source") == "programmed"
                  or str(metadata.get("protocol", collected.get("protocol", ""))).startswith("airbot_programmed"))
    manual_tared = not programmed and collected.get("force_recording") == "software_tared"
    if programmed != args.programmed_hold_last:
        raise ValueError("Programmed recordings require --programmed-hold-last; manual recordings forbid it")
    if manual_tared and not args.confirm_same_setup:
        raise ValueError("Manual tared recordings require --confirm-same-setup")
    if args.confirm_same_setup and not manual_tared:
        raise ValueError("--confirm-same-setup is only valid for manual tared recordings")
    if manual_tared:
        default = "real_training_manual_tared.yaml"
    elif programmed:
        default = "real_training_programmed_wide1200.yaml"
    else:
        default = "real_training_airbot_native.yaml"
    cfg = load_config(args.config or f"configs/real_training/{default}")
    cfg = copy.deepcopy(cfg)
    if args.encoder:
        cfg["encoder"] = str(resolve(args.encoder).resolve())
    if cfg["profile"] == "airbot_sensor_calibrated_offline":
        if not args.calibration or manual_tared or programmed:
            raise ValueError("Calibrated import requires --calibration and raw manual recordings")
    elif args.calibration:
        raise ValueError("--calibration requires a sensor-calibrated --config")
    cfg["recordings"] = {"demonstrations": str(session), "exploration": str(exploration)}
    options = dict(programmed_hold_last=programmed, manual_tared=manual_tared,
                   subtract_recorded_baseline=cfg["profile"] == "airbot_native_tared_offline",
                   confirm_same_setup=args.confirm_same_setup)
    return cfg, session, exploration, options


def run_recordings(args):
    from scripts.real_training.data import prepare
    from scripts.real_training.import_airbot import import_dataset

    cfg, session, exploration, options = recording_config(args)

    def ingest(config, *, audit_only=False):
        if config["profile"] == "airbot_sensor_calibrated_offline":
            from scripts.real_training.airbot_calibrated_data import convert_training

            return convert_training(config, resolve(args.calibration), exploration, session,
                                    audit_only=audit_only)
        return import_dataset(config, exploration, session, audit_only=audit_only, **options)

    # Audit before reserving an output directory or starting a long training run.
    with tempfile.TemporaryDirectory(prefix="recording-path-audit-") as temp:
        audit_cfg = {**cfg, "raw_data": str(Path(temp) / "raw.h5"),
                     "output_dir": str(Path(temp) / "training")}
        report = ingest(audit_cfg, audit_only=True)
    if args.stage == "inspect":
        return report
    output = new_output(args.output_dir or "runs/real_training/from_recordings/training")
    inputs = output.with_name(output.name + "_inputs")
    cfg.update(output_dir=str(output), raw_data=str(inputs / "raw.h5"),
               encoder=str(resolve(cfg["encoder"]).resolve()))
    validate_config(cfg)
    if output.exists() or writable_path(inputs).exists():
        raise FileExistsError("Recording workflow needs a new --output-dir; resume with run_config.yaml")
    output.mkdir(parents=True)
    snapshot = output / "run_config.yaml"
    with snapshot.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, allow_unicode=True, sort_keys=False)
    print(f"Run config (use for subsequent stages): {snapshot}", file=sys.stderr, flush=True)
    ingest(cfg)
    info = prepare(cfg)
    result = {"run_config": str(snapshot), "output_dir": str(output),
              "valid_windows_total": info["valid_windows_total"], "warnings": info["warnings"]}
    if args.stage == "train":
        from scripts.real_training.training import evaluate, export, train

        for label, action in (("train", train), ("evaluate", evaluate), ("export", export)):
            print(f"Stage: {label}", file=sys.stderr, flush=True)
            result[label] = action(cfg)
        result["policy"] = result["export"]["policy"]
    return result
