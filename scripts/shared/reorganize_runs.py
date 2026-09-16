"""Audit and perform the 2026-09-16 runs migration; cleanup remains recoverable."""

import argparse
import fcntl
import hashlib
import json
import os
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from scripts.shared.paths import ROOT
from scripts.shared.run_layout import CATEGORIES, SIM_DATA_FILES, relocated_run_path

MANIFEST = ROOT / "runs/organization_20260916.json"


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def incomplete_targets():
    targets = {}
    demos = ROOT / "runs/real_training/programmed_demonstrations"
    for session in sorted(demos.glob("*/session.json")):
        accepted = list(session.parent.glob("demo_*.json"))
        if len(accepted) < 8:
            targets[session.parent] = f"incomplete programmed session: {len(accepted)}/8 accepted demonstrations"
            continue
        used = {json.loads(p.read_text())["raw_file"] for p in accepted}
        for attempt in session.parent.glob("attempt_*.jsonl"):
            if attempt.name not in used and not any(r.get("event") == "finished" for r in rows(attempt)):
                targets[attempt] = "unaccepted demonstration attempt without finished event"
    for path in (ROOT / "runs/real_training/real_robot").rglob("*.jsonl"):
        if not any(r.get("event") == "sample" for r in rows(path)):
            targets[path] = "exploration aborted before any sample"
    for path in (ROOT / "runs/real_deploy").rglob("events.jsonl"):
        events = Counter(r.get("event") for r in rows(path))
        if not events["policy_complete"]:
            targets[path.parent] = f"deployment policy incomplete: {events['sample']} samples, no policy_complete"
    prepared_only = ROOT / "runs/real_training/training_programmed_hold_last_004"
    if prepared_only.is_dir() and {p.name for p in prepared_only.iterdir()} <= {
        ".real_training.lock", "prepared.h5", "prepared_integrity.json"
    }:
        targets[prepared_only] = "abandoned training preparation: no model, history or completed status"
    for cache in (ROOT / "runs").rglob("__pycache__"):
        targets[cache] = "regenerable Python bytecode cache"
    return targets


def plan():
    if MANIFEST.exists():
        raise FileExistsError(f"Migration already recorded; use verify: {MANIFEST}")
    recovery = Path(".runs_cleanup") / datetime.now().strftime("%Y%m%d_%H%M%S")
    targets = incomplete_targets()
    records = []
    for path in sorted((ROOT / "runs").rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unexpected pre-migration link: {path}")
        if not path.is_file():
            continue
        reason = next((why for target, why in targets.items()
                       if path == target or path.is_relative_to(target)), None)
        old = path.relative_to(ROOT)
        new = recovery / old if reason else relocated_run_path(path, ROOT).relative_to(ROOT)
        st = path.stat()
        records.append(dict(old=str(old), new=str(new), size=st.st_size,
                            mtime_ns=st.st_mtime_ns, sha256=sha(path),
                            action="cleanup" if reason else "retain", reason=reason))
    return dict(schema_version=1, created_at=datetime.now().isoformat(),
                state="planned", categories=CATEGORIES, recovery=str(recovery), records=records,
                cleanup_targets=[dict(path=str(p.relative_to(ROOT)), reason=r) for p, r in targets.items()],
                links=[])


def summary(manifest):
    removed = [r for r in manifest["records"] if r["action"] == "cleanup"]
    return dict(state=manifest["state"], original_files=len(manifest["records"]),
                retained_files=len(manifest["records"]) - len(removed),
                moved_files=sum(r["action"] == "retain" and r["old"] != r["new"] for r in manifest["records"]),
                cleanup_files=len(removed), cleanup_bytes=sum(r["size"] for r in removed),
                recovery=manifest["recovery"], cleanup_targets=manifest["cleanup_targets"])


def save(manifest):
    # This is a generated audit artifact, not a source/configuration rewrite.
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)


def verify(manifest):
    errors = []
    for record in manifest["records"]:
        path = ROOT / record["new"]
        if not path.is_file() or path.stat().st_size != record["size"] or sha(path) != record["sha256"]:
            errors.append(record["new"])
    for link in manifest["links"]:
        path = ROOT / link["path"]
        if not path.is_symlink() or os.readlink(path) != link["target"] or not path.exists():
            errors.append(link["path"])
    return dict(passed=not errors, files=len(manifest["records"]), errors=errors)


def apply(manifest):
    with ExitStack() as stack:
        for path in (ROOT / "runs").rglob("*.lock"):
            stream = stack.enter_context(path.open("rb"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for record in manifest["records"]:
            old, new = ROOT / record["old"], ROOT / record["new"]
            if old.stat().st_mtime_ns != record["mtime_ns"] or sha(old) != record["sha256"]:
                raise RuntimeError(f"File changed during inventory; stop writers and retry: {old}")
            if old != new and (new.exists() or new.is_symlink()):
                raise FileExistsError(new)
        observed = {str(p.relative_to(ROOT)) for p in (ROOT / "runs").rglob("*") if p.is_file()}
        if observed != {r["old"] for r in manifest["records"]}:
            raise RuntimeError("runs changed during inventory; stop writers and retry")
        sdk = ROOT / "logs"
        if not sdk.is_symlink() or os.readlink(sdk) != "runs/robot_control/sdk_logs":
            raise ValueError("Unexpected SDK logs path; refusing to replace it")
        manifest["state"] = "moving"
        save(manifest)
        for record in manifest["records"]:
            old, new = ROOT / record["old"], ROOT / record["new"]
            if old != new:
                new.parent.mkdir(parents=True, exist_ok=True)
                old.rename(new)
        for category in CATEGORIES:
            (ROOT / "runs" / category).mkdir(exist_ok=True)
        train = ROOT / "runs/sim_training/pretrain_wide_1200_v1"
        data = ROOT / "runs/sim_data/pretrain_wide_1200_v1"

        def link(path, target):
            relative = os.path.relpath(target, path.parent)
            path.symlink_to(relative)
            manifest["links"].append(dict(path=str(path.relative_to(ROOT)), target=relative))

        for name in SIM_DATA_FILES:
            link(train / name, data / name)
        link(data / "manifest.json", train / "manifest.json")
        # Keep the original configuration hash valid for existing checkpoints.
        snapshot = train / "run_config.json"
        saved_config = json.loads((train / "manifest.json").read_text())["config"]
        with snapshot.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(saved_config, indent=2) + "\n")
        # Atomically replace only the validated SDK symlink; no SDK log is removed.
        replacement = ROOT / ".runs_sdk_logs_link"
        link(replacement, ROOT / "runs/real_deploy/robot_control/sdk_logs")
        replacement.replace(sdk)
        manifest["links"][-1]["path"] = "logs"
        for folder in sorted((ROOT / "runs").rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if folder.is_dir() and not folder.is_symlink() and folder.name not in CATEGORIES:
                try:
                    folder.rmdir()
                except OSError:
                    pass
        manifest["state"] = "complete"
        save(manifest)
    result = verify(manifest)
    if not result["passed"]:
        raise RuntimeError(f"Migration verification failed: {result}")
    return {**summary(manifest), "verification": result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "apply", "verify"))
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify(json.loads(MANIFEST.read_text()))
    else:
        manifest = plan()
        result = summary(manifest) if args.command == "plan" else apply(manifest)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if result.get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
