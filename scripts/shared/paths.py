"""Project paths and explicit, read-only historical path resolution (stdlib only)."""

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "archive"
RUNS = ROOT / "runs"
ROBOSUITE_REPO = ROOT / "scripts/sim_pretrain/robosuite"
ROBOSUITE_PACKAGE = ROBOSUITE_REPO / "robosuite"


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def project_source_files(*folders):
    """Collect owned Python sources without descending into the nested vendor tree."""
    result = set()
    for folder in folders or (ROOT / "scripts",):
        for current, directories, files in os.walk(project_path(folder)):
            directories[:] = [
                name
                for name in directories
                if not name.startswith(".")
                and name != "__pycache__"
                and Path(current) / name != ROBOSUITE_REPO
            ]
            result.update(Path(current) / name for name in files if name.endswith(".py"))
    return sorted(result)


@lru_cache(maxsize=1)
def historical_file_records():
    records = []
    for name in (
        "artifacts.json",
        "snapshot.json",
        "discoverse.json",
        "package_rename.json",
        "robosuite_relocation_sources.json",
    ):
        path = ARCHIVE / "_migration" / name
        if path.exists():
            records.extend(json.loads(path.read_text(encoding="utf-8")))
    return tuple(records)


@lru_cache(maxsize=1)
def historical_records():
    result = {}
    for record in historical_file_records():
        # Keep the original interpretation of paths shared by multiple snapshots.
        result.setdefault(record["old"], record)
    return result


def read_path(value, *, historical=False, source_sha256=None):
    """Resolve old paths without changing embedded historical metadata.

    Explicit historical=True selects the source snapshot even when current code
    occupies the same path. Missing legacy artifact paths resolve automatically.
    Directory resolution is accepted only when all mapped descendants agree.
    source_sha256 selects an exact historical version when paths were reused.
    """
    path = project_path(value)
    try:
        key = path.relative_to(ROOT).as_posix()
    except ValueError:
        return path
    relocated = ROBOSUITE_REPO / key[len("robosuite/") :] if key.startswith("robosuite/") else path
    if key == "robosuite":
        relocated = ROBOSUITE_REPO
    if source_sha256 is not None:
        for record in historical_file_records():
            if record["old"] == key and record["sha256"] == source_sha256:
                return ROOT / record["new"]
        # The caller must still verify bytes; never substitute another saved version.
        return relocated
    if path.exists() and not historical:
        return path
    records = historical_records()
    if key in records:
        return ROOT / records[key]["new"]
    prefix = key.rstrip("/") + "/"
    candidates = set()
    for old, record in records.items():
        if old.startswith(prefix):
            suffix = old[len(prefix) :]
            candidates.add(record["new"][: -len(suffix)].rstrip("/"))
    if len(candidates) == 1:
        return ROOT / candidates.pop()
    if len(candidates) > 1:
        raise ValueError(f"Historical directory spans categories; select an experiment: {value}")
    return relocated


def writable_path(value):
    original = project_path(value).absolute()
    path = original.resolve()
    if path.is_relative_to(ARCHIVE.resolve()) or any(
        original.is_relative_to(ROOT / name)
        for name in ("outputs", "data", "logs", "DISCOVERSE", "adaptive_wiping", "robosuite")
    ):
        raise PermissionError("Historical paths are read-only; use runs/<category>/<new-run>.")
    return path


def protect_archive():
    """Enforce archival immutability for Python application writes, including h5py.

    Native libraries are additionally guarded at output-directory boundaries.
    This is a project safeguard, not an OS security sandbox.
    """
    if getattr(sys, "_scripts_archive_guard", False):
        return

    def check(value):
        if isinstance(value, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(value)).absolute().resolve()
            if path.is_relative_to(ARCHIVE.resolve()):
                raise PermissionError("Archive is read-only; write a new run instead.")

    def audit(event, args):
        if event == "open":
            _, mode, flags = args
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                check(args[0])
        elif event in ("os.remove", "os.rmdir", "os.mkdir", "os.truncate", "os.chmod", "os.utime"):
            check(args[0])
        elif event in ("os.rename", "os.link", "os.symlink"):
            check(args[0])
            check(args[1])

    sys.addaudithook(audit)
    sys._scripts_archive_guard = True


protect_archive()
