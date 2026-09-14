"""Central simulation assets and their byte-preserving source inventory."""

import argparse
import hashlib
import json

from .paths import ROOT, read_path

ASSETS = ROOT / "asserts"
ROBOSUITE_MODELS = ASSETS / "robosuite_models"
DISCOVERSE_INERTIA_SOURCE = ASSETS / "references/discoverse/airbot_play.xml"
MANIFEST = ASSETS / "manifest.json"


def asset_files():
    """Include all active resources, not just the robot XML, in new run provenance."""
    return [
        MANIFEST,
        *sorted(
            p
            for folder in (ROBOSUITE_MODELS, ASSETS / "references")
            for p in folder.rglob("*")
            if p.is_file()
        ),
    ]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(*, sources=False):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors = []
    for record in manifest["files"]:
        for key in ("path", "source") if sources else ("path",):
            path = (
                read_path(record[key], historical=True) if key == "source" else ROOT / record[key]
            )
            if not path.is_file():
                errors.append(f"Missing: {record[key]}")
            elif path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
                errors.append(f"Size/hash mismatch: {record[key]}")
    return dict(
        passed=not errors,
        files=len(manifest["files"]),
        bytes=sum(r["size"] for r in manifest["files"]),
        source_copies_checked=sources,
        errors=errors,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify",))
    parser.add_argument(
        "--sources",
        action="store_true",
        help="Also check vendor source evidence, including archives",
    )
    args = parser.parse_args(argv)
    result = verify(sources=args.sources)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
