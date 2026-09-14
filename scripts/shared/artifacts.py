"""Inspect and verify immutable historical artifacts without modifying them."""

import argparse
import hashlib
import json
from collections import Counter

from .paths import ARCHIVE, ROOT, historical_file_records, read_path


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(records=None):
    records = historical_file_records() if records is None else records
    errors = []
    total = 0
    for record in records:
        path = ROOT / record["new"]
        if not path.is_file():
            errors.append({"path": record["new"], "error": "missing"})
        elif path.stat().st_size != record["size"] or file_hash(path) != record["sha256"]:
            errors.append({"path": record["new"], "error": "size/hash mismatch"})
        total += record["size"]
    return {
        "passed": not errors,
        "files": len(records),
        "bytes": total,
        "errors": errors,
        "scope": "historical bytes only; not validation of current code or hardware",
    }


def catalog():
    records = json.loads((ARCHIVE / "_migration/artifacts.json").read_text())
    removed_vendor = ARCHIVE / "_migration/discoverse.json"
    if removed_vendor.exists():
        records.extend(json.loads(removed_vendor.read_text()))
    groups = Counter()
    sizes = Counter()
    for r in records:
        parts = r["new"].split("/")
        key = "/".join(parts[:3])
        groups[key] += 1
        sizes[key] += r["size"]
    return [
        {"experiment": name, "files": groups[name], "bytes": sizes[name]} for name in sorted(groups)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "list", "resolve"))
    parser.add_argument("path", nargs="?")
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify()
    elif args.command == "list":
        result = catalog()
    else:
        if not args.path:
            parser.error("resolve requires a historical path")
        result = {"old": args.path, "resolved": str(read_path(args.path, historical=True))}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if isinstance(result, dict) and result.get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
