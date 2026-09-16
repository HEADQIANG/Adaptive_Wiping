"""Publish and restore the local AIRBOT adaptation of the pinned robosuite checkout."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.shared.paths import ROOT, ROBOSUITE_REPO

PATCH = ROOT / "patches/robosuite-airbot.patch"


def git(repo, *args, allowed=(0,)):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if result.returncode not in allowed:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def sha(data):
    return hashlib.sha256(data).hexdigest()


def export_patch(repo=ROBOSUITE_REPO, patch=PATCH):
    """Generate a patch and inventory, including locally added meshes and sources."""
    base = git(repo, "rev-parse", "HEAD").decode().strip()
    modified = git(repo, "diff", "HEAD", "--name-only", "-z").decode().split("\0")
    added = git(repo, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    names = sorted({name for name in modified + added if name})
    if not names:
        raise ValueError("No local robosuite adaptation to export")
    for name in names:
        if not (repo / name).is_file() or (repo / name).is_symlink():
            raise ValueError(f"Only regular added/modified adaptation files are supported: {name}")
    chunks = [git(repo, "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--")]
    for name in sorted(n for n in added if n):
        chunks.append(git(repo, "diff", "--no-index", "--binary", "--no-ext-diff", "--no-textconv",
                          "--", "/dev/null", name, allowed=(1,)))
    payload = b"".join(chunks)
    manifest = dict(schema_version=1, base_commit=base, patch_sha256=sha(payload),
                    files=[dict(path=name, sha256=sha((repo / name).read_bytes())) for name in names])
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_bytes(payload)
    patch.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return dict(files=len(names), bytes=len(payload), base_commit=base, patch=str(patch))


def restore_patch(*, apply=False, repo=ROBOSUITE_REPO, patch=PATCH):
    if not (repo / ".git").exists():
        raise ValueError("Initialize first: git submodule update --init scripts/sim_pretrain/robosuite")
    manifest = json.loads(patch.with_suffix(".json").read_text())
    if manifest["schema_version"] != 1 or sha(patch.read_bytes()) != manifest["patch_sha256"]:
        raise ValueError("Adaptation patch integrity mismatch")
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    if head != manifest["base_commit"]:
        raise ValueError(f"Expected pinned robosuite {manifest['base_commit']}; found {head}")

    def mismatches():
        return [r["path"] for r in manifest["files"]
                if not (repo / r["path"]).is_file()
                or sha((repo / r["path"]).read_bytes()) != r["sha256"]]

    missing = mismatches()
    changed = False
    if missing and apply:
        # All-or-nothing precheck; a partial or different local adaptation is never forced.
        git(repo, "apply", "--check", str(patch.resolve()))
        git(repo, "apply", str(patch.resolve()))
        changed = True
        missing = mismatches()
    if missing:
        raise ValueError(f"Adaptation files missing or different: {', '.join(missing)}")
    return dict(passed=True, applied=changed, files=len(manifest["files"]), base_commit=head)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("apply", "verify", "export"))
    args = parser.parse_args(argv)
    try:
        result = export_patch() if args.command == "export" else restore_patch(apply=args.command == "apply")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
