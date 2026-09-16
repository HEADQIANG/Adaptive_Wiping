"""Allocate new CLI outputs under runs, without redirecting inputs or checkpoints."""

import copy
import re
import sys
from datetime import datetime, timedelta

from scripts.shared import paths


def timestamped(value):
    path = paths.writable_path(value)
    if not path.is_relative_to(paths.RUNS.resolve()):
        return False
    for part in path.relative_to(paths.RUNS.resolve()).parts:
        if re.fullmatch(r"\d{4}_\d{6}", part):
            try:
                datetime.strptime("2000" + part, "%Y%m%d_%H%M%S")
                return True
            except ValueError:
                pass
    return False


class RunOutputs:
    """One allocator per invocation; related files share a reserved directory."""

    def __init__(self, *, now=None):
        self.now = now or datetime.now()
        self.directories = {}

    def path(self, value, *, resume=False):
        if value is None:
            return None
        path = paths.writable_path(value)
        if (not path.is_relative_to(paths.RUNS.resolve()) or timestamped(path)
                or resume):
            return path
        if path == paths.RUNS.resolve():
            raise ValueError("Select an output below runs/<category>, not runs itself")
        parent = path.parent
        if parent not in self.directories:
            parent.mkdir(parents=True, exist_ok=True)
            candidate = self.now
            # Exclusive mkdir also handles simultaneous launches and prior-year names.
            while True:
                directory = parent / candidate.strftime("%m%d_%H%M%S")
                try:
                    directory.mkdir()
                    break
                except FileExistsError:
                    candidate += timedelta(seconds=1)
            self.directories[parent] = directory
        result = self.directories[parent] / path.name
        print(f"Output: {result}", file=sys.stderr, flush=True)
        return result


def new_output(value, *, resume=False):
    return RunOutputs().path(value, resume=resume)


def new_run_config(cfg, *, include_raw=False, resume=False):
    """Freeze relocated output paths once; subsequent stages load this snapshot.

    The snapshot lives beside the output leaf so collectors requiring an empty
    experiment directory retain their original validation and provenance rules.
    """
    if resume or timestamped(cfg["output_dir"]):
        return cfg
    outputs = RunOutputs()
    result = copy.deepcopy(cfg)
    out = outputs.path(cfg["output_dir"])
    if out == paths.writable_path(cfg["output_dir"]):
        return cfg
    result["output_dir"] = str(out)
    if include_raw:
        result["raw_data"] = str(outputs.path(cfg["raw_data"]))
    import yaml

    snapshot = out.parent / "run_config.yaml"
    with snapshot.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(result, stream, allow_unicode=True, sort_keys=False)
    print(f"Run config (use for subsequent stages): {snapshot}", file=sys.stderr, flush=True)
    return result
