"""Resolve one explicitly selected policy and its hash-bound training inputs."""

import copy
import json
from pickle import UnpicklingError

from scripts.real_training.config import load_config, resolve, training_contract, validate_config
from scripts.real_training.data import load_prepared
from scripts.shared.common import file_digest
from scripts.shared.paths import project_source_files
from scripts.shared.policy import OfflinePolicy


def policy_file(value):
    path = resolve(value).resolve()
    return path / "policy.pt" if path.is_dir() else path


def _load_policy(path):
    try:
        return OfflinePolicy(path)
    except (UnpicklingError, EOFError) as exc:
        raise ValueError(f"Not a supported exported policy: {path}") from exc


def policy_mode(value, *, replay=False):
    policy = _load_policy(policy_file(value))
    if policy.source_kind != "real":
        raise ValueError("Synthetic policies cannot enter real deployment")
    profile = policy.metadata["training_contract"]["profile"]
    derivation = policy.metadata["data"]["metadata"].get("derivation")
    if profile == "airbot_native_tared_offline":
        if derivation == "manual_recorded_baseline_10s_v1":
            return "manual-tared"
        if derivation == "programmed_hold_last_10s_v1":
            return "fixed-setup"
    if profile == "airbot_sensor_calibrated_offline" or (replay and profile == "airbot_native_offline"):
        return "calibrated"
    raise ValueError(f"No real deployment adapter for policy profile/derivation: {profile}/{derivation}")


def training_inputs(value):
    path = policy_file(value)
    fingerprint = file_digest(path)
    policy = _load_policy(path)
    if policy.source_kind != "real":
        raise ValueError("Synthetic policies cannot enter real deployment")
    bindings = {str(path): fingerprint}
    cfg = copy.deepcopy(policy.metadata.get("training_config"))
    if cfg is None:
        # Older exports kept the resolved configuration beside the final checkpoint.
        run = path.parent / "final" / "run.json"
        if not run.is_file():
            raise ValueError("Legacy policy needs its sibling final/run.json; select the original training directory")
        bindings[str(run)] = file_digest(run)
        record = json.loads(run.read_text(encoding="utf-8"))
        if (record["bindings"] != policy.metadata["bindings"]
                or record["info"] != policy.metadata["data"]):
            raise ValueError("Legacy run.json does not belong to the selected policy")
        cfg = copy.deepcopy(record["config"])
    if (path.parent / "prepared.h5").is_file():
        cfg["output_dir"] = str(path.parent)
    validate_config(cfg)
    if training_contract(cfg) != policy.metadata["training_contract"]:
        raise ValueError("Selected policy training configuration mismatch")
    _, info = load_prepared(cfg)
    if info != policy.metadata["data"]:
        raise ValueError("Selected policy and prepared data differ")
    for key in ("prepared_sha256", "raw_sha256", "encoder_sha256"):
        expected = info["sha256" if key == "prepared_sha256" else key]
        if policy.metadata["bindings"].get(key) != expected:
            raise ValueError(f"Selected policy binding mismatch: {key}")
    for source, expected in bindings.items():
        if file_digest(source) != expected:
            raise ValueError("Selected policy inputs changed while loading")
    return path, policy, cfg, bindings


def _recorded_source(meta, *, session=False):
    hashes = meta["source_hashes"]
    if session:
        candidates = [p for p in hashes if resolve(p).name == "session.json"]
    else:
        demos = {r["raw_file"] for r in meta["collection_report"]["demonstrations"].values()}
        candidates = [p for p in hashes if p.endswith(".jsonl") and p not in demos]
    if len(candidates) != 1:
        raise ValueError("Policy must identify exactly one collection session and one exploration log")
    original = candidates[0]
    path = resolve(original).resolve()
    if file_digest(path) != hashes[original]:
        raise ValueError(f"Selected policy recording changed: {path}")
    return path, hashes[original]


def select_setup(spec, value=None):
    """Return effective settings without rewriting the installation configuration."""
    spec = copy.deepcopy(spec)
    if value is None:
        path = resolve(spec["training_config"])
        return spec, load_config(path), {str(path): file_digest(path)}
    path, policy, cfg, bindings = training_inputs(value)
    meta = policy.metadata["data"]["metadata"]
    session, session_hash = _recorded_source(meta, session=True)
    exploration, exploration_hash = _recorded_source(meta)
    spec.update(policy=str(path), policy_sha256=bindings[str(path)],
                training_config=None, resolved_training_config=cfg,
                collection_session=str(session.parent), collection_session_sha256=session_hash,
                exploration=str(exploration), exploration_sha256=exploration_hash)
    return spec, cfg, bindings


def bind_runtime_software(spec, policy, bindings):
    # Training source hashes are provenance, not runtime dependencies. Bind the
    # current implementation for preflight/replay and recheck it before connection.
    current = {str(p): file_digest(p) for p in project_source_files(
        "scripts/real_training", "scripts/shared", "scripts/real_deploy",
        "scripts/robot_control", "scripts/force_sensor")}
    bindings.update(current)
    spec["training_software_differences"] = [
        name for name, expected in policy.metadata["bindings"]["software"]["sources"].items()
        if current.get(str(resolve(name))) != expected
    ]
