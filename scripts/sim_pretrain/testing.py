"""Small real-simulation software smoke test, not a contact-acceptance experiment."""

import argparse
import copy
import json

import numpy as np

from scripts.shared.common import load_config, write_json
from scripts.shared.encoder import FrozenSpongeEncoder
from scripts.shared.paths import writable_path
from scripts.sim_pretrain.experiments.collect_control_comparison import (
    collect_mode,
    initialize,
)
from scripts.sim_pretrain.learning import evaluate, export, load_data, train


def smoke_test(config, output):
    cfg = copy.deepcopy(load_config(config))
    cfg["dataset"] = {"train": 2, "validation": 1, "test": 1}
    cfg["training"]["epochs"] = 2
    folder = writable_path(output)
    folder.mkdir(parents=True, exist_ok=False)
    manifest, configs = initialize(folder, cfg)
    normal = configs["normal"]
    collection = collect_mode(normal, manifest, record_only=True)
    if not collection["complete"]:
        raise RuntimeError("Simulation smoke collection incomplete; inspect retained failure logs")
    train(normal)
    evaluation = evaluate(normal)
    export(normal)
    raw, _ = load_data(normal)
    path = folder / "normal/encoder.pt"
    a, b = FrozenSpongeEncoder(path), FrozenSpongeEncoder(path)
    encoded = a.encode(raw["test"])
    np.testing.assert_array_equal(encoded, b.encode(raw["test"]))
    if encoded.shape != (1, 5) or not np.isfinite(encoded).all():
        raise ValueError("Invalid frozen embedding")
    result = {
        "software_pipeline_passed": True,
        "hardware_connected": False,
        "source_kind": "simulation",
        "scope": "small software test, not validated wiping or a quality claim",
        "contact_acceptance_policy": "record-only for this smoke test only",
        "collection": collection,
        "epochs": 2,
        "test_mse": evaluation["test_mse"],
        "embedding_shape": list(encoded.shape),
        "roundtrip_verified": True,
        "output": str(folder),
    }
    write_json(folder / "smoke_report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sim_pretrain/pretrain_paper_mu1p2.yaml")
    parser.add_argument("--output", required=True, help="Fresh runs/sim_pretrain directory")
    args = parser.parse_args(argv)
    print(json.dumps(smoke_test(args.config, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
