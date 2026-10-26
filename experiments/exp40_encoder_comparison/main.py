"""Recompute grouped cross-encoder OOF metrics."""

import json
from pathlib import Path

from smiles_notation.runtime.assets import new_output

from .analysis import evaluate_saved_oof


def run(config, context):
    paths = {}
    for name in ("oof_root", "adopted_metrics"):
        value = config["inputs"][name]
        if not value:
            raise ValueError(f"inputs.{name} is required")
        path = Path(value).expanduser()
        path = path if path.is_absolute() else context.project_root / path
        if path.is_symlink() or not (path.is_dir() if name == "oof_root" else path.is_file()):
            raise FileNotFoundError(f"invalid input: {path}")
        paths[name] = path

    evaluation = config["evaluation"]
    metrics, summary, bootstrap, audit = evaluate_saved_oof(
        paths["oof_root"],
        adopted_metrics=paths["adopted_metrics"],
        bootstrap_resamples=evaluation["bootstrap_resamples"],
        bootstrap_seed=evaluation["bootstrap_seed"],
        oof_seed=evaluation["oof_seed"],
    )
    output = new_output(config, context)
    metrics.to_csv(output / "all_metrics.csv", index=False)
    summary.to_csv(output / "five_seed_summary.csv")
    bootstrap.to_csv(output / "seed3_gap_bootstrap.csv", index=False)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return {
        "status": "success",
        "output": str(output),
        "metrics": len(metrics),
        "adopted_max_error": audit["adopted_max_error"],
    }
