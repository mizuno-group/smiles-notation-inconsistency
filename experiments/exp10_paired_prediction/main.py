"""Recompute paired held-out prediction metrics."""

from pathlib import Path

import pandas as pd

from smiles_notation.runtime.assets import new_output

from .analysis import (
    evaluate_paired_predictions,
    load_dataset_seeds,
    validate_against_adopted,
)


def run(config, context):
    paths = {}
    for name in ("predictions", "seed_reference", "adopted_metrics"):
        value = config["inputs"][name]
        if not value:
            raise ValueError(f"inputs.{name} is required")
        path = Path(value).expanduser()
        path = path if path.is_absolute() else context.project_root / path
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing input: {path}")
        paths[name] = path

    predictions = pd.read_csv(paths["predictions"], float_precision="round_trip")
    seeds = load_dataset_seeds(pd.read_csv(paths["seed_reference"]))
    result, checks = evaluate_paired_predictions(
        predictions,
        resamples=config["evaluation"]["resamples"],
        seed_by_dataset=seeds,
    )
    error = validate_against_adopted(result, pd.read_csv(paths["adopted_metrics"]))
    output = new_output(config, context)
    result.to_csv(output / "paired_prediction_metrics.csv", index=False)
    checks.to_csv(output / "average_precision_check.csv", index=False)
    return {
        "status": "success",
        "output": str(output),
        "rows": len(result),
        "adopted_max_error": error,
    }
