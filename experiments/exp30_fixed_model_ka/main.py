"""Recompute strict fixed-classifier K/A effects from saved held-out scores."""

from pathlib import Path

import pandas as pd

from smiles_notation.runtime.assets import new_output

from .analysis import compare_adopted, evaluate_strict_ka


def run(config, context):
    paths = {}
    for name in ("predictions", "adopted_effects"):
        value = config["inputs"][name]
        if not value:
            raise ValueError(f"inputs.{name} is required")
        path = Path(value).expanduser()
        path = path if path.is_absolute() else context.project_root / path
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing input: {path}")
        paths[name] = path
    actual = evaluate_strict_ka(
        pd.read_csv(paths["predictions"], float_precision="round_trip"),
        seed=config["bootstrap"]["seed"],
        resamples=config["bootstrap"]["resamples"],
        expected_counts={"BBBP": 313, "ClinTox": 205},
    )
    error = compare_adopted(actual, pd.read_csv(paths["adopted_effects"]))
    output = new_output(config, context)
    actual.to_csv(output / "strict_ka_effects.csv", index=False)
    return {
        "status": "success",
        "output": str(output),
        "rows": len(actual),
        "adopted_max_error": error,
    }
