"""Evaluate Figure 4 K/A-label association and writer intervention effects."""

from pathlib import Path

import pandas as pd

from smiles_notation.runtime.assets import new_output

from .analysis import (
    association_from_counts,
    compare_columns,
    counterfactual_point_metrics,
)


def run(config, context):
    sources = {}
    for name in ("contingency", "writer_pairs", "adopted_contingency", "adopted_writer_summary"):
        value = config["inputs"][name]
        if not value:
            raise ValueError(f"inputs.{name} must be specified")
        path = Path(value).expanduser()
        path = path if path.is_absolute() else context.project_root / path
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing regular input: {path}")
        sources[name] = pd.read_csv(path)

    association = association_from_counts(sources["contingency"])
    writer = counterfactual_point_metrics(sources["writer_pairs"])
    association_error = compare_columns(
        association,
        sources["adopted_contingency"],
        ["dataset", "split"],
        list(association.columns.drop(["dataset", "split"])),
    )
    writer_error = compare_columns(
        writer,
        sources["adopted_writer_summary"],
        ["dataset", "sensitivity", "intervention"],
        list(writer.columns.drop(["dataset", "sensitivity", "intervention"])),
    )
    output = new_output(config, context)
    association.to_csv(output / "association_metrics.csv", index=False)
    writer.to_csv(output / "writer_effect_point_metrics.csv", index=False)
    return {
        "status": "success",
        "output": str(output),
        "association_rows": len(association),
        "writer_effect_rows": len(writer),
        "association_max_error": association_error,
        "writer_effect_max_error": writer_error,
    }
