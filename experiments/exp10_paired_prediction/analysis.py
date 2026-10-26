"""Paired, label-stratified bootstrap of binary classifier predictions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

REQUIRED_COLUMNS = frozenset({"dataset", "condition", "record_id", "label", "prediction_score"})
CONDITIONS = frozenset({"raw", "harmonized"})


def evaluate_paired_predictions(
    predictions: pd.DataFrame,
    *,
    resamples: int,
    seed_by_dataset: Mapping[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate two representations on the same records and paired bootstrap samples.

    Each dataset is resampled by label, using a common set of indices for raw and
    harmonized predictions. AUPR is average precision (not trapezoidal PR-AUC).
    One bootstrap RNG is used per dataset, with AUROC and AUPR sharing its draws.
    """
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("prediction input has no rows")
    if predictions[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("prediction input contains missing required values")
    if set(predictions.condition.unique()) != CONDITIONS:
        raise ValueError("exactly raw and harmonized conditions are required")
    if predictions.duplicated(["dataset", "condition", "record_id"]).any():
        raise ValueError("duplicate dataset/condition/record_id keys")
    if "label_matches_pkg0107" in predictions.columns:
        checks = predictions["label_matches_pkg0107"]
        if checks.isna().any() or not checks.astype(str).str.lower().eq("true").all():
            raise ValueError("source label consistency check failed")
    datasets = sorted(predictions.dataset.unique())
    if set(seed_by_dataset) != set(datasets):
        raise ValueError("dataset seed map does not match prediction datasets")

    rows: list[dict[str, float | int | str]] = []
    checks: list[dict[str, float | str]] = []
    for dataset in datasets:
        grouped = predictions[predictions.dataset == dataset]
        raw = grouped[grouped.condition == "raw"].set_index("record_id")
        harmonized = grouped[grouped.condition == "harmonized"].set_index("record_id")
        if not raw.index.is_unique or set(raw.index) != set(harmonized.index):
            raise ValueError(f"paired records do not match: {dataset}")
        harmonized = harmonized.loc[raw.index]
        if not np.array_equal(raw.label.to_numpy(), harmonized.label.to_numpy()):
            raise ValueError(f"paired labels do not match: {dataset}")
        labels = raw.label.to_numpy()
        if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
            raise ValueError(f"both binary label classes are required: {dataset}")
        y = labels.astype(int)
        x = raw.prediction_score.to_numpy(dtype=float)
        z = harmonized.prediction_score.to_numpy(dtype=float)
        if not np.isfinite(x).all() or not np.isfinite(z).all():
            raise ValueError(f"non-finite prediction scores: {dataset}")
        neg = np.flatnonzero(y == 0)
        pos = np.flatnonzero(y == 1)
        seed = int(seed_by_dataset[dataset])
        rng = np.random.default_rng(seed)
        draws = np.empty((resamples, 2, 3))
        for index in range(resamples):
            sample = np.r_[rng.choice(neg, len(neg)), rng.choice(pos, len(pos))]
            for j, metric in enumerate((roc_auc_score, average_precision_score)):
                u = metric(y[sample], x[sample])
                v = metric(y[sample], z[sample])
                draws[index, j] = (u, v, u - v)
        for j, (name, metric) in enumerate(
            (("AUROC", roc_auc_score), ("AUPR", average_precision_score))
        ):
            raw_score = metric(y, x)
            harmonized_score = metric(y, z)
            low, high = np.quantile(draws[:, j], [0.025, 0.975], axis=0)
            rows.append(
                {
                    "dataset": dataset,
                    "metric": name,
                    "n": len(y),
                    "positive": int(y.sum()),
                    "raw": raw_score,
                    "harmonized": harmonized_score,
                    "delta": raw_score - harmonized_score,
                    "raw_low": low[0],
                    "raw_high": high[0],
                    "harm_low": low[1],
                    "harm_high": high[1],
                    "delta_low": low[2],
                    "delta_high": high[2],
                    "seed": seed,
                    "resamples": resamples,
                }
            )
        for condition, values in (("raw", x), ("harmonized", z)):
            precision, recall, _ = precision_recall_curve(y, values)
            checks.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "AP": average_precision_score(y, values),
                    "PR_trapezoid": auc(recall, precision),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(checks)


def load_dataset_seeds(reference: pd.DataFrame) -> dict[str, int]:
    """Load exactly one historical bootstrap seed per dataset."""
    required = {"dataset", "bootstrap_seed"}
    if not required.issubset(reference.columns) or reference.empty:
        raise ValueError("reference must contain dataset and bootstrap_seed")
    mapping = {}
    for dataset, frame in reference.groupby("dataset"):
        if frame.bootstrap_seed.isna().any():
            raise ValueError(f"missing bootstrap seed for {dataset}")
        seeds = frame.bootstrap_seed.unique()
        if len(seeds) != 1 or int(seeds[0]) != seeds[0]:
            raise ValueError(f"conflicting bootstrap seeds for {dataset}")
        mapping[str(dataset)] = int(seeds[0])
    return mapping


def validate_against_adopted(
    metrics: pd.DataFrame, adopted: pd.DataFrame, tolerance: float = 1e-12
) -> float:
    """Require equal datasets/metrics and adopted numerical values, not just row counts."""
    numeric = (
        "n", "positive", "raw", "harmonized", "delta", "raw_low", "raw_high",
        "harm_low", "harm_high", "delta_low", "delta_high", "seed", "resamples"
    )
    required = {"dataset", "metric", *numeric}
    if not required.issubset(metrics.columns) or not required.issubset(adopted.columns):
        raise ValueError("adopted comparison table is missing required columns")
    if (
        metrics.duplicated(["dataset", "metric"]).any()
        or adopted.duplicated(["dataset", "metric"]).any()
    ):
        raise ValueError("duplicate comparison keys")
    left = metrics.set_index(["dataset", "metric"]).sort_index()
    right = adopted.set_index(["dataset", "metric"]).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("adopted comparison keys differ")
    errors = np.abs(left[list(numeric)].to_numpy(float) - right[list(numeric)].to_numpy(float))
    if not np.isfinite(errors).all():
        raise ValueError("non-finite adopted comparison")
    max_error = float(errors.max())
    if max_error > tolerance:
        raise ValueError(f"adopted metrics differ: maximum error {max_error}")
    return max_error
