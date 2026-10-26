"""Evaluate fixed-model Kekulé–aromatic sensitivity on matched molecular identities."""

from __future__ import annotations

import numpy as np
import pandas as pd

INPUT_COLUMNS = (
    "record_id", "dataset", "label", "historical_ka",
    "heldout_identity_seen_train", "split", "molecule_group_id",
    "p_matched_kekule", "p_matched_aromatic",
)
KEYS = ["dataset", "scope", "stratum"]
MEASURES = [
    "mean_k_minus_a", "fraction_k_above_a", "class_flip_fraction_at_0_5",
    "AUROC_K_minus_A", "AP_K_minus_A",
]


def _weighted_metric(y: np.ndarray, scores: np.ndarray, metric: str):
    order = np.argsort(scores, kind="stable")[::-1]
    ss = scores[order]
    starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
    yy = y[order]

    def evaluate(weights: np.ndarray) -> float:
        w = weights[order]
        pos = np.add.reduceat(w * yy, starts)
        neg = np.add.reduceat(w * (1 - yy), starts)
        tp, fp = np.cumsum(pos), np.cumsum(neg)
        if metric == "AUROC":
            return float(np.sum(pos * (neg.sum() - fp + 0.5 * neg)) / (pos.sum() * neg.sum()))
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=tp + fp > 0)
        return float(np.sum(pos * precision) / pos.sum())

    return evaluate


def _summarize(sub: pd.DataFrame, *, seed: int, resamples: int) -> dict:
    y = sub.label.to_numpy(int)
    k = sub.p_matched_kekule.to_numpy(float)
    a = sub.p_matched_aromatic.to_numpy(float)
    delta = k - a
    groups, inverse = np.unique(sub.molecule_group_id.to_numpy(str), return_inverse=True)
    rng = np.random.default_rng(seed)
    functions = (
        {metric: (_weighted_metric(y, k, metric), _weighted_metric(y, a, metric))
         for metric in ("AUROC", "AP")}
        if len(np.unique(y)) == 2 else {}
    )
    ones = np.ones(len(y))
    result = {
        "n": len(y),
        "n_molecular_groups": len(groups),
        "n_positive": int(y.sum()),
        "mean_k_minus_a": float(delta.mean()),
        "fraction_k_above_a": float((delta > 0).mean()),
        "class_flip_fraction_at_0_5": float(((k >= 0.5) != (a >= 0.5)).mean()),
    }
    samples = {name: [] for name in MEASURES[:3]}
    for metric, (fk, fa) in functions.items():
        result[metric + "_K"] = fk(ones)
        result[metric + "_A"] = fa(ones)
        result[metric + "_K_minus_A"] = fk(ones) - fa(ones)
        samples[metric + "_K_minus_A"] = []

    for _ in range(resamples):
        counts = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
        weights = counts[inverse]
        total = weights.sum()
        samples["mean_k_minus_a"].append(float(weights @ delta / total))
        samples["fraction_k_above_a"].append(float(weights @ (delta > 0) / total))
        samples["class_flip_fraction_at_0_5"].append(
            float(weights @ ((k >= 0.5) != (a >= 0.5)) / total)
        )
        if (weights @ y) > 0 and (weights @ (1 - y)) > 0:
            for metric, (fk, fa) in functions.items():
                samples[metric + "_K_minus_A"].append(fk(weights) - fa(weights))

    for name, values in samples.items():
        result[name + "_bootstrap_valid"] = len(values)
        low, high = np.quantile(values, (0.025, 0.975)) if values else (np.nan, np.nan)
        result[name + "_ci_low"] = low
        result[name + "_ci_high"] = high
    result["metric_status"] = (
        "both_classes" if functions else "single_class_AUROC_AP_not_reported"
    )
    return result


def evaluate_strict_ka(
    predictions: pd.DataFrame,
    *,
    seed: int,
    resamples: int,
    expected_counts: dict[str, int],
) -> pd.DataFrame:
    """Recalculate all-heldout and train-overlap-excluded K/A effects."""
    missing = set(INPUT_COLUMNS) - set(predictions.columns)
    if missing or predictions.empty:
        raise ValueError(f"missing required strict K/A data: {sorted(missing)}")
    if predictions[list(INPUT_COLUMNS)].isna().any().any():
        raise ValueError("missing values in strict K/A data")
    if predictions.record_id.duplicated().any():
        raise ValueError("duplicate molecular record IDs")
    if set(predictions.split) != {"test"}:
        raise ValueError("only strict K/A held-out test records are accepted")
    if set(predictions.dataset) != set(expected_counts):
        raise ValueError("strict K/A cohort dataset names differ")
    if (predictions.groupby("dataset").size().to_dict()) != expected_counts:
        raise ValueError("strict K/A cohort sizes differ")
    if not all(pd.api.types.is_bool_dtype(predictions[col])
               for col in ("historical_ka", "heldout_identity_seen_train")):
        raise ValueError("expected boolean cohort flags")
    if not predictions.label.isin([0, 1]).all():
        raise ValueError("labels must be binary")
    probs = predictions[["p_matched_kekule", "p_matched_aromatic"]].to_numpy(float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("invalid prediction probabilities")
    if type(seed) is not int or seed < 0 or type(resamples) is not int or resamples < 1:
        raise ValueError("seed/resamples must be explicit positive integer settings")
    if "delta_k_minus_a" in predictions:
        check = probs[:, 0] - probs[:, 1]
        if not np.allclose(check, predictions.delta_k_minus_a, atol=1e-12, rtol=0):
            raise ValueError("provided signed K/A difference does not match predictions")
    rows = []
    for dataset, ds in predictions.groupby("dataset", sort=True):
        for scope in ("all_heldout", "exclude_train_identity"):
            population = ds if scope == "all_heldout" else ds[~ds.heldout_identity_seen_train]
            for stratum in ("all_eligible", "historical_ka", "other"):
                sample = (
                    population if stratum == "all_eligible"
                    else population[population.historical_ka.eq(stratum == "historical_ka")]
                )
                if sample.empty:
                    raise ValueError(f"empty strict K/A population: {dataset}/{scope}/{stratum}")
                rows.append({
                    "dataset": dataset, "scope": scope, "stratum": stratum,
                    **_summarize(sample, seed=seed, resamples=resamples),
                })
    return pd.DataFrame(rows)


def compare_adopted(actual: pd.DataFrame, reference: pd.DataFrame, *, tol=1e-12) -> float:
    """Fail on any mismatched adopted estimate, interval or class-status metadata."""
    numerical = [
        "n", "n_molecular_groups", "n_positive", *MEASURES[:3],
        *(x for m in MEASURES for x in (m + "_bootstrap_valid", m + "_ci_low", m + "_ci_high")),
        "AUROC_K", "AUROC_A", "AP_K", "AP_A", "AUROC_K_minus_A", "AP_K_minus_A",
    ]
    fields = KEYS + numerical + ["metric_status"]
    if not set(fields).issubset(actual) or not set(fields).issubset(reference):
        raise ValueError("adopted strict K/A schema missing required columns")
    if actual.duplicated(KEYS).any() or reference.duplicated(KEYS).any():
        raise ValueError("duplicate strict K/A comparison keys")
    a = actual.set_index(KEYS).sort_index()
    b = reference.set_index(KEYS).sort_index()
    if not a.index.equals(b.index) or not a.metric_status.equals(b.metric_status):
        raise ValueError("strict K/A populations or class status differ")
    x, y = a[numerical].to_numpy(dtype=float), b[numerical].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(x), np.isnan(y)):
        raise ValueError("missing numerical estimates differ")
    if not np.isfinite(x[~np.isnan(x)]).all() or not np.isfinite(y[~np.isnan(y)]).all():
        raise ValueError("invalid numerical estimates")
    max_error = float(np.abs(x - y)[~np.isnan(x)].max())
    if max_error > tol:
        raise ValueError(f"adopted strict K/A numerical mismatch: {max_error}")
    return max_error
