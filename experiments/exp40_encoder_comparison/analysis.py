"""Evaluate molecular notation sensitivity from grouped out-of-fold predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ENCODERS = ("molformer", "legacy_shared")
SPLIT_FIELDS = ("seed", "dataset", "fold", "record_id", "molecule_group_id", "split")
PRED_FIELDS = (
    "seed", "dataset", "fold", "record_id", "molecule_group_id", "label",
    "p0", "p1", "p2", "model", "regime",
)
KEYS = ["seed", "dataset", "model", "regime"]
METRIC_FIELDS = [
    "n_groups", "n_positive_groups", "mean_abs_KA", "mean_signed_KA", "flip_rate",
    "AUROC_original", "AP_original", "AUROC_K", "AP_K", "AUROC_A", "AP_A",
    "AUROC_KA_mean", "AP_KA_mean",
]


def _read(root: Path, encoder: str, filename: str, *, round_trip: bool = False):
    path = root / encoder / "results" / filename
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required data file is missing: {path}")
    return pd.read_csv(path, float_precision="round_trip" if round_trip else None)


def _check_splits(splits: pd.DataFrame, encoder: str) -> None:
    if not set(SPLIT_FIELDS).issubset(splits) or splits.empty:
        raise ValueError(f"{encoder}: invalid split schema")
    if splits[list(SPLIT_FIELDS)].isna().any().any():
        raise ValueError(f"{encoder}: missing split values")
    if splits.duplicated(["seed", "dataset", "record_id", "molecule_group_id"]).any():
        raise ValueError(f"{encoder}: duplicate split entries")
    records = splits.groupby(["seed", "dataset", "record_id"])
    if (records.molecule_group_id.nunique() > 1).any():
        raise ValueError(f"{encoder}: a record has multiple molecular identities")
    groups = splits.groupby(["seed", "dataset", "molecule_group_id"])
    if (groups.fold.nunique() > 1).any():
        raise ValueError(f"{encoder}: molecule group crosses folds")
    if not splits.split.eq("test").all():
        raise ValueError(f"{encoder}: only out-of-fold test rows are supported")


def _check_predictions(pred: pd.DataFrame, splits: pd.DataFrame, encoder: str) -> None:
    if not set(PRED_FIELDS).issubset(pred) or pred.empty:
        raise ValueError(f"{encoder}: invalid OOF prediction schema")
    if pred[list(PRED_FIELDS)].isna().any().any():
        raise ValueError(f"{encoder}: missing OOF values")
    if pred.duplicated(KEYS + ["record_id"]).any():
        raise ValueError(f"{encoder}: duplicate prediction keys")
    if not np.isin(pred.label.to_numpy(), (0, 1)).all():
        raise ValueError(f"{encoder}: binary labels must be zero or one")
    scores = pred[["p0", "p1", "p2"]].to_numpy(float)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError(f"{encoder}: invalid probability values")
    matched = pred.merge(
        splits,
        on=["seed", "dataset", "record_id", "molecule_group_id"],
        how="left",
        suffixes=("", "_split"),
        validate="many_to_one",
        indicator=True,
    )
    if not matched["_merge"].eq("both").all() or not matched.fold.eq(matched.fold_split).all():
        raise ValueError(f"{encoder}: predictions and OOF splits do not match")
    groups = pred.groupby(["seed", "dataset", "molecule_group_id"])
    if (groups.label.nunique() > 1).any() or (groups.fold.nunique() > 1).any():
        raise ValueError(f"{encoder}: inconsistent group labels or folds")
    records = pred.groupby(["seed", "dataset", "record_id"])
    if (records.label.nunique() > 1).any() or (records.molecule_group_id.nunique() > 1).any():
        raise ValueError(f"{encoder}: inconsistent record identity across regimes")
    subset = ["seed", "dataset", "record_id", "molecule_group_id"]
    observed = set(map(tuple, pred[subset].drop_duplicates().to_numpy()))
    expected = set(map(tuple, splits[subset].to_numpy()))
    if observed != expected:
        raise ValueError(f"{encoder}: OOF/split record populations differ")


def _group_means(pred: pd.DataFrame) -> pd.DataFrame:
    return (
        pred.groupby(KEYS + ["molecule_group_id"], sort=True)
        .agg(label=("label", "first"), p0=("p0", "mean"), p1=("p1", "mean"),
             p2=("p2", "mean"))
        .reset_index()
    )


def calculate_metrics(grouped: pd.DataFrame) -> pd.DataFrame:
    """Calculate notation-response statistics and group-level classification scores."""
    rows = []
    for key, data in grouped.groupby(KEYS, sort=True):
        y = data.label.to_numpy(int)
        if len(np.unique(y)) != 2:
            raise ValueError(f"both label classes are required: {key}")
        k = data.p1.to_numpy(float)
        a = data.p2.to_numpy(float)
        scores = {
            "original": data.p0.to_numpy(float),
            "K": k, "A": a, "KA_mean": (k + a) / 2,
        }
        row = dict(zip(KEYS, key, strict=True))
        row.update(
            n_groups=len(y),
            n_positive_groups=int(y.sum()),
            mean_abs_KA=float(np.abs(k - a).mean()),
            mean_signed_KA=float((k - a).mean()),
            flip_rate=float(((k >= 0.5) != (a >= 0.5)).mean()),
        )
        for name, values in scores.items():
            row["AUROC_" + name] = roc_auc_score(y, values)
            row["AP_" + name] = average_precision_score(y, values)
        rows.append(row)
    return pd.DataFrame(rows)[KEYS + METRIC_FIELDS]


def gap_bootstrap(grouped, encoder, *, resamples=2000, seed=20260921, oof_seed=3):
    """Paired molecular-group bootstrap for change in absolute K/A probability gap."""
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    rows = []
    subset = grouped[
        (grouped.seed == oof_seed) & (grouped.model == "xgboost")
        & grouped.regime.isin(["original", "balanced_KA"])
    ]
    expected_datasets = set(grouped.loc[grouped.model == "xgboost", "dataset"])
    if not expected_datasets or set(subset.dataset) != expected_datasets:
        raise ValueError("specified OOF seed has incomplete xgboost dataset coverage")
    for dataset, data in subset.groupby("dataset", sort=True):
        table = data.assign(gap=np.abs(data.p1 - data.p2)).pivot(
            index="molecule_group_id", columns="regime", values="gap"
        )
        if set(table.columns) != {"original", "balanced_KA"}:
            raise ValueError(f"{encoder}/{dataset}: training regimes are missing")
        difference = (table.balanced_KA - table.original).dropna().to_numpy()
        if len(difference) != len(table):
            raise ValueError("molecular group populations differ between training regimes")
        rng = np.random.default_rng(seed)
        means = np.array([
            rng.choice(difference, size=len(difference), replace=True).mean()
            for _ in range(resamples)
        ])
        rows.append({
            "encoder": encoder, "dataset": dataset, "seed": oof_seed,
            "delta_gap": float(difference.mean()),
            "lower": float(np.quantile(means, 0.025)),
            "upper": float(np.quantile(means, 0.975)),
        })
    return pd.DataFrame(rows)


def evaluate_saved_oof(
    root: Path,
    *,
    adopted_metrics: Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    oof_seed: int,
):
    """Recompute reported statistics from external OOF predictions and split tables.

    Encoder inference and classifier fitting are not performed.
    """
    splits = {name: _read(root, name, "splits.csv.gz") for name in ENCODERS}
    for name, data in splits.items():
        _check_splits(data, name)
    left = splits[ENCODERS[0]].sort_values(list(SPLIT_FIELDS)).reset_index(drop=True)
    right = splits[ENCODERS[1]].sort_values(list(SPLIT_FIELDS)).reset_index(drop=True)
    if not left[list(SPLIT_FIELDS)].equals(right[list(SPLIT_FIELDS)]):
        raise ValueError("encoders use different OOF splits")

    metrics, grouped_inputs, audit = [], {}, {}
    for encoder in ENCODERS:
        pred = _read(root, encoder, "oof_predictions.csv.gz", round_trip=True)
        _check_predictions(pred, splits[encoder], encoder)
        grouped = _group_means(pred)
        calculated = calculate_metrics(grouped)
        calculated["encoder"] = encoder
        metrics.append(calculated)
        grouped_inputs[encoder] = grouped
        audit[encoder] = {
            "prediction_rows": len(pred),
            "metric_rows": len(calculated),
        }

    all_metrics = pd.concat(metrics, ignore_index=True)
    reference = pd.read_csv(adopted_metrics)
    joined_keys = ["encoder"] + KEYS
    if not set(joined_keys + METRIC_FIELDS).issubset(reference):
        raise ValueError("adopted metrics table is incomplete")
    if reference.duplicated(joined_keys).any():
        raise ValueError("duplicate adopted metric keys")
    current = all_metrics.set_index(joined_keys).sort_index()
    expected = reference.set_index(joined_keys).sort_index()
    if not current.index.equals(expected.index):
        raise ValueError("adopted metrics use a different condition set")
    errors = np.abs(
        current[METRIC_FIELDS].to_numpy(float) -
        expected[METRIC_FIELDS].to_numpy(float)
    )
    error = float(errors.max())
    if not np.isfinite(error) or error > 1e-10:
        raise ValueError("adopted cross-encoder metrics do not agree")
    audit["adopted_max_error"] = error

    boots = [
        gap_bootstrap(
            grouped_inputs[encoder], encoder, resamples=bootstrap_resamples,
            seed=bootstrap_seed, oof_seed=oof_seed,
        )
        for encoder in ENCODERS
    ]
    fields = ["mean_abs_KA", "flip_rate", "AUROC_original",
              "AUROC_KA_mean", "AP_KA_mean"]
    summary = all_metrics.groupby(
        ["encoder", "dataset", "model", "regime"]
    )[fields].agg(["mean", "min", "max"])
    audit["identical_encoder_splits"] = True
    return all_metrics, summary, pd.concat(boots, ignore_index=True), audit
