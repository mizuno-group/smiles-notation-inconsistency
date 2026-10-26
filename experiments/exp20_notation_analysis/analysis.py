"""Recalculate Figure 4 association and writer-pair effect summaries."""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm, wilcoxon

ASSOCIATION_COLUMNS = ("dataset", "split", "ka_pos", "ka_neg", "nonka_pos", "nonka_neg")
EFFECT_COLUMNS = (
    "record_id", "dataset", "split", "primary_pair_accepted",
    "same_connectivity_writer_pair", "same_full_stereo_writer_pair",
    "test_identity_seen_in_train", "p_original_reencoded", "p_writer_kekule",
    "p_writer_aromatic", "delta_primary_k_minus_a",
    "delta_secondary_original_minus_a", "delta_writer_original_minus_k",
    "class_original", "class_writer_k", "class_writer_a",
)
INTERVENTIONS = {
    "primary_writer_k_vs_a": ("delta_primary_k_minus_a", "class_writer_k", "class_writer_a"),
    "secondary_original_vs_a": ("delta_secondary_original_minus_a", "class_original", "class_writer_a"),
    "writer_only_original_vs_k": ("delta_writer_original_minus_k", "class_original", "class_writer_k"),
}


def _require(df, names):
    missing = set(names) - set(df)
    if missing or df.empty:
        raise ValueError(f"missing columns or empty input: {sorted(missing)}")
    if df[list(names)].isna().any().any():
        raise ValueError("input contains missing required values")


def _wilson(successes, n):
    z = norm.ppf(0.975)
    rate = successes / n
    divisor = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / divisor
    radius = z / divisor * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
    return center - radius, center + radius


def association_from_counts(counts):
    """Fisher exact test, Haldane OR and Wilson intervals from 2x2 count tables."""
    _require(counts, ASSOCIATION_COLUMNS)
    if counts.duplicated(["dataset", "split"]).any():
        raise ValueError("duplicate association dataset/split")
    if set(counts.dataset) != {"BBBP", "ClinTox"} or set(counts.split) != {"all", "train", "test"}:
        raise ValueError("unexpected association datasets/splits")
    rows = []
    for row in counts.itertuples(index=False):
        a, b, c, d = (getattr(row, name) for name in ASSOCIATION_COLUMNS[2:])
        values = np.asarray([a, b, c, d], dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(values >= 0) or not np.all(values == values.astype(int)):
            raise ValueError("contingency cells must be nonnegative integer counts")
        if min(a + b, a + c, b + d) == 0:
            raise ValueError("undefined conditional rate")
        adjusted = values + 0.5
        odds = adjusted[0] * adjusted[3] / (adjusted[1] * adjusted[2])
        z = norm.ppf(0.975)
        radius = z * np.sqrt(np.sum(1 / adjusted))
        purity_low, purity_high = _wilson(a, a + b)
        pos_low, pos_high = _wilson(a, a + c)
        neg_low, neg_high = _wilson(b, b + d)
        rows.append(dict(
            dataset=row.dataset, split=row.split,
            ka_pos=int(a), ka_neg=int(b), nonka_pos=int(c), nonka_neg=int(d),
            fisher_p_two_sided=fisher_exact([[a, b], [c, d]])[1],
            haldane_odds_ratio=odds,
            haldane_or_ci_low=np.exp(np.log(odds) - radius),
            haldane_or_ci_high=np.exp(np.log(odds) + radius),
            purity_p_y1_given_ka=a / (a + b), purity_ci_low=purity_low,
            purity_ci_high=purity_high, coverage_p_ka_given_y1=a / (a + c),
            coverage_pos_ci_low=pos_low, coverage_pos_ci_high=pos_high,
            negative_coverage_p_ka_given_y0=b / (b + d),
            coverage_neg_ci_low=neg_low, coverage_neg_ci_high=neg_high,
        ))
    result = pd.DataFrame(rows)
    for ds, group in result.groupby("dataset"):
        if set(group.split) != {"all", "train", "test"} or len(group) != 3:
            raise ValueError(f"incomplete association splits for {ds}")
        cols = list(ASSOCIATION_COLUMNS[2:])
        total = group.loc[group.split == "all", cols].to_numpy(dtype=int)[0]
        components = group.loc[group.split != "all", cols].sum().to_numpy(dtype=int)
        if not np.array_equal(total, components):
            raise ValueError(f"train/test association totals do not match for {ds}")
    return result


def counterfactual_point_metrics(records):
    """Recompute point effects from the 195 eligible writer-generated held-out pairs."""
    _require(records, EFFECT_COLUMNS)
    if len(records) != 195 or records.dataset.value_counts().to_dict() != {"BBBP": 178, "ClinTox": 17}:
        raise ValueError("unexpected writer-generated counterfactual population")
    if records.record_id.duplicated().any() or set(records.split) != {"test"}:
        raise ValueError("counterfactual records must be unique held-out IDs")
    for col in ("primary_pair_accepted", "same_connectivity_writer_pair", "same_full_stereo_writer_pair"):
        if not pd.api.types.is_bool_dtype(records[col]) or not records[col].all():
            raise ValueError(f"unverified K/A molecular pairing: {col}")
    if not pd.api.types.is_bool_dtype(records.test_identity_seen_in_train):
        raise ValueError("training identity flag must be boolean")
    prob_cols = ("p_original_reencoded", "p_writer_kekule", "p_writer_aromatic")
    probs = records[list(prob_cols)].to_numpy(float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("invalid fixed-classifier probability")
    comparisons = (
        ("p_writer_kekule", "p_writer_aromatic", "delta_primary_k_minus_a"),
        ("p_original_reencoded", "p_writer_aromatic", "delta_secondary_original_minus_a"),
        ("p_original_reencoded", "p_writer_kekule", "delta_writer_original_minus_k"),
    )
    for left, right, delta in comparisons:
        if not np.allclose(records[left] - records[right], records[delta], rtol=0, atol=1e-7):
            raise ValueError(f"saved writer contrast does not match scores: {delta}")
    for c, p in zip(("class_original", "class_writer_k", "class_writer_a"), prob_cols, strict=True):
        if not np.array_equal(records[c].to_numpy(), (records[p] >= 0.5).to_numpy(int)):
            raise ValueError(f"class threshold is inconsistent: {c}")
    rows = []
    for dataset, frame in records.groupby("dataset", sort=True):
        for scope, sample in (
            ("all_heldout", frame),
            ("identity_unique_heldout", frame.loc[~frame.test_identity_seen_in_train]),
        ):
            if sample.empty:
                raise ValueError(f"empty counterfactual population: {dataset}/{scope}")
            for intervention, (field, base, changed) in INTERVENTIONS.items():
                x = sample[field].to_numpy(float)
                statistic = wilcoxon(x, alternative="two-sided")
                low, high = _wilson(int((x > 0).sum()), len(x))
                rows.append(dict(
                    dataset=dataset, sensitivity=scope, intervention=intervention,
                    n=len(x), mean_delta=float(x.mean()), median_delta=float(np.median(x)),
                    proportion_delta_positive=float((x > 0).mean()),
                    proportion_positive_ci_low=low, proportion_positive_ci_high=high,
                    proportion_delta_negative=float((x < 0).mean()),
                    zero_n=int((x == 0).sum()),
                    wilcoxon_statistic=float(statistic.statistic),
                    wilcoxon_p_two_sided=float(statistic.pvalue),
                    positive_to_negative_flips=int(((sample[base] == 1) & (sample[changed] == 0)).sum()),
                    negative_to_positive_flips=int(((sample[base] == 0) & (sample[changed] == 1)).sum()),
                ))
    return pd.DataFrame(rows)


def compare_columns(actual, reference, keys, columns, *, tolerance=1e-9):
    """Require identical scientific grouping and compare only reproducible columns."""
    needed = set(keys) | set(columns)
    _require(actual, needed)
    _require(reference, needed)
    if actual.duplicated(keys).any() or reference.duplicated(keys).any():
        raise ValueError("duplicate adopted result key")
    a = actual.set_index(keys).sort_index()
    b = reference.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("adopted result populations differ")
    differences = np.abs(a[list(columns)].to_numpy(float) - b[list(columns)].to_numpy(float))
    if not np.isfinite(differences).all() or float(differences.max()) > tolerance:
        raise ValueError(f"adopted numerical result mismatch: {float(differences.max())}")
    return float(differences.max())
