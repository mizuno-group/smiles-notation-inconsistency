"""Strict fixed-model K/A scientific contracts on small matched-score fixtures."""

import pandas as pd
import pytest

from experiments.exp30_fixed_model_ka.analysis import compare_adopted, evaluate_strict_ka


def _records():
    rows = []
    for dataset in ("BBBP", "ClinTox"):
        for i, (label, k, a) in enumerate(
            [(1, 0.9, 0.2), (0, 0.4, 0.3), (1, 0.7, 0.6), (0, 0.2, 0.4)]
        ):
            rows.append({
                "dataset": dataset,
                "record_id": f"{dataset}:{i:04d}",
                "molecule_group_id": f"{dataset}:mol{max(0, i - 1)}",
                "label": label, "historical_ka": i < 2,
                "heldout_identity_seen_train": i == 3,
                "split": "test", "p_matched_kekule": k,
                "p_matched_aromatic": a, "delta_k_minus_a": k - a,
            })
    return pd.DataFrame(rows)


def _evaluate(data):
    return evaluate_strict_ka(
        data, seed=20260919, resamples=20, expected_counts={"BBBP": 4, "ClinTox": 4},
    )


def test_grouped_bootstrap_is_deterministic_and_signed():
    first = _evaluate(_records())
    pd.testing.assert_frame_equal(first, _evaluate(_records()))
    assert len(first) == 12
    primary = first.query("scope == 'all_heldout' and stratum == 'all_eligible'")
    assert primary.mean_k_minus_a.tolist() == pytest.approx([0.175, 0.175])
    assert compare_adopted(first, first.copy()) == 0


@pytest.mark.parametrize("kind", [
    "missing_column", "duplicated_record", "bad_probability", "missing_group",
    "different_split", "bad_label", "bad_delta", "different_count",
])
def test_invalid_records_fail_first(kind):
    data = _records()
    if kind == "missing_column":
        data = data.drop(columns=["p_matched_aromatic"])
    elif kind == "duplicated_record":
        data.loc[1, "record_id"] = data.loc[0, "record_id"]
    elif kind == "bad_probability":
        data.loc[0, "p_matched_kekule"] = 1.2
    elif kind == "missing_group":
        data.loc[0, "molecule_group_id"] = None
    elif kind == "different_split":
        data.loc[0, "split"] = "train"
    elif kind == "bad_label":
        data.loc[0, "label"] = 3
    elif kind == "bad_delta":
        data.loc[0, "delta_k_minus_a"] = 0.0
    elif kind == "different_count":
        data = data.iloc[1:]
    with pytest.raises(ValueError):
        _evaluate(data)


def test_reference_mismatch_fails():
    actual = _evaluate(_records())
    reference = actual.copy()
    reference.loc[0, "mean_k_minus_a_ci_low"] += 0.01
    with pytest.raises(ValueError, match="mismatch"):
        compare_adopted(actual, reference)


def test_invalid_bootstrap_seed_fails():
    with pytest.raises(ValueError, match="seed/resamples"):
        evaluate_strict_ka(
            _records(), seed=-1, resamples=20, expected_counts={"BBBP": 4, "ClinTox": 4},
        )
