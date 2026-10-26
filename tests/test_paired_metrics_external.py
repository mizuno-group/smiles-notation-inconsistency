"""Dataset-independent paired bootstrap and input validation tests."""

import numpy as np
import pandas as pd
import pytest

from experiments.exp10_paired_prediction.analysis import (
    evaluate_paired_predictions,
    load_dataset_seeds,
    validate_against_adopted,
)


def _predictions():
    rows = []
    for condition, scores in (
        ("raw", [0.12, 0.81, 0.22, 0.91, 0.51, 0.67]),
        ("harmonized", [0.1, 0.72, 0.3, 0.95, 0.42, 0.64]),
    ):
        for i, (y, score) in enumerate(zip([0, 1, 0, 1, 0, 1], scores, strict=True)):
            rows.append({
                "dataset": "example", "record_id": f"r{i}", "condition": condition,
                "label": y, "prediction_score": score,
            })
    return pd.DataFrame(rows)


def test_pairing_reproducible_and_seeded():
    first, check = evaluate_paired_predictions(_predictions(), resamples=20, seed_by_dataset={"example": 17})
    second, _ = evaluate_paired_predictions(_predictions(), resamples=20, seed_by_dataset={"example": 17})
    pd.testing.assert_frame_equal(first, second)
    assert first.metric.tolist() == ["AUROC", "AUPR"]
    assert (first.n == 6).all()
    assert (first.resamples == 20).all()
    assert (check.AP <= 1).all()
    assert validate_against_adopted(first, first.copy()) == 0.0


@pytest.mark.parametrize("kind", ["missing_pair", "different_label", "duplicate", "nan_score", "one_class"])
def test_pairing_rejects_invalid_scientific_input(kind):
    data = _predictions()
    if kind == "missing_pair":
        data = data.drop(index=[6]).reset_index(drop=True)
    elif kind == "different_label":
        data.loc[6, "label"] = 1
    elif kind == "duplicate":
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    elif kind == "nan_score":
        data.loc[0, "prediction_score"] = np.nan
    elif kind == "one_class":
        data.label = 0
    with pytest.raises(ValueError):
        evaluate_paired_predictions(data, resamples=3, seed_by_dataset={"example": 17})


def test_seed_map_and_adopted_metrics_fail_closed():
    data = _predictions()
    with pytest.raises(ValueError, match="seed map"):
        evaluate_paired_predictions(data, resamples=3, seed_by_dataset={"other": 1})
    with pytest.raises(ValueError, match="conflicting"):
        load_dataset_seeds(pd.DataFrame({
            "dataset": ["example", "example"], "bootstrap_seed": [3, 4],
        }))
    metrics, _ = evaluate_paired_predictions(data, resamples=3, seed_by_dataset={"example": 17})
    expected = metrics.copy()
    expected.loc[0, "raw_high"] += 0.01
    with pytest.raises(ValueError, match="adopted metrics differ"):
        validate_against_adopted(metrics, expected)
