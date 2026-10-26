"""Grouped OOF evaluation and input-integrity tests."""

import pandas as pd
import pytest

from experiments.exp40_encoder_comparison.analysis import calculate_metrics, evaluate_saved_oof


def make_data(root):
    adopted = []
    for encoder in ("molformer", "legacy_shared"):
        splits, predictions = [], []
        for dataset in ("BBBP", "ClinTox"):
            for i in range(10):
                record = f"{dataset}:{i:03d}"
                group = f"{dataset}_g{i}"
                label = i % 2
                splits.append(dict(seed=3, dataset=dataset, fold=i % 5,
                                   record_id=record, molecule_group_id=group, split="test"))
                for regime in ("original", "balanced_KA"):
                    score = 0.8 if label else 0.2
                    shift = (0.05 if encoder == "molformer" else 0.1)
                    difference = shift if regime == "original" else shift / 2
                    predictions.append(dict(
                        record_id=record, molecule_group_id=group, label=label,
                        p0=score, p1=score + difference, p2=score - difference,
                        dataset=dataset, seed=3, fold=i % 5,
                        model="xgboost", regime=regime,
                    ))
        folder = root / encoder / "results"
        folder.mkdir(parents=True)
        pd.DataFrame(splits).to_csv(folder / "splits.csv.gz", index=False)
        p = pd.DataFrame(predictions)
        p.to_csv(folder / "oof_predictions.csv.gz", index=False)
        g = p.groupby(
            ["seed", "dataset", "model", "regime", "molecule_group_id"]
        ).agg(label=("label", "first"), p0=("p0", "mean"),
              p1=("p1", "mean"), p2=("p2", "mean")).reset_index()
        m = calculate_metrics(g)
        m.to_csv(folder / "metrics.csv", index=False)
        m["encoder"] = encoder
        adopted.append(m)
    path = root / "adopted.csv"
    pd.concat(adopted, ignore_index=True).to_csv(path, index=False)
    return path


def test_recalculation_from_saved_predictions(tmp_path):
    adopted = make_data(tmp_path)
    metrics, summary, bootstrap, audit = evaluate_saved_oof(
        tmp_path, adopted_metrics=adopted, bootstrap_resamples=12, bootstrap_seed=20260921, oof_seed=3,
    )
    assert len(metrics) == 8
    assert len(summary) == 8
    assert len(bootstrap) == 4
    assert audit["adopted_max_error"] < 1e-12
    assert audit["identical_encoder_splits"] is True


def test_missing_adopted_metrics_fails(tmp_path):
    make_data(tmp_path)
    (tmp_path / "adopted.csv").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_saved_oof(
            tmp_path, adopted_metrics=tmp_path / "adopted.csv",
            bootstrap_resamples=3, bootstrap_seed=20260921, oof_seed=3,
        )


def test_rejects_different_encoder_splits(tmp_path):
    make_data(tmp_path)
    path = tmp_path / "legacy_shared" / "results" / "splits.csv.gz"
    frame = pd.read_csv(path)
    frame.loc[0, "fold"] = 4
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="different OOF splits"):
        evaluate_saved_oof(tmp_path, adopted_metrics=tmp_path / "adopted.csv", bootstrap_resamples=3, bootstrap_seed=20260921, oof_seed=3)


def test_rejects_fold_leakage(tmp_path):
    make_data(tmp_path)
    path = tmp_path / "molformer" / "results" / "splits.csv.gz"
    frame = pd.read_csv(path)
    frame.loc[1, "molecule_group_id"] = frame.loc[0, "molecule_group_id"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="crosses folds"):
        evaluate_saved_oof(tmp_path, adopted_metrics=tmp_path / "adopted.csv", bootstrap_resamples=3, bootstrap_seed=20260921, oof_seed=3)


def test_rejects_invalid_probability(tmp_path):
    make_data(tmp_path)
    path = tmp_path / "molformer" / "results" / "oof_predictions.csv.gz"
    frame = pd.read_csv(path)
    frame.loc[0, "p0"] = 1.5
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="invalid probability"):
        evaluate_saved_oof(tmp_path, adopted_metrics=tmp_path / "adopted.csv", bootstrap_resamples=3, bootstrap_seed=20260921, oof_seed=3)


def test_rejects_inconsistent_recorded_metrics(tmp_path):
    make_data(tmp_path)
    path = tmp_path / "adopted.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "AUROC_original"] = 0.1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="adopted cross-encoder metrics do not agree"):
        evaluate_saved_oof(tmp_path, adopted_metrics=tmp_path / "adopted.csv", bootstrap_resamples=3, bootstrap_seed=20260921, oof_seed=3)


def test_rejects_missing_oof(tmp_path):
    make_data(tmp_path)
    (tmp_path / "molformer" / "results" / "oof_predictions.csv.gz").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_saved_oof(tmp_path, adopted_metrics=tmp_path / "adopted.csv", bootstrap_resamples=3, bootstrap_seed=20260921, oof_seed=3)
