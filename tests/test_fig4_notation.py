"""Focused Figure 4 scientific and fail-first tests."""

import pandas as pd
import pytest

from experiments.exp20_notation_analysis.analysis import (
    association_from_counts,
    counterfactual_point_metrics,
)


def counts():
    rows = []
    for dataset in ("BBBP", "ClinTox"):
        for split, values in (("train", (6, 0, 3, 19)), ("test", (2, 0, 2, 7)),
                              ("all", (8, 0, 5, 26))):
            rows.append(dict(dataset=dataset, split=split,
                             **dict(zip(("ka_pos", "ka_neg", "nonka_pos", "nonka_neg"), values, strict=True))))
    return pd.DataFrame(rows)


def writer():
    rows = []
    for dataset, n in (("BBBP", 178), ("ClinTox", 17)):
        for i in range(n):
            k, a, o = (.8 if i % 2 else .3), .4, (.7 if i % 2 else .2)
            rows.append(dict(
                record_id=f"{dataset}:{i}", dataset=dataset, split="test",
                primary_pair_accepted=True, same_connectivity_writer_pair=True,
                same_full_stereo_writer_pair=True, test_identity_seen_in_train=i < 4,
                p_original_reencoded=o, p_writer_kekule=k, p_writer_aromatic=a,
                delta_primary_k_minus_a=k-a, delta_secondary_original_minus_a=o-a,
                delta_writer_original_minus_k=o-k, class_original=int(o >= .5),
                class_writer_k=int(k >= .5), class_writer_a=int(a >= .5),
            ))
    return pd.DataFrame(rows)


def test_association():
    result = association_from_counts(counts())
    assert len(result) == 6
    assert result.haldane_odds_ratio.gt(1).all()


def test_writer():
    result = counterfactual_point_metrics(writer())
    assert len(result) == 12


@pytest.mark.parametrize("issue", ("bad_total", "duplicate", "negative", "missing"))
def test_invalid_counts(issue):
    data = counts()
    if issue == "bad_total":
        data.loc[data.split == "all", "ka_pos"] += 1
    elif issue == "duplicate":
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    elif issue == "negative":
        data.loc[0, "ka_pos"] = -1
    else:
        data = data.drop(columns=["ka_neg"])
    with pytest.raises(ValueError):
        association_from_counts(data)


@pytest.mark.parametrize("issue", ("missing", "duplicate", "invalid_prob", "bad_delta", "unverified"))
def test_invalid_writer(issue):
    data = writer()
    if issue == "missing":
        data = data.drop(columns=["p_writer_aromatic"])
    elif issue == "duplicate":
        data.loc[1, "record_id"] = data.loc[0, "record_id"]
    elif issue == "invalid_prob":
        data.loc[0, "p_writer_kekule"] = 1.2
    elif issue == "bad_delta":
        data.loc[0, "delta_primary_k_minus_a"] = 100
    else:
        data.loc[0, "same_full_stereo_writer_pair"] = False
    with pytest.raises(ValueError):
        counterfactual_point_metrics(data)

def test_empty_identity_unique_population_fails_first():
    data = writer()
    data.loc[data.dataset == "ClinTox", "test_identity_seen_in_train"] = True
    with pytest.raises(ValueError, match="empty counterfactual population"):
        counterfactual_point_metrics(data)
