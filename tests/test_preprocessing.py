"""Contracts for strict RDKit molecular-translation preprocessing."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from smiles_notation.preprocessing import make_translation_pair, preprocess_csv  # noqa: E402


def test_preprocess_csv_preserves_every_valid_record(tmp_path: Path):
    source = tmp_path / "molecules.csv"
    source.write_text("record_id,smiles\nr1,CCO\nr2,CCl\n", encoding="utf-8")
    output = tmp_path / "pairs.csv"
    assert preprocess_csv(source, output, seed=17) == 2
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["record_id"] for row in rows] == ["r1", "r2"]
    assert all(row["input_smiles"] and row["target_smiles"] for row in rows)


def test_preprocessing_rejects_invalid_smiles_instead_of_skipping():
    with pytest.raises(ValueError, match="could not parse"):
        make_translation_pair("not a molecule", seed=1)

