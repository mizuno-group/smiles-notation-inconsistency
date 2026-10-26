"""Strict RDKit preprocessing for molecular-translation training pairs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


class SmilesTokenizer:
    """Greedy tokenizer backed by an explicit ordered vocabulary."""

    def __init__(self, tokens: list[str]):
        if len(tokens) < 4 or tokens[:3] != ["<padding>", "<start>", "<end>"]:
            raise ValueError("Vocabulary must begin with <padding>, <start>, <end>")
        if len(tokens) != len(set(tokens)):
            raise ValueError("Vocabulary tokens must be unique")
        self.tokens = tokens
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.lexemes = sorted(tokens[3:], key=lambda token: (-len(token), token))

    @classmethod
    def from_csv(cls, path: Path) -> SmilesTokenizer:
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["token_id", "token"]:
                raise ValueError("Vocabulary header must be exactly token_id,token")
            rows = list(reader)
        if [row["token_id"] for row in rows] != [str(index) for index in range(len(rows))]:
            raise ValueError("token_id values must be contiguous and zero-based")
        return cls([row["token"] for row in rows])

    def encode(self, smiles: str, max_length: int) -> list[int]:
        if not isinstance(smiles, str) or not smiles:
            raise ValueError("SMILES must be a non-empty string")
        encoded = [1]
        remainder = smiles
        while remainder:
            token = next((item for item in self.lexemes if remainder.startswith(item)), None)
            if token is None:
                raise ValueError(f"Unknown SMILES token near {remainder!r}")
            encoded.append(self.token_to_id[token])
            remainder = remainder[len(token) :]
        encoded.append(2)
        if len(encoded) > max_length:
            raise ValueError(
                f"Tokenized SMILES length {len(encoded)} exceeds maximum {max_length}; "
                "truncation is prohibited"
            )
        return encoded


def _rdkit_chem():
    try:
        from rdkit import Chem
    except ImportError as error:
        raise RuntimeError("RDKit is required for SMILES preprocessing") from error
    return Chem


def make_translation_pair(smiles: str, *, seed: int) -> tuple[str, str]:
    """Return a randomized input and canonical target for one valid molecule."""
    if not smiles:
        raise ValueError("SMILES must be non-empty")
    chem = _rdkit_chem()
    molecule = chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    canonical = chem.MolToSmiles(molecule, canonical=True)
    order = np.arange(molecule.GetNumAtoms())
    np.random.default_rng(seed).shuffle(order)
    randomized = chem.MolToSmiles(chem.RenumberAtoms(molecule, order.tolist()), canonical=False)
    return randomized, canonical


def preprocess_csv(input_csv: Path, output_csv: Path, *, seed: int) -> int:
    """Convert ``record_id,smiles`` into explicit input/target training pairs."""
    with Path(input_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["record_id", "smiles"]:
            raise ValueError("Input CSV header must be exactly record_id,smiles")
        rows = list(reader)
    if not rows:
        raise ValueError("Input CSV is empty")
    identifiers = [row["record_id"] for row in rows]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("record_id values must be non-empty and unique")
    if Path(output_csv).exists():
        raise FileExistsError(f"Output file already exists: {output_csv}")
    output = []
    for index, row in enumerate(rows):
        randomized, canonical = make_translation_pair(row["smiles"], seed=seed + index)
        output.append((row["record_id"], randomized, canonical))
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_csv).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_id", "input_smiles", "target_smiles"])
        writer.writerows(output)
    return len(output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args(argv)
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    count = preprocess_csv(args.input_csv, args.output_csv, seed=args.seed)
    print(f"records={count}")


if __name__ == "__main__":
    main()
