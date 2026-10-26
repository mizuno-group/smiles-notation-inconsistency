"""Standalone CPU inference for the adopted Transformer–VAE posterior mean."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import torch

from smiles_notation.models.pretrained_encoder import (
    MAX_TOKEN_LENGTH,
    MODEL_WIDTH,
    PretrainedTokenizer,
    PretrainedTransformerVAEEncoder,
)

EXPECTED_TOKEN_MAPPING_SHA256 = "0504b266f67fda4acfd941a29a6231859ecee0971b7b7445aca2ee62333a9419"
EXPECTED_WEIGHT_SHA256 = {
    "enc_embedding.pth": "7d6961d93dfa9dd7dd3d48a850ccadc39ef5690ae2b6009b4b9832f8a9f7b861",
    "encoder.pth": "768231a66091965395bc0b31671fb9afa323c9cb39e8b9f9d95230b018335afc",
    "latent2mu.pth": "a4ff2fe07cd9e29df1f8b447e67a3d9ad2d48a94ad486279115c608bfb84a033",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} must be a regular file: {path}")


def validate_model_assets(token_mapping: Path, weights_dir: Path) -> dict[str, str]:
    _require_regular_file(token_mapping, "Token mapping")
    observed = sha256(token_mapping)
    if observed != EXPECTED_TOKEN_MAPPING_SHA256:
        raise ValueError(
            "SHA256 mismatch for token mapping: "
            f"expected {EXPECTED_TOKEN_MAPPING_SHA256}, got {observed}"
        )
    hashes = {"token_mapping": observed}
    for name, expected in EXPECTED_WEIGHT_SHA256.items():
        path = weights_dir / name
        _require_regular_file(path, name)
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"SHA256 mismatch for {name}: expected {expected}, got {observed}")
        hashes[name] = observed
    return hashes


def read_smiles_csv(path: Path) -> list[tuple[str, str]]:
    _require_regular_file(path, "SMILES input")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["record_id", "smiles"]:
            raise ValueError("SMILES CSV header must be exactly: record_id,smiles")
        rows = []
        seen = set()
        for line_number, row in enumerate(reader, start=2):
            record_id = (row["record_id"] or "").strip()
            smiles = (row["smiles"] or "").strip()
            if not record_id or not smiles:
                raise ValueError(
                    f"record_id and smiles must be non-empty at CSV line {line_number}"
                )
            if record_id in seen:
                raise ValueError(f"duplicate record_id at CSV line {line_number}: {record_id}")
            seen.add(record_id)
            rows.append((record_id, smiles))
    if not rows:
        raise ValueError("SMILES CSV must contain at least one record")
    return rows


def _write_latent_csv(path: Path, records: list[tuple[str, list[float]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_id", *(f"mu_{index:03d}" for index in range(MODEL_WIDTH))])
        for record_id, vector in records:
            writer.writerow([record_id, *vector])


def _temporary_path(destination: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    ) as handle:
        return Path(handle.name)


def run(
    input_csv: Path,
    token_mapping: Path,
    weights_dir: Path,
    output_csv: Path,
    manifest_path: Path,
    *,
    batch_size: int = 16,
) -> None:
    input_csv = input_csv.resolve()
    token_mapping = token_mapping.resolve()
    weights_dir = weights_dir.resolve()
    output_csv = output_csv.resolve()
    manifest_path = manifest_path.resolve()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if output_csv == manifest_path:
        raise ValueError("Output CSV and manifest must be different paths")
    for path in (output_csv, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {path.parent}")

    asset_hashes = validate_model_assets(token_mapping, weights_dir)
    rows = read_smiles_csv(input_csv)
    tokenizer = PretrainedTokenizer(token_mapping)
    sequences = []
    for record_id, smiles in rows:
        sequence = tokenizer.tokenize(smiles)
        if len(sequence) > MAX_TOKEN_LENGTH:
            raise ValueError(
                f"Tokenized input {record_id!r} has {len(sequence)} positions; "
                f"the {MAX_TOKEN_LENGTH}-token limit applies and truncation is prohibited"
            )
        sequences.append(sequence)

    torch.set_num_threads(4)
    model = PretrainedTransformerVAEEncoder().load_pretrained(weights_dir).cpu().eval()
    inferred: list[tuple[str, list[float]]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            tokens = torch.zeros(
                (len(batch_sequences), max(map(len, batch_sequences))), dtype=torch.long
            )
            for row_index, sequence in enumerate(batch_sequences):
                tokens[row_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            latent = model(tokens)
            if tuple(latent.shape) != (len(batch_sequences), MODEL_WIDTH):
                raise ValueError(
                    "Encoder returned an unexpected shape: "
                    f"expected {(len(batch_sequences), MODEL_WIDTH)}, got {tuple(latent.shape)}"
                )
            if not torch.isfinite(latent).all():
                raise ValueError("Encoder returned a nonfinite latent mean")
            for (record_id, _smiles), vector in zip(
                rows[start : start + batch_size], latent.cpu().tolist(), strict=True
            ):
                inferred.append((record_id, vector))

    output_tmp = _temporary_path(output_csv)
    manifest_tmp = _temporary_path(manifest_path)
    try:
        _write_latent_csv(output_tmp, inferred)
        output_digest = sha256(output_tmp)
        manifest = {
            "status": "PASS",
            "scope": "adopted Transformer–VAE encoder posterior-mean inference only",
            "created_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "batch_size": batch_size,
            "records": len(rows),
            "latent_dimension": MODEL_WIDTH,
            "max_token_length_allowed": MAX_TOKEN_LENGTH,
            "max_token_length_observed": max(map(len, sequences)),
            "input_sha256": sha256(input_csv),
            "model_asset_sha256": asset_hashes,
            "output_sha256": output_digest,
            "training_performed": False,
            "decoder_inference_performed": False,
        }
        manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(output_tmp, output_csv)
        os.replace(manifest_tmp, manifest_path)
    finally:
        output_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 512-dimensional posterior means with the adopted Transformer–VAE encoder."
    )
    parser.add_argument("--input-smiles", type=Path, required=True)
    parser.add_argument("--token-mapping", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    run(
        args.input_smiles,
        args.token_mapping,
        args.weights_dir,
        args.output_csv,
        args.manifest,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
