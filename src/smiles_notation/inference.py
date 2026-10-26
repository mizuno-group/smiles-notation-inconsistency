"""Extract posterior-mean latents from a newly trained Transformer–VAE checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from smiles_notation.models.transformer_vae import ModelConfig, TransformerVAE
from smiles_notation.preprocessing import SmilesTokenizer

CHECKPOINT_KEYS = {
    "model_config",
    "training_config",
    "model_state",
    "optimizer_state",
    "completed_epochs",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_inputs(path: Path) -> list[tuple[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Input CSV must be a regular file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["record_id", "smiles"]:
            raise ValueError("Input CSV header must be exactly record_id,smiles")
        rows = [(row["record_id"].strip(), row["smiles"].strip()) for row in reader]
    if not rows:
        raise ValueError("Input CSV is empty")
    identifiers = [identifier for identifier, _smiles in rows]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("record_id values must be non-empty and unique")
    if any(not smiles for _identifier, smiles in rows):
        raise ValueError("smiles values must be non-empty")
    return rows


def _temporary_path(destination: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    ) as handle:
        return Path(handle.name)


def extract_latent_means(
    input_csv: Path,
    *,
    checkpoint: Path,
    vocabulary_csv: Path,
    output_csv: Path,
    manifest_path: Path,
    device: str,
    batch_size: int = 32,
) -> dict[str, object]:
    """Load one self-describing training checkpoint and write posterior means."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be exactly 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    checkpoint = Path(checkpoint).resolve()
    vocabulary_csv = Path(vocabulary_csv).resolve()
    output_csv = Path(output_csv).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise FileNotFoundError(f"Checkpoint must be a regular file: {checkpoint}")
    if output_csv == manifest_path:
        raise ValueError("Output CSV and manifest must be different paths")
    for path in (output_csv, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {path.parent}")

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or set(saved) != CHECKPOINT_KEYS:
        actual = sorted(saved) if isinstance(saved, dict) else type(saved).__name__
        raise ValueError(f"Checkpoint keys must be exactly {sorted(CHECKPOINT_KEYS)}; got {actual}")
    if not isinstance(saved["model_config"], dict):
        raise TypeError("checkpoint model_config must be a mapping")
    model_config = ModelConfig(**saved["model_config"])
    model_config.validate()
    tokenizer = SmilesTokenizer.from_csv(vocabulary_csv)
    if len(tokenizer.tokens) != model_config.vocab_size:
        raise ValueError("Vocabulary size does not match checkpoint model_config.vocab_size")
    rows = _read_inputs(Path(input_csv).resolve())
    sequences = [tokenizer.encode(smiles, model_config.max_length) for _identifier, smiles in rows]

    runtime_device = torch.device(device)
    model = TransformerVAE(model_config).to(runtime_device)
    model.load_state_dict(saved["model_state"], strict=True)
    model.eval()
    inferred: list[tuple[str, list[float]]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            subset = sequences[start : start + batch_size]
            tokens = torch.zeros(
                (len(subset), max(map(len, subset))), dtype=torch.long, device=runtime_device
            )
            for row_index, sequence in enumerate(subset):
                tokens[row_index, : len(sequence)] = torch.tensor(
                    sequence, dtype=torch.long, device=runtime_device
                )
            latent = model.encode_mean(tokens)
            expected_shape = (len(subset), model_config.latent_size)
            if tuple(latent.shape) != expected_shape or not torch.isfinite(latent).all():
                raise ValueError(
                    f"Model returned invalid latent means: expected finite {expected_shape}, "
                    f"got {tuple(latent.shape)}"
                )
            for (record_id, _smiles), vector in zip(
                rows[start : start + batch_size], latent.cpu().tolist(), strict=True
            ):
                inferred.append((record_id, vector))

    output_tmp = _temporary_path(output_csv)
    manifest_tmp = _temporary_path(manifest_path)
    try:
        with output_tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["record_id", *(f"mu_{index:03d}" for index in range(model_config.latent_size))]
            )
            for record_id, vector in inferred:
                writer.writerow([record_id, *vector])
        manifest: dict[str, object] = {
            "status": "PASS",
            "scope": "newly trained Transformer–VAE posterior-mean inference",
            "device": device,
            "records": len(rows),
            "latent_size": model_config.latent_size,
            "checkpoint_sha256": _sha256(checkpoint),
            "vocabulary_sha256": _sha256(vocabulary_csv),
            "output_sha256": _sha256(output_tmp),
        }
        manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(output_tmp, output_csv)
        os.replace(manifest_tmp, manifest_path)
    finally:
        output_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    extract_latent_means(
        args.input_csv,
        checkpoint=args.checkpoint,
        vocabulary_csv=args.vocabulary_csv,
        output_csv=args.output_csv,
        manifest_path=args.manifest,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
