"""Train and validate a new Transformer–VAE molecular-translation model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from smiles_notation.models.transformer_vae import ModelConfig, TransformerVAE
from smiles_notation.preprocessing import SmilesTokenizer


@dataclass(frozen=True)
class TrainingConfig:
    """Strict runtime and optimization settings."""

    device: str = "cpu"
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    beta: float = 1e-3
    seed: int = 111

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be exactly 'cpu' or 'cuda'")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if self.epochs < 1 or self.batch_size < 1 or self.seed < 0:
            raise ValueError("epochs and batch_size must be positive; seed must be non-negative")
        if self.learning_rate <= 0 or self.beta < 0:
            raise ValueError("learning_rate must be positive and beta non-negative")


def _read_pairs(path: Path, tokenizer: SmilesTokenizer, max_length: int):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["record_id", "input_smiles", "target_smiles"]:
            raise ValueError(
                "Training CSV header must be exactly record_id,input_smiles,target_smiles"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Training CSV is empty: {path}")
    identifiers = [row["record_id"] for row in rows]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("record_id values must be non-empty and unique")
    return [
        (
            tokenizer.encode(row["input_smiles"], max_length),
            tokenizer.encode(row["target_smiles"], max_length),
        )
        for row in rows
    ]


def _pad(sequences: list[list[int]], width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [sequence + [0] * (width - len(sequence)) for sequence in sequences],
        dtype=torch.long,
        device=device,
    )


def _batches(pairs, batch_size: int, device: torch.device):
    for start in range(0, len(pairs), batch_size):
        subset = pairs[start : start + batch_size]
        width = max(max(len(source), len(target)) for source, target in subset)
        yield (
            _pad([source for source, _ in subset], width, device),
            _pad([target for _, target in subset], width, device),
        )


def _evaluate(model, pairs, config: TrainingConfig, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for source, target in _batches(pairs, config.batch_size, device):
            losses.append(float(model(source, target).loss(config.beta)))
    if not losses or not all(math.isfinite(value) for value in losses):
        raise ValueError("Validation produced no finite losses")
    return float(np.mean(losses))


def train_from_csv(
    training_csv: Path,
    *,
    validation_csv: Path,
    vocabulary_csv: Path,
    output_dir: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> dict:
    """Train from explicit CSV inputs and save a self-describing checkpoint."""
    model_config.validate()
    training_config.validate()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    tokenizer = SmilesTokenizer.from_csv(vocabulary_csv)
    if len(tokenizer.tokens) != model_config.vocab_size:
        raise ValueError("Vocabulary size does not match model_config.vocab_size")
    training_pairs = _read_pairs(training_csv, tokenizer, model_config.max_length)
    validation_pairs = _read_pairs(validation_csv, tokenizer, model_config.max_length)

    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    device = torch.device(training_config.device)
    model = TransformerVAE(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    generator = random.Random(training_config.seed)
    epoch_losses = []
    validation_losses = []
    for _epoch in range(training_config.epochs):
        order = list(range(len(training_pairs)))
        generator.shuffle(order)
        shuffled = [training_pairs[index] for index in order]
        model.train()
        losses = []
        for source, target in _batches(shuffled, training_config.batch_size, device):
            optimizer.zero_grad(set_to_none=True)
            loss = model(source, target).loss(training_config.beta)
            if not torch.isfinite(loss):
                raise ValueError("Training produced a non-finite loss")
            loss.backward()
            gradients = [
                parameter.grad for parameter in model.parameters() if parameter.requires_grad
            ]
            if any(value is None or not torch.isfinite(value).all() for value in gradients):
                raise ValueError("Training produced a missing or non-finite gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_losses.append(float(np.mean(losses)))
        validation_losses.append(_evaluate(model, validation_pairs, training_config, device))

    output_dir.mkdir(parents=True)
    checkpoint = {
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "completed_epochs": training_config.epochs,
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    report = {
        "training_examples": len(training_pairs),
        "validation_examples": len(validation_pairs),
        "epochs": training_config.epochs,
        "device": training_config.device,
        "training_loss": epoch_losses,
        "validation_loss": validation_losses,
        "checkpoint": "checkpoint.pt",
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--vocabulary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args(argv)
    train_from_csv(
        args.train_csv,
        validation_csv=args.validation_csv,
        vocabulary_csv=args.vocabulary_csv,
        output_dir=args.output_dir,
        model_config=ModelConfig(),
        training_config=TrainingConfig(
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            beta=args.beta,
            seed=args.seed,
        ),
    )


if __name__ == "__main__":
    main()
