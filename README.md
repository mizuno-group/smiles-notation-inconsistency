# Molecular notation can act as a shortcut in AI benchmark evaluation

## Overview

This repository accompanies the study of identity-preserving SMILES notation changes in
molecular machine learning. It provides reusable preprocessing and Transformer–VAE code,
strict inference for the adopted pretrained encoder, and experiment-specific analyses used to
verify the reported numerical results.

## Contents / Directory Structure

```text
src/smiles_notation/   preprocessing, models, training, inference, and runtime support
experiments/           experiment-local analysis code, configurations, and usage notes
scripts/               thin command-line entrypoints
data/                  distributable aggregate source data and training vocabulary
environment/           pinned analysis and model requirements
tests/                 numerical, inference, orchestration, and CPU-training checks
```

## Installation

Python 3.12 is supported.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[test]"
# Include PyTorch and RDKit for preprocessing, training, and inference:
.venv/bin/python -m pip install -e ".[model,test]"
```

## Usage

### Preprocessing

Input uses the exact header `record_id,smiles`. Invalid molecules raise errors; records are
never silently omitted.

```bash
python scripts/preprocess.py \
  --input-csv data/local/molecules.csv \
  --output-csv data/local/training_pairs.csv --seed 111
```

### Training

Training and validation files use `record_id,input_smiles,target_smiles`; the vocabulary uses
`token_id,token`. The command writes model and optimizer state, model and training configurations,
and a training report.

```bash
python scripts/train.py \
  --train-csv data/local/training_pairs.csv \
  --validation-csv data/local/validation_pairs.csv \
  --vocabulary-csv data/vocabulary/training_vocabulary.csv \
  --output-dir results/pretraining/run_001 \
  --device cpu --epochs 1 --batch-size 32
```

### Inference

Extract posterior means from a checkpoint produced by `scripts/train.py`:

```bash
python scripts/infer_checkpoint.py \
  --input-csv data/local/molecules.csv \
  --checkpoint results/pretraining/run_001/checkpoint.pt \
  --vocabulary-csv data/vocabulary/training_vocabulary.csv \
  --output-csv results/new_latent_mean.csv \
  --manifest results/new_latent_manifest.json --device cpu
```

Inference with the external pretrained encoder requires its exact token mapping and three encoder
weight files:

```bash
python scripts/infer_pretrained.py \
  --input-smiles data/local/molecules.csv \
  --token-mapping models/exact_token_mapping.csv \
  --weights-dir models/adopted_encoder \
  --output-csv results/adopted_latent_mean.csv \
  --manifest results/adopted_latent_manifest.json
```

### Analysis

The included data are sufficient to validate 19 saved aggregate facts for Figures 2--5 and
generate Supplementary Figures S6 and S7:

```bash
python scripts/run.py exp50_supplementary_figures
```

Other experiments require saved row-level inputs described in their README and config:

```bash
python scripts/run.py exp10_paired_prediction --dry-run
python scripts/run.py exp20_notation_analysis --dry-run
python scripts/run.py exp30_fixed_model_ka --dry-run
python scripts/run.py exp40_encoder_comparison --dry-run
```

## Data and pretrained models

`data/figure_source/` contains small project-derived aggregate tables. Their hashes are listed
in `data/SHA256SUMS`. Row-level benchmark inputs, third-party datasets, and pretrained weights
are not redistributed. Upstream sources include [PubChem](https://pubchem.ncbi.nlm.nih.gov/),
[MoleculeNet](https://moleculenet.org/), and
[Therapeutics Data Commons](https://tdcommons.ai/); their respective terms apply.

The strict pretrained encoder verifies these external assets before loading:

- `exact_token_mapping.csv`: `0504b266f67fda4acfd941a29a6231859ecee0971b7b7445aca2ee62333a9419`
- `enc_embedding.pth`: `7d6961d93dfa9dd7dd3d48a850ccadc39ef5690ae2b6009b4b9832f8a9f7b861`
- `encoder.pth`: `768231a66091965395bc0b31671fb9afa323c9cb39e8b9f9d95230b018335afc`
- `latent2mu.pth`: `a4ff2fe07cd9e29df1f8b447e67a3d9ad2d48a94ad486279115c608bfb84a033`

These hashes identify the assets and do not grant redistribution permission.

## Authors

- [Yosuke Kikuchi](https://github.com/KikuchiY16) — first author and primary software contributor
- [Tadahaya Mizuno](https://github.com/tadahayamiz) — corresponding author and supervision

Article authors: Yosuke Kikuchi, Yasuhiro Yoshikai, Shumpei Nemoto, Ayako Furuhama,
Takashi Yamada, Hiroyuki Kusuhara, and Tadahaya Mizuno.

## Contact

Contact Tadahaya Mizuno at [tadahaya@gmail.com](mailto:tadahaya@gmail.com).

## Citation

Use the software citation metadata provided in `CITATION.cff`.

## License

The source code is available under the MIT License. See `LICENSE`. Third-party data remain under
their respective terms.
