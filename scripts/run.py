#!/usr/bin/env python3
"""Thin repository entrypoint for the reusable experiment runner."""

from pathlib import Path

from smiles_notation.runtime.runner import main

if __name__ == "__main__":
    raise SystemExit(main(project_root=Path(__file__).resolve().parents[1]))
