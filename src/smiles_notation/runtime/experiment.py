"""Reusable contracts for experiment execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")


@dataclass(frozen=True)
class ExperimentContext:
    """Repository and output context supplied to an experiment entrypoint."""

    project_root: Path
    experiment_dir: Path
    experiment_name: str
    config_path: Path
    results_root: Path
    overrides: tuple[str, ...]

    def result_dir(self, run_id: str) -> Path:
        """Return the run directory after validating its portable identifier."""

        if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(f"run_id must match [a-z0-9][a-z0-9._-]*; got {run_id!r}")
        root = self.results_root.resolve()
        candidate = (root / self.experiment_name / run_id).resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError(f"run directory escapes results root: {candidate}")
        return candidate
