"""Run the included aggregate analysis through the public runner."""

import json
from pathlib import Path

from smiles_notation.runtime.runner import main


def test_included_aggregate_analysis_runs(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    assert main(["exp50_supplementary_figures", "--results-root", str(tmp_path)],
                project_root=root) == 0
    outputs = list(tmp_path.glob("exp50_supplementary_figures/*/validation_summary.json"))
    assert len(outputs) == 1
    report = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["check_count"] == 19
    assert report["numeric_check_count"] == 11
    assert report["structural_check_count"] == 8
    assert report["max_error"] == 0.0
    output = outputs[0].parent
    for name in ("supplementary_figure_s6", "supplementary_figure_s7"):
        assert (output / f"{name}.png").is_file()
        assert (output / f"{name}.svg").is_file()
