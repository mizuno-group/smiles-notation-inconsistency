"""Verify adopted aggregate values and generate Supplementary Figures S6 and S7."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE_FILES = {
    "fig2": "figure_2a_reconstruction_effects.csv",
    "latent": "figure_2b_latent_summary.csv",
    "spearman": "figure_2c_string_latent_spearman.csv",
    "fig3": "figure_3_paired_recalculation.csv",
    "ka": "figure_4_ka_global_association.csv",
    "notation": "figure_4c_notation_only_performance.csv",
    "cross": "figure_5_crossencoder_metrics.csv",
    "population": "population_counts.csv",
    "s6": "supplementary_figure_s6_dataset_heterogeneity.csv",
    "s7": "supplementary_figure_s7_latent_performance.csv",
}
S6_FILE = SOURCE_FILES["s6"]
S7_FILE = SOURCE_FILES["s7"]

NUMERIC_EXPECTED = {
    "fig2_sider_connectivity_raw": 0.1923631123919308,
    "fig2_sider_connectivity_harmonized": 0.946685878962536,
    "fig4_clintox_ka_positive": 96,
    "fig4_clintox_ka_negative": 0,
    "fig4_bbbp_ka_positive": 835,
    "fig4_bbbp_ka_negative": 8,
    "fig4_clintox_auroc": 0.8863636363636364,
    "fig4_clintox_aupr": 0.7897340754483612,
    "fig4_bbbp_auroc": 0.7784455128205128,
    "fig4_bbbp_aupr": 0.8950029233819716,
    "primary_identity_valid_total": 22776,
}
STRUCTURAL_EXPECTED = {
    "fig2_rows": 18,
    "fig2_latent_summary_rows": 6,
    "fig2_string_latent_rows": 6,
    "fig3_adopted_aggregate_rows": 12,
    "fig5_metric_rows": 220,
    "population_rows": 6,
    "s6_ci_rows": 6,
    "s7_labeled_points_per_panel": 6,
}


def _read_table(source: Path, key: str, columns: set[str]) -> pd.DataFrame:
    path = source / SOURCE_FILES[key]
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing regular source table: {path}")
    table = pd.read_csv(path)
    missing = sorted(columns - set(table.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if table.empty:
        raise ValueError(f"{path.name} contains no records")
    if table[list(columns)].isna().any().any():
        raise ValueError(f"{path.name} contains missing required values")
    return table


def _one_value(table: pd.DataFrame, query: str, column: str, label: str) -> float:
    matches = table.query(query)
    if len(matches) != 1:
        raise ValueError(f"{label} must select exactly one row; selected {len(matches)}")
    return float(matches.iloc[0][column])


def _load_sources(source: Path) -> dict[str, pd.DataFrame]:
    tables = {
        "fig2": _read_table(source, "fig2", {"dataset", "metric", "raw_rate", "harmonized_rate"}),
        "latent": _read_table(source, "latent", {"dataset"}),
        "spearman": _read_table(source, "spearman", {"dataset"}),
        "fig3": _read_table(source, "fig3", {"dataset"}),
        "ka": _read_table(source, "ka", {"dataset", "split", "ka_pos", "ka_neg"}),
        "notation": _read_table(source, "notation", {"dataset", "model", "metric", "estimate"}),
        "cross": _read_table(source, "cross", {"dataset"}),
        "population": _read_table(source, "population", {"primary_identity_valid"}),
        "s6": _read_table(
            source,
            "s6",
            {
                "dataset",
                "changed_string_pct",
                "connectivity_loss_harmonized_minus_raw",
                "connectivity_loss_ci_low",
                "connectivity_loss_ci_high",
                "median_latent_rms_distance",
            },
        ),
        "s7": _read_table(
            source,
            "s7",
            {
                "dataset",
                "median_latent_rms_distance",
                "absolute_delta_auroc",
                "absolute_delta_aupr",
            },
        ),
    }
    for key in ("s6", "s7"):
        if tables[key]["dataset"].duplicated().any():
            raise ValueError(f"{SOURCE_FILES[key]} contains duplicate datasets")
    if set(tables["s6"].dataset) != set(tables["s7"].dataset):
        raise ValueError("S6 and S7 dataset sets differ")
    return tables


def _verify_aggregates(
    tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, float | int], dict[str, int], dict[str, float]]:
    fig2 = tables["fig2"]
    ka = tables["ka"]
    notation = tables["notation"]
    s6 = tables["s6"]
    numeric: dict[str, float | int] = {
        "fig2_sider_connectivity_raw": _one_value(
            fig2,
            "dataset == 'SIDER' and metric == 'connectivity_match_current_rdkit'",
            "raw_rate",
            "Figure 2 SIDER raw connectivity",
        ),
        "fig2_sider_connectivity_harmonized": _one_value(
            fig2,
            "dataset == 'SIDER' and metric == 'connectivity_match_current_rdkit'",
            "harmonized_rate",
            "Figure 2 SIDER harmonized connectivity",
        ),
        "fig4_clintox_ka_positive": int(
            _one_value(ka, "dataset == 'ClinTox' and split == 'all'", "ka_pos", "ClinTox K/A+")
        ),
        "fig4_clintox_ka_negative": int(
            _one_value(ka, "dataset == 'ClinTox' and split == 'all'", "ka_neg", "ClinTox K/A-")
        ),
        "fig4_bbbp_ka_positive": int(
            _one_value(ka, "dataset == 'BBBP' and split == 'all'", "ka_pos", "BBBP K/A+")
        ),
        "fig4_bbbp_ka_negative": int(
            _one_value(ka, "dataset == 'BBBP' and split == 'all'", "ka_neg", "BBBP K/A-")
        ),
        "fig4_clintox_auroc": _one_value(
            notation,
            "dataset == 'ClinTox' and model == 'K/A indicator' and metric == 'AUROC'",
            "estimate",
            "ClinTox notation AUROC",
        ),
        "fig4_clintox_aupr": _one_value(
            notation,
            "dataset == 'ClinTox' and model == 'K/A indicator' and metric == 'AUPR'",
            "estimate",
            "ClinTox notation AUPR",
        ),
        "fig4_bbbp_auroc": _one_value(
            notation,
            "dataset == 'BBBP' and model == 'K/A indicator' and metric == 'AUROC'",
            "estimate",
            "BBBP notation AUROC",
        ),
        "fig4_bbbp_aupr": _one_value(
            notation,
            "dataset == 'BBBP' and model == 'K/A indicator' and metric == 'AUPR'",
            "estimate",
            "BBBP notation AUPR",
        ),
        "primary_identity_valid_total": int(tables["population"].primary_identity_valid.sum()),
    }
    structural = {
        "fig2_rows": int(len(fig2)),
        "fig2_latent_summary_rows": int(len(tables["latent"])),
        "fig2_string_latent_rows": int(len(tables["spearman"])),
        "fig3_adopted_aggregate_rows": int(len(tables["fig3"])),
        "fig5_metric_rows": int(len(tables["cross"])),
        "population_rows": int(len(tables["population"])),
        "s6_ci_rows": int(
            (
                (s6.connectivity_loss_ci_low <= s6.connectivity_loss_harmonized_minus_raw)
                & (s6.connectivity_loss_harmonized_minus_raw <= s6.connectivity_loss_ci_high)
            ).sum()
        ),
        "s7_labeled_points_per_panel": int(len(tables["s7"])),
    }
    numeric_errors = {
        key: abs(float(numeric[key]) - float(expected))
        for key, expected in NUMERIC_EXPECTED.items()
    }
    if max(numeric_errors.values()) > 1e-12:
        raise ValueError(f"Adopted numerical value mismatch: {numeric_errors}")
    structural_mismatches = {
        key: {"observed": structural[key], "expected": expected}
        for key, expected in STRUCTURAL_EXPECTED.items()
        if structural[key] != expected
    }
    if structural_mismatches:
        raise ValueError(f"Aggregate structure mismatch: {structural_mismatches}")
    return numeric, structural, numeric_errors


def _make_s6(data: pd.DataFrame):
    figure, axis = plt.subplots(figsize=(5.6, 5.2), layout="constrained")
    color_map = LinearSegmentedColormap.from_list(
        "latent_distance", ["#e9f1f4", "#70a5ae", "#174967"]
    )
    normalization = Normalize(vmin=0.0, vmax=float(data.median_latent_rms_distance.max()))
    lower = data.connectivity_loss_harmonized_minus_raw - data.connectivity_loss_ci_low
    upper = data.connectivity_loss_ci_high - data.connectivity_loss_harmonized_minus_raw
    axis.errorbar(
        data.changed_string_pct,
        data.connectivity_loss_harmonized_minus_raw,
        yerr=np.array([lower, upper]),
        fmt="none",
        ecolor="#6f7880",
        capsize=3,
        zorder=1,
    )
    points = axis.scatter(
        data.changed_string_pct,
        data.connectivity_loss_harmonized_minus_raw,
        c=data.median_latent_rms_distance,
        cmap=color_map,
        norm=normalization,
        s=250,
        edgecolors="#162b3d",
        linewidths=1.2,
        zorder=2,
    )
    offsets = {
        "BACE": (-28, -22),
        "BBBP": (-28, -21),
        "ClinTox": (-68, 18),
        "SIDER": (-48, 20),
        "Tox21": (12, 10),
        "ToxCast": (-65, 20),
    }
    for row in data.itertuples():
        axis.annotate(
            row.dataset,
            (row.changed_string_pct, row.connectivity_loss_harmonized_minus_raw),
            xytext=offsets[row.dataset],
            textcoords="offset points",
        )
    axis.set(
        xlabel="Records with changed SMILES (%)",
        ylabel="Connectivity-identity loss",
    )
    axis.grid(color="#d7e0e5", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    colorbar = figure.colorbar(points, ax=axis, pad=0.04)
    colorbar.set_label("Median latent RMS distance")
    return figure


def _make_s7(data: pd.DataFrame):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = {
        "BACE": "#41388b",
        "BBBP": "#d4b54c",
        "ClinTox": "#2a9fc0",
        "SIDER": "#157763",
        "Tox21": "#7c9148",
        "ToxCast": "#815b88",
    }
    for axis, column, title, label in zip(
        axes,
        ["absolute_delta_auroc", "absolute_delta_aupr"],
        ["AUROC", "AUPR"],
        ["Absolute AUROC difference", "Absolute AUPR difference"],
        strict=True,
    ):
        for row in data.itertuples():
            axis.scatter(
                row.median_latent_rms_distance,
                getattr(row, column),
                color=colors[row.dataset],
                s=90,
                label=row.dataset,
                zorder=2,
            )
        axis.set(title=title, ylabel=label)
        axis.title.set_fontweight("bold")
        axis.grid(color="#d7e0e5", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0.1, y=0.12)
    figure.subplots_adjust(bottom=0.3, left=0.08, right=0.98, top=0.9, wspace=0.28)
    figure.supxlabel("Median paired latent RMS distance", y=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=6,
        frameon=False,
    )
    return figure


def verify_and_redraw(source: Path, output: Path) -> dict[str, object]:
    """Check 19 adopted aggregate facts and generate final S6/S7 figure files."""
    source = Path(source)
    output = Path(output)
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"Source directory must be a regular directory: {source}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    tables = _load_sources(source)
    numeric, structural, numeric_errors = _verify_aggregates(tables)

    output.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    for name, figure in (
        ("supplementary_figure_s6", _make_s6(tables["s6"])),
        ("supplementary_figure_s7", _make_s7(tables["s7"])),
    ):
        for extension in ("png", "svg"):
            figure.savefig(output / f"{name}.{extension}", dpi=220, bbox_inches="tight")
        plt.close(figure)
    result: dict[str, object] = {
        "status": "PASS",
        "check_count": len(numeric) + len(structural),
        "numeric_check_count": len(numeric),
        "structural_check_count": len(structural),
        "checks": {"numeric": numeric, "structural": structural},
        "expected": {"numeric": NUMERIC_EXPECTED, "structural": STRUCTURAL_EXPECTED},
        "numeric_errors": numeric_errors,
        "max_error": max(numeric_errors.values()),
        "generated_figures": ["Supplementary Figure S6", "Supplementary Figure S7"],
    }
    (output / "validation_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
