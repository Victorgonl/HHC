"""Reusable report loading, scoring, and plotting for HHC experiments.

Plot functions accept ReportData and an output path, use PlotStyle, return the
saved Path, and close their figure. Display is opt-in with show=True. Styling is
scoped with rc_context so importing or plotting does not change global settings.

Run all reports: python -m src.report --output-dir outputs --report-dir figs
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from html import escape
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import LogFormatterMathtext

try:
    from src.hhc import evaluate
except ImportError:
    from hhc import evaluate

METHODS = ("HHC", "HHC-SE", "HHC-GM")
METRICS = (
    ("B-cubed precision", "b3_precision"),
    ("B-cubed recall", "b3_recall"),
    ("B-cubed F1", "b3_f1"),
    ("Pairwise precision", "pairwise_precision"),
    ("Pairwise recall", "pairwise_recall"),
    ("Pairwise F1", "pairwise_f1"),
    ("ACP", "average_cluster_purity"),
    ("AAP", "average_author_purity"),
    ("K", "k_metric"),
)


@dataclass(frozen=True)
class PlotStyle:
    """Shared palette, typography, dimensions, and export settings for reports."""

    background: str = "none"
    panel: str = "none"
    text: str = "#757575"
    muted_text: str = "#757575"
    grid: str = "#999999"
    border: str = "#888888"
    header: str = "none"
    summary: str = "none"
    positive_background: str = "none"
    positive_text: str = "#0072B2"
    negative_background: str = "none"
    negative_text: str = "#B35400"
    colors: tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73")
    font_family: str = "DejaVu Serif"
    font_size: float = 10
    title_size: float = 12
    table_font_size: float = 8
    figure_size: tuple[float, float] = (7.2, 3.4)
    table_size: tuple[float, float] = (10.5, 5.2)
    display_dpi: int = 140
    export_dpi: int = 450
    line_width: float = 1.3
    marker_size: float = 6
    grid_width: float = 0.5

    export_pdf: bool = False
    transparent_background: bool = True

    def rc_params(self) -> dict:
        return {
            "figure.dpi": self.display_dpi,
            "savefig.dpi": self.export_dpi,
            "figure.facecolor": self.background,
            "axes.facecolor": self.panel,
            "savefig.facecolor": self.background,
            "savefig.transparent": self.transparent_background,
            "axes.edgecolor": self.border,
            "axes.labelcolor": self.text,
            "xtick.color": self.muted_text,
            "ytick.color": self.muted_text,
            "text.color": self.text,
            "grid.color": self.grid,
            "font.size": self.font_size,
            "font.family": self.font_family,
            "lines.antialiased": True,
            "axes.linewidth": 0.6,
            "axes.titleweight": "normal",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }


DEFAULT_STYLE = PlotStyle()


@dataclass
class ReportData:
    """Scores on aligned publications, alongside original run summaries."""

    common: pd.DataFrame
    scores: dict[str, dict]
    run_summaries: dict[str, dict]
    methods: tuple[str, ...] = METHODS
    experiment_metadata: dict[str, dict] = field(default_factory=dict)


def load_report_data(output_dir: str | Path) -> ReportData:
    """Align duplicate publication occurrences before evaluating each method.

    HHC-GM publications define the comparison cohort. Every one must have a
    matching baseline and semantic prediction; those runs may contain extras.
    """
    output_dir = Path(output_dir)
    key_columns = ["paper_id", "ambiguous_name", "label"]
    frames, run_summaries, prediction_columns, experiment_metadata = {}, {}, {}, {}
    for method in METHODS:
        stem = method.lower().replace("-", "_")
        run_summaries[method] = json.loads(
            (output_dir / f"{stem}_metrics.json").read_text(encoding="utf-8")
        )
        metadata_file = output_dir / f"{stem}_experiment.json"
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            if metadata.get("method") != method:
                raise ValueError(
                    f"{method}: experiment metadata names a different method"
                )
            for key, value in metadata.get("parameters", {}).items():
                if method == "HHC-GM" and key == "retrieval_diagnostics":
                    # Older metrics overwrite the boolean argument with results.
                    summary_key = "retrieval_diagnostics_enabled"
                    if summary_key in run_summaries[method]:
                        if value != run_summaries[method][summary_key]:
                            raise ValueError(
                                f"{method}: experiment parameter {key!r} conflicts with its metrics"
                            )
                        continue
                    if isinstance(run_summaries[method].get(key), dict):
                        continue
                if (
                    key != "device"
                    and key in run_summaries[method]
                    and value != run_summaries[method][key]
                ):
                    raise ValueError(
                        f"{method}: experiment parameter {key!r} conflicts with its metrics"
                    )
            experiment_metadata[method] = metadata
        frame = pd.read_csv(output_dir / f"{stem}_predictions.csv", dtype=str)
        required = set(key_columns + ["predicted_cluster"])
        if missing := required - set(frame.columns):
            raise ValueError(f"{method}: missing prediction columns {sorted(missing)}")
        if frame[list(required)].isna().any().any():
            raise ValueError(
                f"{method}: report predictions require complete identifiers and labels"
            )
        frame["occurrence"] = frame.groupby(key_columns, dropna=False).cumcount()
        prediction_columns[method] = f"predicted_cluster_{stem}"
        frames[method] = frame.rename(
            columns={"predicted_cluster": prediction_columns[method]}
        )

    join_columns = key_columns + ["occurrence"]
    common = frames["HHC-GM"][join_columns + [prediction_columns["HHC-GM"]]]
    for method in ("HHC", "HHC-SE"):
        common = common.merge(
            frames[method][join_columns + [prediction_columns[method]]],
            on=join_columns,
            how="inner",
            validate="one_to_one",
        )
    if len(common) != len(frames["HHC-GM"]):
        raise ValueError(
            "Some HHC-GM records do not have predictions from every method"
        )
    if common.empty:
        raise ValueError("No publications are available for comparison")
    rows = common[["ambiguous_name", "label"]].to_dict(orient="records")
    scores = {
        m: evaluate(rows, common[prediction_columns[m]].tolist()) for m in METHODS
    }
    return ReportData(
        common, scores, run_summaries, experiment_metadata=experiment_metadata
    )


def _source_records(
    data: ReportData, cohort: pd.DataFrame, input_path: str | Path | None = None
) -> pd.DataFrame:
    """Attach source publication fields to aligned prediction occurrences."""
    if input_path is None:
        input_path = data.run_summaries["HHC-GM"].get("input")
    if input_path is None:
        raise ValueError("Provide input_path to the source publication CSV")
    source = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    keys = ["paper_id", "ambiguous_name", "label"]
    source["occurrence"] = source.groupby(keys, dropna=False).cumcount()
    keys.append("occurrence")
    cohort = cohort[keys]
    examples = cohort.merge(
        source, on=keys, how="left", validate="one_to_one", indicator=True
    )
    if (examples["_merge"] != "both").any():
        raise ValueError(
            "Some shared records are missing from the source publication CSV"
        )
    return examples.drop(columns="_merge")


def build_group_examples_html(
    data: ReportData,
    *,
    seed: int = 42,
    group_count: int = 1,
    records_per_group: int = 5,
    min_real_authors: int = 1,
    input_path: str | Path | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> str:
    """Show source publication fields for reproducibly sampled shared groups."""
    if group_count < 1 or records_per_group < 1:
        raise ValueError("Group count and records per group must be positive")
    if min_real_authors < 1:
        raise ValueError("Minimum real authors must be positive")
    author_counts = data.common.groupby("ambiguous_name")["label"].nunique()
    names = sorted(author_counts[author_counts >= min_real_authors].index)
    if not names:
        raise ValueError(f"No groups have at least {min_real_authors} real authors")
    selected = random.Random(seed).sample(names, k=min(group_count, len(names)))
    examples = _source_records(
        data, data.common.loc[data.common["ambiguous_name"].isin(selected)], input_path
    )
    columns = {
        "paper_id": "Paper ID",
        "author_name": "Author name",
        "coauthors": "Coauthors",
        "title": "Title",
        "venue": "Venue",
        "label": "True author ID",
    }
    columns = {key: label for key, label in columns.items() if key in examples.columns}
    sections = []
    for name in selected:
        records = examples.loc[examples["ambiguous_name"] == name]
        # Give each real author a slot before showing additional publications.
        first_per_author = ~records["label"].duplicated()
        shown = pd.concat(
            [records.loc[first_per_author], records.loc[~first_per_author]]
        ).head(records_per_group)
        headers = "".join(
            f'<th style="padding:10px;text-align:left">{escape(label)}</th>'
            for label in columns.values()
        )
        rows = []
        for values in shown[list(columns)].itertuples(index=False, name=None):
            cells = "".join(
                f'<td style="padding:10px;border-top:1px solid {escape(style.border, quote=True)};'
                f'vertical-align:top;min-width:140px;white-space:normal">{escape(str(value))}</td>'
                for value in values
            )
            rows.append(f"<tr>{cells}</tr>")
        sections.append(
            f'<section style="margin:16px 0"><h3>{escape(str(name))}</h3>'
            f"<p>{len(records):,} records · {records['label'].nunique():,} true authors"
            f" · Showing {len(shown):,} of {len(records):,} records"
            f" ({shown['label'].nunique():,} authors shown) · Seed {seed}</p>"
            '<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%">'
            f'<thead style="background:{escape(style.header, quote=True)}"><tr>{headers}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div></section>"
        )
    return (
        f'<div style="background:{escape(style.panel, quote=True)};'
        f'color:{escape(style.text, quote=True)};padding:16px;border-radius:8px">'
        f"{''.join(sections)}</div>"
    )


def display_group_examples(
    data: ReportData,
    *,
    seed: int = 42,
    group_count: int = 1,
    records_per_group: int = 5,
    min_real_authors: int = 1,
    input_path: str | Path | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> None:
    """Display group examples as notebook tables with per-group summaries."""
    from IPython.display import HTML, display

    display(
        HTML(
            build_group_examples_html(
                data,
                seed=seed,
                group_count=group_count,
                records_per_group=records_per_group,
                min_real_authors=min_real_authors,
                input_path=input_path,
                style=style,
            )
        )
    )


PAIR_OUTCOMES = {
    "TP": "True positive — same author, correctly merged",
    "TN": "True negative — different authors, correctly separated",
    "FP": "False positive — different authors, incorrectly merged",
    "FN": "False negative — same author, incorrectly separated",
}


def select_model_examples(
    data: ReportData, *, seed: int = 42, examples_per_outcome: int = 1
) -> dict:
    """Reservoir-sample record pairs per model and outcome within name groups.

    Positive means assigned to the same cluster. Sample all aligned record
    occurrences, without retaining the full (potentially quadratic) pair list.
    Record positions refer to data.common.iloc, independent of its index labels.
    """
    if examples_per_outcome < 1:
        raise ValueError("Examples per outcome must be positive")
    rng = random.Random(seed)
    results = {
        method: {outcome: {"count": 0, "pairs": []} for outcome in PAIR_OUTCOMES}
        for method in data.methods
    }
    labels = data.common["label"].to_numpy()
    predictions = {
        method: data.common[
            f"predicted_cluster_{method.lower().replace('-', '_')}"
        ].to_numpy()
        for method in data.methods
    }
    for positions in data.common.groupby("ambiguous_name", sort=True).indices.values():
        for left, right in combinations(positions, 2):
            same_author = labels[left] == labels[right]
            for method in data.methods:
                same_cluster = predictions[method][left] == predictions[method][right]
                outcome = (
                    ("TP" if same_cluster else "FN")
                    if same_author
                    else ("FP" if same_cluster else "TN")
                )
                bucket = results[method][outcome]
                bucket["count"] += 1
                if len(bucket["pairs"]) < examples_per_outcome:
                    bucket["pairs"].append((int(left), int(right)))
                else:
                    slot = rng.randrange(bucket["count"])
                    if slot < examples_per_outcome:
                        bucket["pairs"][slot] = (int(left), int(right))
    return results


def display_model_examples(
    data: ReportData,
    *,
    seed: int = 42,
    examples_per_outcome: int = 1,
    input_path: str | Path | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> None:
    """Display source records illustrating each model's pairwise successes/errors."""
    from IPython.display import HTML, display

    results = select_model_examples(
        data, seed=seed, examples_per_outcome=examples_per_outcome
    )
    positions = sorted(
        {
            position
            for outcomes in results.values()
            for bucket in outcomes.values()
            for pair in bucket["pairs"]
            for position in pair
        }
    )
    if positions:
        source = _source_records(data, data.common.iloc[positions], input_path)
        records_by_position = dict(
            zip(positions, source.to_dict("records"), strict=True)
        )
    sections = [
        f"<p>Within-name record pairs · Seed {seed}. Positive = same predicted cluster.</p>"
    ]
    for method, outcomes in results.items():
        sections.append(f"<h2>{escape(method)}</h2>")
        prediction_key = f"predicted_cluster_{method.lower().replace('-', '_')}"
        for outcome, bucket in outcomes.items():
            sections.append(
                f"<h3>{escape(PAIR_OUTCOMES[outcome])}</h3>"
                f"<p>{bucket['count']:,} qualifying pairs</p>"
            )
            if not bucket["pairs"]:
                sections.append("<p>No examples available in the shared cohort.</p>")
            for number, pair in enumerate(bucket["pairs"], start=1):
                rows = []
                for letter, position in zip(("A", "B"), pair, strict=True):
                    record = records_by_position[position]
                    fields = {"Record": letter}
                    for key, label in (
                        ("paper_id", "Paper ID"),
                        ("author_name", "Author name"),
                        ("coauthors", "Coauthors"),
                        ("title", "Title"),
                        ("venue", "Venue"),
                        ("label", "True author ID"),
                    ):
                        if key in record:
                            fields[label] = record[key]
                    fields["Predicted cluster"] = data.common.iloc[position][
                        prediction_key
                    ]
                    rows.append(fields)
                name = data.common.iloc[pair[0]]["ambiguous_name"]
                sections.append(
                    f"<h4>Example {number} · {escape(str(name))}</h4>"
                    '<div style="overflow-x:auto">'
                    + pd.DataFrame(rows).to_html(index=False, escape=True, border=0)
                    + "</div>"
                )
    display(
        HTML(
            f'<div style="background:{escape(style.panel, quote=True)};'
            f'color:{escape(style.text, quote=True)};padding:16px">'
            + "".join(sections)
            + "</div>"
        )
    )


def build_error_changes(data: ReportData) -> pd.DataFrame:
    """Count within-name FP/FN errors corrected or introduced versus baseline."""
    baseline, *extensions = data.methods
    counts = {
        method: dict.fromkeys(
            (
                "False merges corrected",
                "False merges introduced",
                "False splits corrected",
                "False splits introduced",
            ),
            0,
        )
        for method in extensions
    }
    labels = data.common["label"].to_numpy()
    predictions = {
        method: data.common[
            f"predicted_cluster_{method.lower().replace('-', '_')}"
        ].to_numpy()
        for method in data.methods
    }
    for positions in data.common.groupby("ambiguous_name", sort=True).indices.values():
        for left, right in combinations(positions, 2):
            truth = labels[left] == labels[right]
            baseline_same = predictions[baseline][left] == predictions[baseline][right]
            for method in extensions:
                same = predictions[method][left] == predictions[method][right]
                if same == baseline_same:
                    continue
                error = "False splits" if truth else "False merges"
                change = "corrected" if same == truth else "introduced"
                counts[method][f"{error} {change}"] += 1
    return pd.DataFrame.from_dict(counts, orient="index")


def build_pair_confusion_counts(data: ReportData) -> pd.DataFrame:
    """Count publication-pair outcomes within blocks without enumerating pairs."""
    counts = {
        method: dict.fromkeys(("TP", "TN", "FP", "FN"), 0) for method in data.methods
    }

    def pairs(sizes):
        return int(sum(int(size) * (int(size) - 1) // 2 for size in sizes))

    for _, records in data.common.groupby("ambiguous_name", sort=True):
        total = len(records) * (len(records) - 1) // 2
        true_pairs = pairs(records.groupby("label").size())
        for method in data.methods:
            key = f"predicted_cluster_{method.lower().replace('-', '_')}"
            predicted_pairs = pairs(records.groupby(key).size())
            tp = pairs(records.groupby(["label", key]).size())
            counts[method]["TP"] += tp
            counts[method]["FP"] += predicted_pairs - tp
            counts[method]["FN"] += true_pairs - tp
            counts[method]["TN"] += total - predicted_pairs - true_pairs + tp
    return pd.DataFrame.from_dict(counts, orient="index")


def plot_error_changes(
    data: ReportData,
    output_path: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> Path:
    """Plot error corrections and regressions on the shared evaluation cohort."""
    changes = build_error_changes(data)
    baseline = data.methods[0]
    with plt.rc_context(style.rc_params()):
        fig, axes = plt.subplots(
            1, 2, figsize=(style.table_size[0], style.figure_size[1]), sharey=False
        )
        try:
            positions = np.arange(len(changes))
            for axis, error, title in zip(
                axes,
                ("False merges", "False splits"),
                ("False positives (false merges)", "False negatives (false splits)"),
                strict=True,
            ):
                for offset, change, color in (
                    (-0.18, "corrected", style.positive_text),
                    (0.18, "introduced", style.negative_text),
                ):
                    values = changes[f"{error} {change}"].to_numpy()
                    bars = axis.bar(
                        positions + offset,
                        values,
                        width=0.36,
                        label=change.capitalize(),
                        color=color,
                        edgecolor=style.text,
                        linewidth=0.5,
                        hatch="///" if change == "introduced" else None,
                    )
                    axis.bar_label(
                        bars, labels=[f"{value:,}" for value in values], padding=4
                    )
                axis.set_xticks(positions, changes.index)
                axis.set_title(title, fontsize=style.font_size)
                axis.set_ylim(
                    0,
                    max(
                        1,
                        changes[[f"{error} corrected", f"{error} introduced"]]
                        .to_numpy()
                        .max(),
                    )
                    * 1.25,
                )
                axis.set_ylabel("Number of record pairs")
                axis.grid(axis="y", alpha=0.35)
                axis.set_axisbelow(True)
                axis.spines[["top", "right"]].set_visible(False)
            axes[1].legend(frameon=False, labelcolor=style.text)
            fig.suptitle(
                f"Errors corrected or introduced vs {baseline}",
                fontsize=style.title_size,
                fontweight="normal",
            )
            fig.text(
                0.5,
                0.01,
                "Within-name pairs · Panel scales differ · Lower introduced counts are better",
                ha="center",
                color=style.muted_text,
                fontsize=style.table_font_size,
            )
            fig.tight_layout(rect=(0, 0.04, 1, 0.95))
            return _save_figure(fig, output_path, style=style, show=show)
        finally:
            plt.close(fig)


def format_duration(seconds: float) -> str:
    """Format elapsed seconds with explicit h, min, and s units."""
    if seconds < 1:
        return f"{seconds:.3f}s"
    hours, remainder = divmod(round(seconds), 3_600)
    minutes, whole_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}min {whole_seconds}s"
    if minutes:
        return f"{minutes}min {whole_seconds}s"
    return f"{whole_seconds}s"


def _save_figure(
    fig: Figure,
    output_path: str | Path,
    *,
    style: PlotStyle,
    show: bool,
    pad_inches: float = 0.1,
) -> Path:
    """Save and optionally display; the plotting function owns figure cleanup."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=style.export_dpi,
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="none" if style.transparent_background else fig.get_facecolor(),
        transparent=style.transparent_background,
    )
    if style.export_pdf and path.suffix.lower() == ".png":
        fig.savefig(
            path.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=pad_inches,
            facecolor="none" if style.transparent_background else fig.get_facecolor(),
            transparent=style.transparent_background,
        )
    if show:
        plt.show()
    return path


def plot_metrics(
    data: ReportData,
    output_path: str | Path,
    *,
    metric_keys: tuple[str, ...],
    labels: tuple[str, ...],
    title: str,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> Path:
    """Compare methods using grouped metric bars on a shared zero-based axis."""
    if not metric_keys or len(metric_keys) != len(labels):
        raise ValueError("Provide one label for each metric")
    if len(style.colors) < len(metric_keys):
        raise ValueError("The style needs at least one color per metric")
    values_by_metric = [
        np.array([data.scores[m][key] for m in data.methods]) for key in metric_keys
    ]
    x = np.arange(len(data.methods))
    with plt.rc_context(style.rc_params()):
        fig, axis = plt.subplots(figsize=style.figure_size)
        try:
            bar_width = 0.72 / len(labels)
            for index, (label, values, color) in enumerate(
                zip(labels, values_by_metric, style.colors)
            ):
                positions = x + (index - (len(labels) - 1) / 2) * bar_width
                bars = axis.bar(
                    positions,
                    values,
                    width=bar_width * 0.90,
                    linewidth=0.5,
                    color=color,
                    edgecolor=style.border,
                    hatch=("", "///", "...")[index % 3],
                    label=label,
                )
                axis.bar_label(
                    bars,
                    labels=[f"{value:.3f}" for value in values],
                    padding=4,
                    color=style.text,
                    fontsize=style.table_font_size,
                )
            axis.set_xticks(x, data.methods)
            axis.set_ylim(0, 1.05)
            axis.set_xlim(-0.5, len(data.methods) - 0.5)
            axis.set_axisbelow(True)
            ticks = axis.get_yticks()
            axis.set_yticks(ticks[(ticks >= 0) & (ticks <= 1)])
            axis.set_ylabel("Score")
            axis.set_title(
                title, fontsize=style.title_size, fontweight="normal", pad=12
            )
            axis.grid(axis="y", linewidth=style.grid_width, alpha=0.65, linestyle=":")
            axis.grid(axis="x", visible=False)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(
                frameon=False,
                facecolor=style.panel,
                edgecolor=style.grid,
                labelcolor=style.text,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.13),
                ncol=len(labels),
            )
            fig.tight_layout()
            return _save_figure(fig, output_path, style=style, show=show)
        finally:
            plt.close(fig)


def plot_pairwise(
    data: ReportData,
    output_path: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> Path:
    """Save pairwise precision, recall, and F1."""
    return plot_metrics(
        data,
        output_path,
        metric_keys=("pairwise_precision", "pairwise_recall", "pairwise_f1"),
        labels=("Precision", "Recall", "F1"),
        title="Pairwise metrics",
        style=style,
        show=show,
    )


def plot_bcubed(
    data: ReportData,
    output_path: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> Path:
    """Save B-cubed precision, recall, and F1."""
    return plot_metrics(
        data,
        output_path,
        metric_keys=("b3_precision", "b3_recall", "b3_f1"),
        labels=("Precision", "Recall", "F1"),
        title="B-cubed metrics",
        style=style,
        show=show,
    )


def plot_purity(
    data: ReportData,
    output_path: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> Path:
    """Save average cluster purity, average author purity, and K."""
    return plot_metrics(
        data,
        output_path,
        metric_keys=("average_cluster_purity", "average_author_purity", "k_metric"),
        labels=("ACP", "AAP", "K"),
        title="Purity and K metrics",
        style=style,
        show=show,
    )


def build_data_statistics_rows(data: ReportData) -> tuple[list[str], list[list[str]]]:
    """Describe the shared cohort using distinct papers and ground-truth authors.

    Duplicate publication occurrences do not inflate paper counts. An author is
    identified by its ground-truth label, as in the evaluation metrics.
    """
    common = data.common
    papers = common.drop_duplicates(["ambiguous_name", "label", "paper_id"])
    authors = common[["ambiguous_name", "label"]].drop_duplicates()
    distributions = (
        ("Papers / group", papers.groupby("ambiguous_name")["paper_id"].nunique()),
        ("Papers / author", papers.groupby("label")["paper_id"].nunique()),
        ("Authors / group", authors.groupby("ambiguous_name").size()),
    )
    columns = [
        "Compared records",
        "Distinct papers",
        "Ambiguous groups",
        "Real authors",
    ]
    values = [
        f"{len(common):,}",
        f"{common['paper_id'].nunique():,}",
        f"{common['ambiguous_name'].nunique():,}",
        f"{common['label'].nunique():,}",
    ]
    for label, counts in distributions:
        columns.append(f"{label.title()}\n(Min/Max/Avg)")
        values.append(f"{counts.min():,} / {counts.max():,} / {counts.mean():.2f}")
    return columns, [values]


def build_summary_rows(
    data: ReportData,
) -> tuple[list[str], list[list[str]], list[dict[str, float]]]:
    """Build the comparison table, with each extension's delta beside its scores."""
    baseline, *extensions = data.methods
    columns = ["Statistic", baseline]
    for method in extensions:
        columns.extend([method, f"{method} Δ vs {baseline}"])

    def row(label, values, deltas=None):
        cells = [label, values[baseline]]
        for method in extensions:
            cells.extend([values[method], (deltas or {}).get(method, "—")])
        return cells

    methods, scores, summaries = data.methods, data.scores, data.run_summaries
    table_rows = [
        row("Run publications", {m: f"{summaries[m]['records']:,}" for m in methods}),
        row(
            "Total process time",
            {m: format_duration(summaries[m]["runtime_seconds"]) for m in methods},
        ),
        row(
            "Predicted clusters",
            {m: f"{scores[m]['predicted_clusters']:,}" for m in methods},
            {
                m: f"{scores[m]['predicted_clusters'] - scores[baseline]['predicted_clusters']:+,}"
                for m in extensions
            },
        ),
    ]
    changes = []
    for label, key in METRICS:
        deltas = {m: (scores[m][key] - scores[baseline][key]) * 100 for m in extensions}
        changes.append(deltas)
        table_rows.append(
            row(
                label,
                {m: f"{scores[m][key]:.3f}" for m in methods},
                {m: f"{delta:+.3f} pp" for m, delta in deltas.items()},
            )
        )
    return columns, table_rows, changes


def plot_data_statistics(
    data: ReportData,
    output_path: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> Path:
    """Save a standalone table describing the shared evaluation cohort."""
    statistics_columns, statistics_rows = build_data_statistics_rows(data)
    with plt.rc_context(style.rc_params()):
        fig, statistics_axis = plt.subplots(figsize=(style.table_size[0], 1.6))
        try:
            statistics_axis.axis("off")
            statistics_axis.set_title(
                "Shared evaluation data (all methods)", fontsize=style.font_size, pad=8
            )
            statistics_table = statistics_axis.table(
                cellText=statistics_rows,
                colLabels=statistics_columns,
                colWidths=[0.12, 0.12, 0.12, 0.10, 0.18, 0.18, 0.18],
                cellLoc="center",
                bbox=[0, 0, 1, 1],
            )
            statistics_table.auto_set_font_size(False)
            statistics_table.set_fontsize(style.table_font_size)
            for (row, column), cell in statistics_table.get_celld().items():
                cell.set_facecolor(style.header if row == 0 else style.summary)
                cell.set_edgecolor(style.border)
                cell.set_linewidth(style.grid_width)
                cell.visible_edges = "horizontal"
                cell.set_text_props(
                    color=style.text,
                    fontweight="bold" if row == 0 or column == 0 else "normal",
                )
            fig.subplots_adjust(left=0, right=1, bottom=0, top=0.75)
            return _save_figure(fig, output_path, style=style, show=show)
        finally:
            plt.close(fig)


def plot_summary_table(
    data: ReportData,
    output_path: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> Path:
    """Save experiment scores and extension deltas against HHC."""
    columns, rows, changes = build_summary_rows(data)
    summary_count = len(rows) - len(changes)
    with plt.rc_context(style.rc_params()):
        fig, axis = plt.subplots(figsize=style.table_size)
        try:
            axis.axis("off")
            table = axis.table(
                cellText=rows,
                colLabels=columns,
                colLoc="center",
                cellLoc="center",
                bbox=[0, 0, 1, 1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(style.table_font_size)
            for (row, column), cell in table.get_celld().items():
                cell.set_facecolor(style.panel)
                cell.set_edgecolor(style.border)
                cell.set_linewidth(style.grid_width)
                cell.visible_edges = "horizontal"
                cell.set_text_props(color=style.text)
                if row == 0:
                    cell.set_facecolor(style.header)
                    cell.set_text_props(fontweight="bold")
                elif row <= summary_count:
                    cell.set_facecolor(style.summary)
                    if column == 0:
                        cell.set_text_props(fontweight="bold")
            for row, deltas in enumerate(changes, start=summary_count + 1):
                for column, method in enumerate(data.methods[1:], start=1):
                    cell = table[(row, column * 2 + 1)]
                    positive = deltas[method] >= 0
                    cell.set_facecolor(
                        style.positive_background
                        if positive
                        else style.negative_background
                    )
                    cell.set_text_props(
                        color=style.positive_text if positive else style.negative_text,
                        fontweight="normal",
                    )
            primary_row = next(
                index
                for index, row in enumerate(rows, start=1)
                if row[0] == "B-cubed F1"
            )
            for column in range(len(columns)):
                table[(primary_row, column)].set_text_props(fontweight="bold")
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            return _save_figure(fig, output_path, style=style, show=show, pad_inches=0)
        finally:
            plt.close(fig)


def plot_execution_time(
    data: ReportData,
    output_path: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> Path:
    """Compare positive saved run durations on a logarithmic axis."""
    seconds = np.array(
        [data.run_summaries[method]["runtime_seconds"] for method in data.methods],
        dtype=float,
    )
    if not np.isfinite(seconds).all() or (seconds <= 0).any():
        raise ValueError("Execution times must be finite and positive for a log scale")
    hhc_seconds = float(data.run_summaries["HHC"]["runtime_seconds"])
    if not np.isfinite(hhc_seconds) or hhc_seconds <= 0:
        raise ValueError("HHC execution time must be finite and positive")
    lower_bound = 10 ** (np.floor(np.log10(seconds.min())) - 1)
    with plt.rc_context(style.rc_params()):
        fig, axis = plt.subplots(figsize=style.figure_size)
        try:
            positions = np.arange(len(data.methods))
            for index, (method, duration) in enumerate(
                zip(data.methods, seconds, strict=True)
            ):
                axis.barh(
                    index,
                    duration - lower_bound,
                    left=lower_bound,
                    height=0.52,
                    color=style.colors[index % len(style.colors)],
                    edgecolor=style.border,
                    linewidth=0.5,
                    hatch=("", "///", "...")[index % 3],
                )
                label = format_duration(duration)
                if method != "HHC":
                    label += f"\n({duration / hhc_seconds:.1f}× slower)"
                axis.annotate(
                    label,
                    (duration, index),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=style.font_size,
                )
            axis.set_yticks(positions, data.methods)
            axis.invert_yaxis()
            axis.set_xscale("log")
            axis.set_xlim(lower_bound, seconds.max() * 2)
            axis.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
            axis.set_xlabel("Reported execution time (s, logarithmic scale)")
            axis.set_title("Execution time", fontsize=style.title_size, pad=12)
            axis.grid(axis="x", linewidth=style.grid_width, linestyle=":", alpha=0.6)
            axis.set_axisbelow(True)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.tick_params(axis="y", length=0)
            fig.tight_layout()
            return _save_figure(fig, output_path, style=style, show=show)
        finally:
            plt.close(fig)


def plot_methodology(
    data: ReportData,
    output_path: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> Path:
    """Compare the implemented clustering workflows, independent of run scores."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    with plt.rc_context(style.rc_params()):
        fig, ax = plt.subplots(figsize=(14, 9))
        try:
            ax.set(xlim=(0, 14), ylim=(-0.6, 10.5))
            ax.axis("off")
            hhc, se, gm = style.colors
            ax.text(
                7,
                10.1,
                "Method: shared HHC steps and model extensions",
                ha="center",
                fontsize=style.title_size,
                weight="bold",
            )
            ax.text(
                7,
                9.65,
                "Each workflow operates independently within an ambiguous-name block",
                ha="center",
                fontsize=10,
                color=style.muted_text,
            )

            def box(x, y, title, body, color, width=3.8):
                ax.add_patch(
                    FancyBboxPatch(
                        (x - width / 2, y - 0.6),
                        width,
                        1.2,
                        boxstyle="round,pad=0.04,rounding_size=0.09",
                        facecolor=style.panel,
                        edgecolor=color,
                        linewidth=style.line_width,
                    )
                )
                ax.text(
                    x,
                    y + 0.31,
                    title,
                    ha="center",
                    va="center",
                    fontsize=10,
                    weight="bold",
                    color=color,
                )
                ax.text(
                    x,
                    y - 0.16,
                    body,
                    ha="center",
                    va="center",
                    fontsize=9,
                    linespacing=1.5,
                )

            def arrow(points, color):
                ax.plot(*zip(*points), color=color, lw=1.2, solid_capstyle="round")
                ax.annotate(
                    "",
                    xy=points[-1],
                    xytext=points[-2],
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                )

            box(
                7,
                8.55,
                "Shared HHC initialization",
                "Normalize names, titles, and venues; initialize clusters\nwith compatible author names and shared coauthors",
                hhc,
                width=8,
            )
            arrow([(4, 7.91), (4, 7.55), (2.3, 7.55), (2.3, 7.14)], hhc)
            arrow([(10, 7.91), (10, 7.55), (11.7, 7.55), (11.7, 7.14)], se)
            box(
                2.3,
                6.5,
                "HHC · lexical merging",
                "Compatible names AND\n(title OR venue similarity above threshold)",
                hhc,
            )
            arrow([(2.3, 5.86), (2.3, 5.14)], hhc)
            ax.text(
                2.3,
                5.5,
                "Aggregate evidence; repeat until stable",
                ha="center",
                fontsize=8,
                color=hhc,
            )
            box(
                2.3,
                4.5,
                "HHC · final author clusters",
                "Baseline output\nReused as HHC-GM starting clusters",
                hhc,
            )
            arrow([(4.24, 4.5), (5.06, 4.5)], gm)
            box(
                7,
                4.5,
                "HHC-GM · retrieve pairs",
                "SemCSE ranking of compatible clusters\nSimilarity cutoff + top-k filtering",
                gm,
            )
            arrow([(7, 3.86), (7, 3.14)], gm)
            box(
                7,
                2.5,
                "HHC-GM · judge and merge",
                "Select evidence; validate LLM JSON\nPositive answer + probability threshold",
                gm,
            )
            arrow([(7, 1.86), (7, 1.34)], gm)
            box(
                7,
                0.7,
                "HHC-GM · final author clusters",
                "Stop: no approved pair or budget exhausted\nOriginal HHC clusters are never split",
                gm,
            )
            arrow([(5.06, 2.5), (4.65, 2.5), (4.65, 5.45), (7, 5.45), (7, 5.14)], gm)
            ax.text(
                6.35,
                5.63,
                "After a merge: update clusters and retrieve again",
                ha="center",
                fontsize=8,
                color=gm,
            )
            box(
                11.7,
                6.5,
                "HHC-SE · semantic evidence",
                "Encode title + venue with SemCSE\nSum normalized record embeddings",
                se,
            )
            arrow([(11.7, 5.86), (11.7, 5.14)], se)
            box(
                11.7,
                4.5,
                "HHC-SE · extended merging",
                "Compatible names AND similarity above\na title, venue, OR embedding threshold",
                se,
            )
            arrow([(11.7, 3.86), (11.7, 3.14)], se)
            ax.text(
                11.7,
                3.5,
                "Update lexical evidence and vector sums",
                ha="center",
                fontsize=8,
                color=se,
            )
            box(
                11.7,
                2.5,
                "HHC-SE · final author clusters",
                "Lexical and semantic merges interleave\nStop: a complete pass has no merge",
                se,
            )
            ax.legend(
                handles=[
                    Line2D([0], [0], color=color, lw=3, label=label)
                    for color, label in (
                        (hhc, "HHC / shared baseline steps"),
                        (se, "HHC-SE extension"),
                        (gm, "HHC-GM extension"),
                    )
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, -0.055),
                ncol=3,
                fontsize=9,
                handlelength=2,
                columnspacing=2.5,
            )
            ax.text(
                7,
                -0.35,
                "Ground-truth labels are reserved for evaluation and audit logs.",
                ha="center",
                fontsize=8.5,
                color=style.muted_text,
            )
            return _save_figure(fig, output_path, style=style, show=show)
        finally:
            plt.close(fig)


def generate_reports(
    data: ReportData,
    report_dir: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> dict[str, Path]:
    """Generate all standard report images and return their paths."""
    plots = {
        "data_statistics.png": plot_data_statistics,
        "pairwise_metrics.png": plot_pairwise,
        "bcubed_metrics.png": plot_bcubed,
        "purity_k_metrics.png": plot_purity,
        "reports_table.png": plot_summary_table,
        "error_changes.png": plot_error_changes,
        "execution_time.png": plot_execution_time,
        "methodology.png": plot_methodology,
    }
    return {
        name: plot(data, Path(report_dir) / name, style=style, show=show)
        for name, plot in plots.items()
    }


PAPER_SECTIONS = (
    "ABSTRACT",
    "METHODOLOGY",
    "DATA",
    "SETTINGS",
    "RESULTS",
    "FIGURES",
    "CONCLUSION",
)
REPORT_FIGURES = {
    "methodology.png": "Method workflows for HHC, HHC-SE, and HHC-GM. HHC-SE interleaves lexical and semantic merging; HHC-GM adds retrieval and generative judgments after complete HHC",
    "data_statistics.png": "Shared data: totals and min/max/average distributions",
    "bcubed_metrics.png": "B-cubed precision, recall, and F1 on aligned records (primary metric)",
    "pairwise_metrics.png": "Pairwise precision, recall, and F1 on aligned records (secondary metric)",
    "purity_k_metrics.png": "Macro-averaged group purity (ACP, AAP) and K",
    "reports_table.png": "Complete scores, run times, and changes against HHC",
    "error_changes.png": "Within-name false merges and false splits corrected or introduced against HHC",
    "execution_time.png": "Reported execution time in seconds (logarithmic axis). Neural model setup is excluded. Durations reflect the recorded executions and are not a controlled hardware benchmark",
}


def build_machine_configuration(
    data: ReportData, output_dir: str | Path = "outputs"
) -> str:
    """Build a concise Markdown table from saved experiment environments."""
    environment_path = Path(output_dir) / "experiment_environment.json"
    legacy = (
        json.loads(environment_path.read_text(encoding="utf-8"))
        if environment_path.exists()
        else {}
    )
    environments = [
        data.experiment_metadata.get(method, {}).get("environment", legacy)
        for method in data.methods
    ]
    fields = (
        ("CPU", ("cpu", "model")),
        ("Logical CPU cores", ("cpu", "logical_processors")),
        ("RAM", ("system_memory_mib",)),
        ("GPU", ("gpu", "model")),
        ("GPU memory", ("gpu", "memory_mib")),
        ("Operating system", ("operating_system",)),
        ("Architecture", ("architecture",)),
        ("Python", ("python_version",)),
        ("PyTorch", ("pytorch_version",)),
        ("CUDA runtime (PyTorch)", ("pytorch_cuda_version",)),
        ("NVIDIA driver", ("gpu", "driver_version")),
    )

    def cell(environment, path):
        value = environment
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is None:
            return "Not recorded"
        if path[-1] in ("system_memory_mib", "memory_mib"):
            value = f"{value / 1024:.2f} GiB"
        return escape(str(value)).replace("|", "&#124;").replace("\n", " ")

    rows = [
        [label, *(cell(env, path) for env in environments)] for label, path in fields
    ]
    # Collapse identical machines into one configuration column.
    shared = all(len(set(row[1:]) - {"Not recorded"}) <= 1 for row in rows)
    header = ["Component", "Configuration"] if shared else ["Component", *data.methods]
    if shared:
        # Retain which runs recorded software details without duplicating hardware.
        combined = []
        for row in rows:
            known = [
                (method, value)
                for method, value in zip(data.methods, row[1:])
                if value != "Not recorded"
            ]
            value = known[0][1] if known else "Not recorded"
            if known and len(known) < len(data.methods):
                value += (
                    " (recorded for " + ", ".join(method for method, _ in known) + ")"
                )
            combined.append([row[0], value])
        rows = combined
    return "\n".join(
        "| " + " | ".join(row) + " |" for row in [header, ["---"] * len(header), *rows]
    )


def build_machine_configuration_html(
    data: ReportData,
    output_dir: str | Path = "outputs",
    *,
    style: PlotStyle = DEFAULT_STYLE,
) -> str:
    """Render the machine table using the report palette and typography."""
    lines = build_machine_configuration(data, output_dir).splitlines()
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for index, line in enumerate(lines)
        if index != 1
    ]
    return _configuration_table_html(rows, "Machine configuration", style)


def _configuration_table_html(rows, title: str, style: PlotStyle) -> str:
    """Render already escaped table cells with the shared report style."""
    border = escape(style.border, quote=True)
    body = []
    for index, row in enumerate(rows):
        tag = "th" if index == 0 else "td"
        background = style.header if index == 0 else style.panel
        cells = "".join(
            f'<{tag} style="padding:8px 12px;text-align:left;border-bottom:1px solid {border};'
            f'background:{escape(background, quote=True)}">{value}</{tag}>'
            for value in row
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        f'<div style="overflow-x:auto;color:{escape(style.text, quote=True)};'
        f"font-family:{escape(style.font_family, quote=True)};font-size:{style.font_size}pt;"
        f'background:{escape(style.background, quote=True)}">'
        f'<h3 style="font-size:{style.title_size}pt">{escape(title)}</h3>'
        '<table style="border-collapse:collapse;width:100%">'
        + "".join(body)
        + "</table></div>"
    )


def display_machine_configuration(
    data: ReportData,
    output_dir: str | Path = "outputs",
    *,
    style: PlotStyle = DEFAULT_STYLE,
) -> None:
    """Display the styled machine configuration in a notebook."""
    from IPython.display import HTML, display

    display(HTML(build_machine_configuration_html(data, output_dir, style=style)))


def build_model_parameters_html(
    data: ReportData, *, style: PlotStyle = DEFAULT_STYLE
) -> str:
    """Show recorded run parameters in one styled table per method."""
    excluded = {
        "output",
        "metrics_output",
        "prompts_output",
        "decisions_output",
        "decision_cache",
        "embedding_cache",
        "model_cache",
        "rebuild_cache",
        "no_progress",
    }
    tables = []
    for method in data.methods:
        metadata = data.experiment_metadata.get(method, {})
        # Historical runs may only have parameters embedded in their metrics.
        parameters = metadata.get("parameters")
        if parameters is None:
            parameters = {
                key: value
                for key, value in data.run_summaries[method].items()
                if key
                in {
                    "input",
                    "n_ambiguous_group",
                    "seed",
                    "device",
                    "model",
                    "model_revision",
                    "title_threshold",
                    "venue_threshold",
                    "semantic_threshold",
                    "batch_size",
                    "max_length",
                    "embedding_model",
                    "embedding_batch_size",
                    "embedding_max_length",
                    "llm_confidence_threshold",
                    "semantic_candidate_threshold",
                    "semantic_top_k",
                    "candidate_score",
                    "max_llm_comparisons_per_group",
                    "max_records_per_cluster",
                    "max_input_tokens",
                    "max_new_tokens",
                    "llm_retries",
                    "cpu_threads",
                    "cpu_int8",
                }
            }
        parameters = {
            key: value for key, value in parameters.items() if key not in excluded
        }
        if "device" in metadata.get("resolved", {}):
            parameters["resolved_device"] = metadata["resolved"]["device"]
        rows = [["Parameter", "Value"]]
        rows.extend(
            [escape(str(key)), escape(str(value))]
            for key, value in sorted(parameters.items())
        )
        if len(rows) == 1:
            rows.append(["Parameters", "Not recorded"])
        tables.append(_configuration_table_html(rows, f"{method} parameters", style))
    return "\n".join(tables)


def display_model_parameters(
    data: ReportData, *, style: PlotStyle = DEFAULT_STYLE
) -> None:
    """Display the saved parameter tables in a notebook."""
    from IPython.display import HTML, display

    display(HTML(build_model_parameters_html(data, style=style)))


def build_paper_sections(
    data: ReportData,
    *,
    readme_path: str | Path = "README.md",
    report_dir: str | Path = "figs",
    output_dir: str | Path = "outputs",
) -> dict[str, str]:
    """Generate factual paper sections from current aligned scores and run metadata."""
    baseline = data.methods[0]
    summaries = {
        method: {
            **data.run_summaries[method],
            **data.experiment_metadata.get(method, {}).get("parameters", {}),
            **data.experiment_metadata.get(method, {}).get("resolved", {}),
        }
        for method in data.methods
    }
    scores = data.scores
    best = max(data.methods, key=lambda method: scores[method]["b3_f1"])
    readme_parent = Path(readme_path).resolve().parent

    def link(path):
        return Path(os.path.relpath(Path(path).resolve(), readme_parent)).as_posix()

    abstract = (
        "*"
        "This work studies Author Name Disambiguation (AND) using heuristic hierarchical clustering (HHC), "
        "a semantic-evidence extension (HHC-SE), and a generative-model extension (HHC-GM). "
        f"Using a subset of the LAGOS-AND dataset, the shared evaluation cohort contains {len(data.common):,} records from "
        f"{data.common['ambiguous_name'].nunique():,} ambiguous-name groups and "
        f"{data.common['label'].nunique():,} real authors. "
        + "B-cubed F1 is "
        + ", ".join(
            f"{scores[method]['b3_f1']:.4f} for {method}" for method in data.methods
        )
        + ". "
        f"{best} has the highest B-cubed F1 in the reported experimental evaluation. "
        "The results describe one experiment per method, without a significance test."
        "*"
    )
    dataset = (
        "We evaluate on a subset of LAGOS-AND [[6]](#ref-6), using ORCID-based "
        "ground-truth author identities stored in the `label` field. "
        "Evaluation aligns `(paper_id, ambiguous_name, label, occurrence)` across methods, "
        "retaining every HHC-GM occurrence and requiring a match in both other runs. "
        "Baseline and semantic runs can contain extra records. Duplicate occurrences remain "
        "in evaluation, while the descriptive paper counts use distinct paper IDs.\n\n"
        + "See Figure 2 for shared-data statistics."
        + "\n\nAuthors are identified by ground-truth ORCID IDs. The number of authors per "
        "ambiguous-name group is computed independently for each group, and a publication "
        "may appear in more than one group."
    )

    def enumerate_methods(methods):
        return (
            methods[0]
            if len(methods) == 1
            else ", ".join(methods[:-1]) + " and " + methods[-1]
        )

    def numeric(value):
        return f"{value:,}" if isinstance(value, int) else str(value)

    # Consolidate identical settings, while retaining differences between runs.
    grouped_settings = {}
    for method in data.methods:
        summary = summaries[method]
        key = tuple(
            summary.get(field, "Not recorded")
            for field in (
                "input",
                "seed",
                "records",
                "n_ambiguous_group",
                "title_threshold",
                "venue_threshold",
            )
        )
        grouped_settings.setdefault(key, []).append(method)
    settings = "#### 4.2.1 Experimental cohort and lexical thresholds\n\n"
    descriptions = []
    for (
        source,
        seed,
        records,
        groups,
        title,
        venue,
    ), methods in grouped_settings.items():
        descriptions.append(
            f"{enumerate_methods(methods)} {'was' if len(methods) == 1 else 'were'} evaluated using `{source}`, with "
            f"{numeric(records)} publication records and a requested sample of {numeric(groups)} "
            f"complete ambiguous-name groups. Group sampling used random seed {seed}. "
            f"The title and venue cosine-similarity thresholds were set to {title} and {venue}, "
            "respectively. Lexical merging required compatible author names and similarity exceeding "
            "at least one of these lexical thresholds."
        )
    settings += "\n\n".join(descriptions)
    settings += (
        " The sampling seed determines the selected name blocks and does not represent a neural training seed. "
        "Sampling is performed without replacement from alphabetically sorted group names using Python's "
        "seeded pseudorandom generator. All records in each selected group are retained in their original "
        "input order. Each method operates on the selected records independently, and the reported scores "
        "are recomputed on their aligned evaluation cohort. No encoder or generative model is fine-tuned "
        "on this cohort. The reported settings describe the saved runs, rather than any later changes "
        "to experiment scripts. Threshold selection is not documented as a separate validation experiment."
    )
    semantic = summaries.get("HHC-SE", {})
    gm = summaries.get("HHC-GM", {})
    settings += (
        "\n\n#### 4.2.2 Semantic representations and candidate retrieval\n\n"
        f"HHC-SE used the pretrained `{semantic.get('model', 'Not recorded')}` encoder "
        f"on {str(semantic.get('device', 'Not recorded')).replace('cuda', 'CUDA').replace('cpu', 'CPU')}, with an additional semantic merge "
        f"threshold of {semantic.get('semantic_threshold', 'Not recorded')}. This threshold "
        "is a cosine-similarity cutoff rather than an estimated same-author probability. "
        f"HHC-GM used `{gm.get('embedding_model', 'Not recorded')}` for candidate retrieval "
        f"with {gm.get('candidate_score', 'Not recorded')} scoring. Candidate pairs were retained "
        f"at a similarity cutoff of {gm.get('semantic_candidate_threshold', 'Not recorded')}, "
        f"with at most {gm.get('semantic_top_k', 'Not recorded')} neighbors per cluster "
        "(a value of zero denotes no neighbor limit). A pair was eligible when selected by "
        "either cluster. Retrieval identified pairs for LLM assessment and did not itself authorize a merge. "
        f"The retrieval encoder processed batches of {gm.get('embedding_batch_size', 'Not recorded')} "
        f"records with a maximum sequence length of {gm.get('embedding_max_length', 'Not recorded')} tokens. "
    )
    if "batch_size" in semantic and "max_length" in semantic:
        settings += (
            f"For HHC-SE, batch size and maximum sequence length were "
            f"{semantic['batch_size']} and {semantic['max_length']}, respectively."
        )
    else:
        settings += "The HHC-SE metadata do not record its encoder batch size and maximum sequence length."
    retries = gm.get("llm_retries", "Not recorded")
    quantization = {True: "enabled", False: "disabled"}.get(
        gm.get("cpu_int8"), "not recorded"
    )
    settings += (
        "\n\n#### 4.2.3 Generative model and decision criteria\n\n"
        f"Cluster-pair assessment used `{gm.get('model', 'Not recorded')}` "
        f"at revision `{gm.get('model_revision', 'Not recorded')}` on "
        f"{str(gm.get('device', 'Not recorded')).replace('cuda', 'CUDA').replace('cpu', 'CPU')}. Each prompt included at most "
        f"{gm.get('max_records_per_cluster', 'Not recorded')} selected publications per cluster. "
        f"Input length was limited to {gm.get('max_input_tokens', 'Not recorded')} tokens, "
        f"and generation was limited to {gm.get('max_new_tokens', 'Not recorded')} new tokens. "
        f"Malformed responses permitted up to {retries} additional "
        f"{'attempt' if retries == 1 else 'attempts'}. "
        "Inference used greedy decoding, with thinking disabled for Qwen. "
        f"A valid positive response authorized a merge only when its stated same-author "
        f"probability was at least {gm.get('llm_confidence_threshold', 'Not recorded')}. "
        f"The comparison budget was {gm.get('max_llm_comparisons_per_group', 'Not recorded')} "
        "cluster-pair assessments per name block (zero denotes an unlimited budget). "
        "Retries can increase the number of inference calls without increasing this pair budget. "
        f"CPU int8 quantization was {quantization}, with a configured CPU thread count of "
        f"{gm.get('cpu_threads', 'Not recorded')}."
    )
    settings += (
        "\n\n#### 4.2.4 Computational environment\n\n"
        + build_machine_configuration(data, output_dir)
    )
    prompt_path = (
        Path(output_dir) / Path(gm.get("prompts_output", "hhc_gm_prompts.json")).name
    )
    settings += "\n\n#### 4.2.5 LLM prompt and response protocol\n\n"
    if prompt_path.exists():
        entries = json.loads(prompt_path.read_text(encoding="utf-8"))
        prompts = [
            entry["prompt"] for entry in entries if isinstance(entry.get("prompt"), str)
        ]
        if prompts:
            instructions = prompts[0].split("\nInput:\n", 1)[0]
            variants = {prompt.split("\nInput:\n", 1)[0] for prompt in prompts}
            settings += (
                "The following instruction text is extracted verbatim from the first saved prompt "
                f"in [the run's prompt log]({link(prompt_path)}), rather than reconstructed from current "
                f"source defaults. The log contains {len(prompts):,} submitted prompts with "
                f"{len(variants):,} distinct instruction "
                f"{'prefix' if len(variants) == 1 else 'prefixes'}.\n\n```text\n"
                + instructions.replace("```", "` ` `")
                + "\n```\n\n"
                "Each instruction is followed by `Input:` and a `<clusters>` JSON payload with "
                "clusters `A` and `B`. Each contains `focal_author_names`, `total_publications`, "
                "and selected `publications` with `title`, `venue`, and `collaborators`. "
                "`name_evidence` records compatible names and the presence of full names. "
                "The payload closes with `</clusters>` and `Answer:`. Full populated prompts "
                "are optional supporting artifacts.\n\n"
                "The response fields are `same_author` (boolean), `same_author_probability` "
                "(number from 0 to 1), and `reason` (evidence). The prompt's 0.5 cutoff defines "
                "consistency between its boolean and probability. The experiment's higher merge "
                "threshold is applied afterwards. The log's separate `label` field is ground truth "
                "for auditing and is never sent as part of the prompt."
            )
            # Include a complete small input from the run, without ground-truth labels.
            payload_example = None
            for prompt in sorted(prompts, key=len):
                try:
                    payload = json.loads(
                        prompt.split("<clusters>\n", 1)[1].split("\n</clusters>", 1)[0]
                    )
                except (IndexError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict) and all(
                    key in payload for key in ("A", "B", "name_evidence")
                ):
                    payload_example = payload
                    break
            if payload_example is not None:
                settings += (
                    "\n\nThe following is the complete populated input from the shortest recorded "
                    "prompt. It follows the instructions above. The example was selected for brevity, "
                    "not as evidence of model accuracy.\n\n```text\nInput:\n<clusters>\n"
                    + json.dumps(payload_example, ensure_ascii=False, indent=2).replace(
                        "```", "` ` `"
                    )
                    + "\n</clusters>\nAnswer:\n```\n\n"
                    "A syntactically valid response has the following form. This response is "
                    "illustrative and is not an observed judgment of the example above.\n\n```json\n"
                    + json.dumps(
                        {
                            "same_author": True,
                            "same_author_probability": 0.9,
                            "reason": "Matching focal names and a specific research connection support identity.",
                        },
                        indent=2,
                    )
                    + "\n```\n\n"
                    "The boolean and probability must agree with the prompt's 0.5 decision rule. "
                    "A positive judgment below the experimental merge threshold leaves the clusters "
                    "separate. Malformed responses are retried within the stated limit, and unresolved "
                    "or oversized comparisons leave clusters separate. For Qwen, the prompt is supplied "
                    "as a user message through the model's chat template with thinking disabled."
                )
        else:
            settings += "The saved prompt log contains no sent prompts. No run-specific prompt is available."
    else:
        settings += "No saved prompt log was found for this run. Its exact prompt cannot be verified from these artifacts."
    settings += (
        "\n\n#### 4.2.6 Evaluation protocol\n\n"
        "B-cubed F1 is the primary metric. Let N be the number of evaluated publication "
        "occurrences, C(i) the predicted cluster containing occurrence i, and G(i) its "
        "ground-truth author cluster. Record-level precision and recall are averaged over "
        "all N occurrences, giving equal weight to each occurrence rather than each author:\n\n"
        "$$\nP_{B^3}=\\frac{1}{N}\\sum_{i=1}^{N}\\frac{|C(i)\\cap G(i)|}{|C(i)|},"
        "\\qquad R_{B^3}=\\frac{1}{N}\\sum_{i=1}^{N}\\frac{|C(i)\\cap G(i)|}{|G(i)|}.\n$$\n\n"
        "$$\nF_{B^3}=\\frac{2P_{B^3}R_{B^3}}{P_{B^3}+R_{B^3}}.\n$$\n\n"
        "For example, if a predicted cluster contains three records, two belonging to the "
        "same real author, those two records each receive precision 2/3. If that author "
        "has four records in the evaluation cohort, their record-level recall is 2/4. "
        "The final B-cubed F1 is the harmonic mean of the averaged precision and recall, "
        "not an average of individual F1 scores.\n\n"
        "Pairwise scores provide secondary diagnostics. A true positive (TP) is a pair "
        "with the same ground-truth author and predicted cluster. A false positive (FP) "
        "joins different authors, and a false negative (FN) separates the same author. "
        "A true negative (TN) correctly separates different authors. Pairwise precision "
        "is TP/(TP+FP), recall is TP/(TP+FN), and F1 is their harmonic mean. The implementation "
        "assigns precision or recall a value of one when its denominator is zero. "
        "Global pairwise and B-cubed metrics use ground-truth identity across the entire "
        "aligned cohort. The error-change analysis restricts pairs to the same ambiguous-name "
        "block and compares each extension with HHC on identical record pairs.\n\n"
        "Within each name block, average cluster purity (ACP) equals its record-averaged "
        "cluster precision and average author purity (AAP) equals its record-averaged author recall. For block b, K(b)=sqrt(ACP(b) × AAP(b)). "
        "Reported ACP, AAP, and K are arithmetic means over blocks, giving equal weight "
        "to each block. Higher values indicate better performance for all reported metrics.\n\n"
        "#### 4.2.7 Scope and reproducibility limits\n\n"
        "The study reports one execution per method on the selected cohort. No repeated-seed "
        "confidence intervals or significance tests were performed. The model revision `main` "
        "is mutable, and exact checkpoint hashes were not preserved. Missing historical settings "
        "are not reconstructed from current defaults or later experiment scripts. "
        "A fully reproducible follow-up should record all inference settings "
        "and hardware, pin model revisions, and select thresholds on disjoint validation groups "
        "before evaluating a final test cohort. The present results describe this experiment "
        "without establishing general superiority."
    )
    results = (
        "The B-cubed F1 scores were "
        + ", ".join(
            f"{scores[method]['b3_f1']:.4f} for {method}" for method in data.methods
        )
        + "."
    )
    results += (
        "\n\nChanges in B-cubed F1 relative to HHC (absolute percentage points): "
        + ", ".join(
            f"{(scores[method]['b3_f1'] - scores[baseline]['b3_f1']) * 100:+.3f} percentage points for {method}"
            for method in data.methods[1:]
        )
        + ". Figure 3 presents the B-cubed results, Figure 6 summarizes all evaluation metrics, and Figure 7 quantifies changes in pairwise errors."
    )
    changes = build_error_changes(data)
    results += (
        "\n\nA corrected false merge separates different real authors. A corrected false split "
        "joins records of the same author. Introduced errors are regressions on those same pairs. "
        "These are pair counts, not numbers of authors or cluster merges. Improvements from "
        "semantic evidence are not guaranteed to preserve HHC's exact merge sequence. HHC-GM "
        "starts with completed HHC clusters and only merges, so it cannot repair a baseline false merge."
        "\n\n### 5.1 Timing\n\nFigure 8 compares the reported execution times using a logarithmic scale. HHC includes data loading, preprocessing, clustering, evaluation, and prediction "
        "export. HHC-SE additionally accounts for embedding computation, and HHC-GM reports "
        "the measured duration of its recorded execution. Both neural methods "
        "exclude model setup/download. Metrics JSON writing occurs after the timers. These times "
        "reflect the recorded executions rather than a controlled hardware benchmark."
    )
    results += (
        "\n\nThe measured execution times (s, min, h) were "
        + ", ".join(
            f"{format_duration(summaries[method]['runtime_seconds'])} for {method}"
            for method in data.methods
        )
        + "."
    )
    if "llm_model_calls" in gm:
        results += (
            f" The GM run made {gm['llm_model_calls']} model calls, with "
            f"{gm.get('llm_merges', 'Not recorded')} accepted cluster merges."
        )
    diagnostics = summaries.get("HHC-GM", {}).get("decision_diagnostics", {})
    if diagnostics:
        results += (
            "\n\n### 5.2 LLM judgment audit\n\n"
            f"Evaluation of {diagnostics.get('labeled_valid_comparisons', 'Not recorded')} "
            "valid labeled cluster-pair judgments yielded "
            f"{diagnostics.get('true_positive', 'Not recorded')} correct matches, "
            f"{diagnostics.get('false_positive', 'Not recorded')} incorrect matches, "
            f"{diagnostics.get('true_negative', 'Not recorded')} correct "
            f"{'separation' if diagnostics.get('true_negative') == 1 else 'separations'}, and "
            f"{diagnostics.get('false_negative', 'Not recorded')} missed matches. "
        )
        brier = diagnostics.get("brier_score")
        if brier is not None:
            results += f"The Brier score was {brier:.4f}. "
        results += (
            "These judgments concern retrieved cluster pairs "
            "and differ from the publication-pair errors in Figure 7. The stated probabilities "
            "are not assumed calibrated. Brier score evaluates their squared probability error. "
            "Candidate selection and earlier merges condition this audit sample."
        )
        tp, fn = (
            diagnostics.get("true_positive", 0),
            diagnostics.get("false_negative", 0),
        )
        if tp + fn:
            results += (
                f" At the configured threshold, the judge accepts {tp:,} of {tp + fn:,} "
                f"labeled same-author comparisons (recall {tp / (tp + fn):.1%}), "
                "indicating missed merge opportunities among the reviewed candidates."
            )
    if not summaries.get("HHC-GM", {}).get("retrieval_diagnostics"):
        results += (
            "\n\nCandidate-retrieval coverage was not recorded for the current GM run."
        )
    results += (
        "\n\nSaved predictions and run metadata: "
        + ", ".join(
            f"[{method} predictions]({link(Path(output_dir) / (method.lower().replace('-', '_') + '_predictions.csv'))}), "
            f"[metrics]({link(Path(output_dir) / (method.lower().replace('-', '_') + '_metrics.json'))})"
            for method in data.methods
        )
        + "."
    )
    figure_blocks = {
        filename: f"**Figure {number}. {caption}.**\n\n![{caption}]({link(Path(report_dir) / filename)})"
        for number, (filename, caption) in enumerate(REPORT_FIGURES.items(), start=1)
    }
    methodology = (
        "Figure 1 summarizes the steps of all three methods, including their merge criteria "
        "and stopping conditions.\n\n" + figure_blocks.pop("methodology.png")
    )
    figures = "\n\n".join(figure_blocks.values())
    conclusion = (
        f"In the current shared cohort, {best} obtains the highest B-cubed F1 "
        f"({scores[best]['b3_f1']:.4f}). "
        + " ".join(
            f"{method} corrects {int(row['False splits corrected']):,} false splits and "
            f"{int(row['False merges corrected']):,} false merges, while introducing "
            f"{int(row['False splits introduced']):,} false splits and "
            f"{int(row['False merges introduced']):,} false merges versus {baseline}."
            for method, row in changes.iterrows()
        )
        + " These results support evaluating semantic retrieval and generative judgments as "
        "additional evidence rather than assuming topical similarity implies identity. "
        "A stronger study needs disjoint validation/test groups, repeated seeds, confidence intervals "
        "over name blocks, pinned checkpoints, and controlled runtime measurements. "
        "A splitting mechanism is also needed to recover from incorrect baseline merges."
    )
    return dict(
        zip(
            PAPER_SECTIONS,
            (abstract, methodology, dataset, settings, results, figures, conclusion),
            strict=True,
        )
    )


def update_readme_report(
    data: ReportData,
    readme_path: str | Path = "README.md",
    *,
    report_dir: str | Path = "figs",
    output_dir: str | Path = "outputs",
) -> Path:
    """Replace only explicit paper tags; fail before writing if tags are malformed."""
    path = Path(readme_path)
    content = path.read_text(encoding="utf-8")
    sections = build_paper_sections(
        data, readme_path=path, report_dir=report_dir, output_dir=output_dir
    )
    spans = []
    for name in PAPER_SECTIONS:
        start, end = f"<!-- HHC:{name}:START -->", f"<!-- HHC:{name}:END -->"
        if content.count(start) != 1 or content.count(end) != 1:
            raise ValueError(f"README requires exactly one {start} and {end}")
        left, right = content.index(start) + len(start), content.index(end)
        if left > right:
            raise ValueError(f"README markers are reversed for {name}")
        spans.append((left, right, name))
    ordered = sorted(spans)
    if any(
        left < previous_right
        for (_, previous_right, _), (left, _, _) in zip(ordered, ordered[1:])
    ):
        raise ValueError("README paper sections must not overlap")
    for left, right, name in reversed(ordered):
        content = content[:left] + "\n\n" + sections[name] + "\n\n" + content[right:]
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("figs"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument(
        "--no-update-readme",
        action="store_true",
        help="Generate figures without refreshing paper tags",
    )
    args = parser.parse_args()
    data = load_report_data(args.output_dir)
    for path in generate_reports(data, args.report_dir, show=args.show).values():
        print(path)
    if not args.no_update_readme:
        print(
            update_readme_report(
                data,
                args.readme,
                report_dir=args.report_dir,
                output_dir=args.output_dir,
            )
        )


if __name__ == "__main__":
    main()
