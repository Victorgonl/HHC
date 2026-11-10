"""Reusable report loading, scoring, and plotting for HHC experiments.

Plot functions accept ReportData and an output path, use PlotStyle, return the
saved Path, and close their figure. Display is opt-in with show=True. Styling is
scoped with rc_context so importing or plotting does not change global settings.
"""

from __future__ import annotations

import argparse
from functools import wraps
import json
import os
import random
from numbers import Integral, Real
from dataclasses import dataclass, field
from html import escape
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext

try:
    from src.hhc import evaluate
except ImportError:
    from hhc import evaluate

def format_number(value, spec: str = ",") -> str:
    """Formata números em pt-BR sem alterar o locale global do processo."""
    return format(value, spec).translate(str.maketrans({",": ".", ".": ","}))


METHODS = ("HHC", "HHC-SE", "HHC-GM")
METRICS = (
    ("Precisão B-cubed", "b3_precision"),
    ("Revocação B-cubed", "b3_recall"),
    ("F1 B-cubed", "b3_f1"),
    ("Precisão pairwise", "pairwise_precision"),
    ("Revocação pairwise", "pairwise_recall"),
    ("F1 pairwise", "pairwise_f1"),
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
                    f"{method}: os metadados do experimento identificam outro método"
                )
            for key, value in metadata.get("parameters", {}).items():
                if method == "HHC-GM" and key == "retrieval_diagnostics":
                    # Older metrics overwrite the boolean argument with results.
                    summary_key = "retrieval_diagnostics_enabled"
                    if summary_key in run_summaries[method]:
                        if value != run_summaries[method][summary_key]:
                            raise ValueError(
                                f"{method}: o parâmetro experimental {key!r} diverge das métricas"
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
                        f"{method}: o parâmetro experimental {key!r} diverge das métricas"
                    )
            experiment_metadata[method] = metadata
        frame = pd.read_csv(output_dir / f"{stem}_predictions.csv", dtype=str)
        required = set(key_columns + ["predicted_cluster"])
        if missing := required - set(frame.columns):
            raise ValueError(f"{method}: colunas de predição ausentes: {sorted(missing)}")
        if frame[list(required)].isna().any().any():
            raise ValueError(
                f"{method}: as predições do relatório exigem identificadores e rótulos completos"
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
            "Alguns registros do HHC-GM não têm predições de todos os métodos"
        )
    if common.empty:
        raise ValueError("Nenhuma publicação disponível para comparação")
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
        raise ValueError("Informe input_path para o CSV original de publicações")
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
            "Alguns registros da base comum estão ausentes do CSV original de publicações"
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
        raise ValueError("O número de grupos e de registros por grupo deve ser positivo")
    if min_real_authors < 1:
        raise ValueError("O número mínimo de autores reais deve ser positivo")
    author_counts = data.common.groupby("ambiguous_name")["label"].nunique()
    names = sorted(author_counts[author_counts >= min_real_authors].index)
    if not names:
        raise ValueError(f"Nenhum grupo tem pelo menos {min_real_authors} autores reais")
    selected = random.Random(seed).sample(names, k=min(group_count, len(names)))
    examples = _source_records(
        data, data.common.loc[data.common["ambiguous_name"].isin(selected)], input_path
    )
    columns = {
        "paper_id": "ID da publicação",
        "author_name": "Nome do autor",
        "coauthors": "Coautores",
        "title": "Título",
        "venue": "Veículo",
        "label": "ID do autor verdadeiro",
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
            f"<p>{format_number(len(records), ',')} registros · {format_number(records['label'].nunique(), ',')} autores reais"
            f" · Exibindo {format_number(len(shown), ',')} de {format_number(len(records), ',')} registros"
            f" ({format_number(shown['label'].nunique(), ',')} autores exibidos) · Semente {seed}</p>"
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
    "TP": "Verdadeiro positivo — mesmo autor, unido corretamente",
    "TN": "Verdadeiro negativo — autores diferentes, separados corretamente",
    "FP": "Falso positivo — autores diferentes, unidos incorretamente",
    "FN": "Falso negativo — mesmo autor, separado incorretamente",
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
        raise ValueError("O número de exemplos por resultado deve ser positivo")
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
        f"<p>Pares de registros no mesmo bloco · Semente {seed}. Positivo = mesmo cluster previsto.</p>"
    ]
    for method, outcomes in results.items():
        sections.append(f"<h2>{escape(method)}</h2>")
        prediction_key = f"predicted_cluster_{method.lower().replace('-', '_')}"
        for outcome, bucket in outcomes.items():
            sections.append(
                f"<h3>{escape(PAIR_OUTCOMES[outcome])}</h3>"
                f"<p>{format_number(bucket['count'], ',')} pares elegíveis</p>"
            )
            if not bucket["pairs"]:
                sections.append("<p>Nenhum exemplo disponível na base comum.</p>")
            for number, pair in enumerate(bucket["pairs"], start=1):
                rows = []
                for letter, position in zip(("A", "B"), pair, strict=True):
                    record = records_by_position[position]
                    fields = {"Registro": letter}
                    for key, label in (
                        ("paper_id", "ID da publicação"),
                        ("author_name", "Nome do autor"),
                        ("coauthors", "Coautores"),
                        ("title", "Título"),
                        ("venue", "Veículo"),
                        ("label", "ID do autor verdadeiro"),
                    ):
                        if key in record:
                            fields[label] = record[key]
                    fields["Cluster previsto"] = data.common.iloc[position][
                        prediction_key
                    ]
                    rows.append(fields)
                name = data.common.iloc[pair[0]]["ambiguous_name"]
                sections.append(
                    f"<h4>Exemplo {number} · {escape(str(name))}</h4>"
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
    """Conta erros FP/FN corrigidos ou introduzidos dentro dos blocos."""
    baseline, *extensions = data.methods
    counts = {
        method: dict.fromkeys(
            (
                "Fusões incorretas corrigidas",
                "Fusões incorretas introduzidas",
                "Separações incorretas corrigidas",
                "Separações incorretas introduzidas",
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
                error = "Separações incorretas" if truth else "Fusões incorretas"
                change = "corrigidas" if same == truth else "introduzidas"
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
                ("Fusões incorretas", "Separações incorretas"),
                ("Falsos positivos (fusões incorretas)", "Falsos negativos (separações incorretas)"),
                strict=True,
            ):
                for offset, change, color in (
                    (-0.18, "corrigidas", style.positive_text),
                    (0.18, "introduzidas", style.negative_text),
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
                        hatch="///" if change == "introduzidas" else None,
                    )
                    axis.bar_label(
                        bars, labels=[f"{format_number(value, ',')}" for value in values], padding=4
                    )
                axis.set_xticks(positions, changes.index)
                axis.set_title(title, fontsize=style.font_size)
                axis.set_ylim(
                    0,
                    max(
                        1,
                        changes[[f"{error} corrigidas", f"{error} introduzidas"]]
                        .to_numpy()
                        .max(),
                    )
                    * 1.25,
                )
                axis.set_ylabel("Número de pares de registros")
                axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_number(x, ",.0f")))
                axis.grid(axis="y", alpha=0.35)
                axis.set_axisbelow(True)
                axis.spines[["top", "right"]].set_visible(False)
            axes[1].legend(frameon=False, labelcolor=style.text)
            fig.suptitle(
                f"Erros corrigidos ou introduzidos em relação ao {baseline}",
                fontsize=style.title_size,
                fontweight="normal",
            )
            fig.text(
                0.5,
                0.01,
                "Pares no mesmo bloco · Escalas distintas · Menos erros introduzidos é melhor",
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
        return f"{format_number(seconds, '.3f')}s"
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
        raise ValueError("Informe um rótulo para cada métrica")
    if len(style.colors) < len(metric_keys):
        raise ValueError("O estilo precisa de pelo menos uma cor por métrica")
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
                    labels=[f"{format_number(value, '.3f')}" for value in values],
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
            axis.set_ylabel("Pontuação")
            axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_number(x, ".1f")))
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
        labels=("Precisão", "Revocação", "F1"),
        title="Métricas pairwise",
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
    """Save Precisão B-cubed, recall, and F1."""
    return plot_metrics(
        data,
        output_path,
        metric_keys=("b3_precision", "b3_recall", "b3_f1"),
        labels=("Precisão", "Revocação", "F1"),
        title="Métricas B-cubed",
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
        title="Métricas de pureza e K",
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
        ("Publicações / grupo", papers.groupby("ambiguous_name")["paper_id"].nunique()),
        ("Publicações / autor", papers.groupby("label")["paper_id"].nunique()),
        ("Autores / grupo", authors.groupby("ambiguous_name").size()),
    )
    columns = [
        "Registros comparados",
        "Publicações distintas",
        "Grupos ambíguos",
        "Autores reais",
    ]
    values = [
        f"{format_number(len(common), ',')}",
        f"{format_number(common['paper_id'].nunique(), ',')}",
        f"{format_number(common['ambiguous_name'].nunique(), ',')}",
        f"{format_number(common['label'].nunique(), ',')}",
    ]
    for label, counts in distributions:
        columns.append(f"{label}\n(Mín/Máx/Média)")
        values.append(f"{format_number(counts.min(), ',')} / {format_number(counts.max(), ',')} / {format_number(counts.mean(), '.2f')}")
    return columns, [values]


def build_summary_rows(
    data: ReportData,
) -> tuple[list[str], list[list[str]], list[dict[str, float]]]:
    """Build the comparison table, with each extension's delta beside its scores."""
    baseline, *extensions = data.methods
    columns = ["Estatística", baseline]
    for method in extensions:
        columns.extend([method, f"{method}\nΔ em relação ao {baseline}"])

    def row(label, values, deltas=None):
        cells = [label, values[baseline]]
        for method in extensions:
            cells.extend([values[method], (deltas or {}).get(method, "—")])
        return cells

    methods, scores, summaries = data.methods, data.scores, data.run_summaries
    table_rows = [
        row("Registros da execução", {m: f"{format_number(summaries[m]['records'], ',')}" for m in methods}),
        row(
            "Tempo total de execução",
            {m: format_duration(summaries[m]["runtime_seconds"]) for m in methods},
        ),
        row(
            "Clusters previstos",
            {m: f"{format_number(scores[m]['predicted_clusters'], ',')}" for m in methods},
            {
                m: f"{format_number(scores[m]['predicted_clusters'] - scores[baseline]['predicted_clusters'], '+,')}"
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
                {m: f"{format_number(scores[m][key], '.3f')}" for m in methods},
                {m: f"{format_number(delta, '+.3f')} pp" for m, delta in deltas.items()},
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
                "Dados comuns de avaliação (todos os métodos)", fontsize=style.font_size, pad=8
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
                if row[0] == "F1 B-cubed"
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
        raise ValueError("Os tempos de execução devem ser finitos e positivos para uma escala logarítmica")
    hhc_seconds = float(data.run_summaries["HHC"]["runtime_seconds"])
    if not np.isfinite(hhc_seconds) or hhc_seconds <= 0:
        raise ValueError("O tempo de execução do HHC deve ser finito e positivo")
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
                    label += f"\n({format_number(duration / hhc_seconds, '.1f')}× mais lento)"
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
            axis.set_xlabel("Tempo de execução relatado (s, escala logarítmica)")
            axis.set_title("Tempo de execução", fontsize=style.title_size, pad=12)
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
                "Método: etapas comuns do HHC e extensões",
                ha="center",
                fontsize=style.title_size,
                weight="bold",
            )
            ax.text(
                7,
                9.65,
                "Cada método opera independentemente dentro de um bloco de nomes ambíguos",
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
                "Inicialização comum do HHC",
                "Normalizar nomes, títulos e veículos; iniciar clusters\ncom nomes compatíveis e coautores em comum",
                hhc,
                width=8,
            )
            arrow([(4, 7.91), (4, 7.55), (2.3, 7.55), (2.3, 7.14)], hhc)
            arrow([(10, 7.91), (10, 7.55), (11.7, 7.55), (11.7, 7.14)], se)
            box(
                2.3,
                6.5,
                "HHC · fusão lexical",
                "Nomes compatíveis E similaridade\nde título OU veículo acima do limiar",
                hhc,
            )
            arrow([(2.3, 5.86), (2.3, 5.14)], hhc)
            ax.text(
                2.3,
                5.5,
                "Agregar evidências; repetir até estabilizar",
                ha="center",
                fontsize=8,
                color=hhc,
            )
            box(
                2.3,
                4.5,
                "HHC · clusters finais de autores",
                "Saída do método de referência\nClusters iniciais do HHC-GM",
                hhc,
            )
            arrow([(4.24, 4.5), (5.06, 4.5)], gm)
            box(
                7,
                4.5,
                "HHC-GM · recuperar pares",
                "Ordenar clusters compatíveis por SemCSE\nCorte de similaridade + filtro top-k",
                gm,
            )
            arrow([(7, 3.86), (7, 3.14)], gm)
            box(
                7,
                2.5,
                "HHC-GM · julgar e unir",
                "Selecionar evidências; validar JSON do LLM\nResposta positiva + limiar de probabilidade",
                gm,
            )
            arrow([(7, 1.86), (7, 1.34)], gm)
            box(
                7,
                0.7,
                "HHC-GM · clusters finais de autores",
                "Parar: sem par aprovado ou sem orçamento\nNunca dividir os clusters originais do HHC",
                gm,
            )
            arrow([(5.06, 2.5), (4.65, 2.5), (4.65, 5.45), (7, 5.45), (7, 5.14)], gm)
            ax.text(
                6.35,
                5.63,
                "Após uma fusão: atualizar clusters e recuperar novamente",
                ha="center",
                fontsize=8,
                color=gm,
            )
            box(
                11.7,
                6.5,
                "HHC-SE · evidência semântica",
                "Codificar título + veículo com SemCSE\nSomar embeddings normalizados",
                se,
            )
            arrow([(11.7, 5.86), (11.7, 5.14)], se)
            box(
                11.7,
                4.5,
                "HHC-SE · fusão estendida",
                "Nomes compatíveis E similaridade acima\ndo limiar de título, veículo OU embedding",
                se,
            )
            arrow([(11.7, 3.86), (11.7, 3.14)], se)
            ax.text(
                11.7,
                3.5,
                "Atualizar evidências lexicais e vetoriais",
                ha="center",
                fontsize=8,
                color=se,
            )
            box(
                11.7,
                2.5,
                "HHC-SE · clusters finais de autores",
                "Intercalar fusões lexicais e semânticas\nParar: passagem completa sem fusão",
                se,
            )
            ax.legend(
                handles=[
                    Line2D([0], [0], color=color, lw=3, label=label)
                    for color, label in (
                        (hhc, "HHC / etapas comuns de referência"),
                        (se, "Extensão HHC-SE"),
                        (gm, "Extensão HHC-GM"),
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
                "Rótulos verdadeiros são reservados à avaliação e aos logs de auditoria.",
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
    """Gera as figuras em português com sufixo -pt_BR e retorna seus caminhos."""
    plots = {
        "data_statistics-pt_BR.png": plot_data_statistics,
        "pairwise_metrics-pt_BR.png": plot_pairwise,
        "bcubed_metrics-pt_BR.png": plot_bcubed,
        "purity_k_metrics-pt_BR.png": plot_purity,
        "reports_table-pt_BR.png": plot_summary_table,
        "error_changes-pt_BR.png": plot_error_changes,
        "execution_time-pt_BR.png": plot_execution_time,
        "methodology-pt_BR.png": plot_methodology,
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
    'methodology-pt_BR.png': 'Fluxos dos métodos HHC, HHC-SE e HHC-GM. O HHC-SE intercala fusões lexicais e semânticas; o HHC-GM acrescenta recuperação e julgamentos generativos após a execução completa do HHC',
    'data_statistics-pt_BR.png': 'Dados compartilhados: totais e distribuições de mínimo, máximo e média',
    'bcubed_metrics-pt_BR.png': 'Precisão, revocação e F1 B-cubed nos registros alinhados (métrica principal)',
    'pairwise_metrics-pt_BR.png': 'Precisão, revocação e F1 pairwise nos registros alinhados (métrica secundária)',
    'purity_k_metrics-pt_BR.png': 'Pureza dos grupos (ACP e AAP) e K com média macro',
    'reports_table-pt_BR.png': 'Métricas completas, tempos de execução e mudanças em relação ao HHC',
    'error_changes-pt_BR.png': 'Fusões e separações incorretas corrigidas ou introduzidas em relação ao HHC, dentro dos blocos de nomes',
    'execution_time-pt_BR.png': 'Tempo de execução relatado em segundos, com eixo logarítmico. A preparação dos modelos neurais é excluída. As durações refletem as execuções registradas, não um benchmark com hardware controlado',
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
        ("Núcleos lógicos de CPU", ("cpu", "logical_processors")),
        ("RAM", ("system_memory_mib",)),
        ("GPU", ("gpu", "model")),
        ("Memória da GPU", ("gpu", "memory_mib")),
        ("Sistema operacional", ("operating_system",)),
        ("Arquitetura", ("architecture",)),
        ("Python", ("python_version",)),
        ("PyTorch", ("pytorch_version",)),
        ("Runtime CUDA (PyTorch)", ("pytorch_cuda_version",)),
        ("Driver NVIDIA", ("gpu", "driver_version")),
    )

    def cell(environment, path):
        value = environment
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is None:
            return "Não registrado"
        if path[-1] in ("system_memory_mib", "memory_mib"):
            value = f"{format_number(value / 1024, '.2f')} GiB"
        return escape(str(value)).replace("|", "&#124;").replace("\n", " ")

    rows = [
        [label, *(cell(env, path) for env in environments)] for label, path in fields
    ]
    # Collapse identical machines into one configuration column.
    shared = all(len(set(row[1:]) - {"Não registrado"}) <= 1 for row in rows)
    header = ["Componente", "Configuração"] if shared else ["Componente", *data.methods]
    if shared:
        # Retain which runs recorded software details without duplicating hardware.
        combined = []
        for row in rows:
            known = [
                (method, value)
                for method, value in zip(data.methods, row[1:])
                if value != "Não registrado"
            ]
            value = known[0][1] if known else "Não registrado"
            if known and len(known) < len(data.methods):
                value += (
                    " (registrado para " + ", ".join(method for method, _ in known) + ")"
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
    return _configuration_table_html(rows, "Configuração da máquina", style)


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
        labels = {
            "input": "Arquivo de entrada",
            "n_ambiguous_group": "Número de grupos ambíguos",
            "seed": "Semente",
            "device": "Dispositivo",
            "resolved_device": "Dispositivo utilizado",
            "model": "Modelo",
            "model_revision": "Revisão do modelo",
            "title_threshold": "Limiar de título",
            "venue_threshold": "Limiar de veículo",
            "semantic_threshold": "Limiar semântico",
            "batch_size": "Tamanho do lote",
            "max_length": "Comprimento máximo",
            "embedding_model": "Modelo de embeddings",
            "embedding_batch_size": "Lote de embeddings",
            "embedding_max_length": "Comprimento máximo dos embeddings",
            "llm_confidence_threshold": "Limiar de confiança do LLM",
            "semantic_candidate_threshold": "Limiar de candidatos semânticos",
            "semantic_top_k": "Número máximo de vizinhos",
            "candidate_score": "Pontuação dos candidatos",
            "max_llm_comparisons_per_group": "Máximo de comparações por grupo",
            "max_records_per_cluster": "Máximo de registros por cluster",
            "max_input_tokens": "Máximo de tokens de entrada",
            "max_new_tokens": "Máximo de novos tokens",
            "llm_retries": "Tentativas adicionais do LLM",
            "cpu_threads": "Threads de CPU",
            "cpu_int8": "Quantização int8 em CPU",
            "retrieval_diagnostics": "Diagnósticos de recuperação",
        }

        def display_value(value):
            if isinstance(value, bool):
                return "Sim" if value else "Não"
            if isinstance(value, Real):
                return format_number(value, "," if isinstance(value, Integral) else "g")
            return {"centroid": "Centroide", "hybrid": "Híbrida", "auto": "Automático"}.get(str(value), str(value))

        rows = [["Parâmetro", "Valor"]]
        rows.extend(
            [escape(f"{labels.get(key, key)} ({key})"), escape(display_value(value))]
            for key, value in sorted(parameters.items())
        )
        if len(rows) == 1:
            rows.append(["Parâmetros", "Não registrado"])
        tables.append(_configuration_table_html(rows, f"Parâmetros de {method}", style))
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
    readme_path: str | Path = "README-pt_BR.md",
    report_dir: str | Path = "figs",
    output_dir: str | Path = "outputs",
) -> dict[str, str]:
    """Gera as seções do artigo em português a partir dos resultados salvos."""
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

    def numeric(value, spec=""):
        if isinstance(value, Real):
            return format_number(value, spec or ("," if isinstance(value, Integral) else "g"))
        return str(value)

    def setting(summary, key):
        return numeric(summary.get(key, "Não registrado"))

    def device(summary):
        return str(summary.get("device", "Não registrado")).replace("cuda", "CUDA").replace("cpu", "CPU")

    score_list = ", ".join(
        f"{numeric(scores[method]['b3_f1'], '.4f')} para {method}" for method in data.methods
    )
    abstract = (
        "*Este trabalho estuda a desambiguação de nomes de autores (Author Name Disambiguation, AND) "
        "usando agrupamento hierárquico heurístico (HHC), uma extensão com evidências semânticas "
        "(HHC-SE) e uma extensão com modelo generativo (HHC-GM). Em um subconjunto do LAGOS-AND, "
        f"a base comum de avaliação contém {numeric(len(data.common))} registros de "
        f"{numeric(data.common['ambiguous_name'].nunique())} grupos de nomes ambíguos e "
        f"{numeric(data.common['label'].nunique())} autores reais. O F1 B-cubed é {score_list}. "
        f"O {best} apresenta o maior F1 B-cubed no experimento relatado. "
        "Os resultados descrevem um experimento por método, sem teste de significância.*"
    )
    dataset = (
        "A avaliação usa um subconjunto do LAGOS-AND [[6]](#ref-6), com identidades verdadeiras "
        "baseadas em ORCID no campo `label`. A avaliação alinha "
        "`(paper_id, ambiguous_name, label, occurrence)` entre os métodos, preservando todas "
        "as ocorrências do HHC-GM e exigindo correspondência nas outras duas execuções. "
        "As execuções de referência e semântica podem conter registros adicionais. Ocorrências "
        "duplicadas permanecem na avaliação, enquanto as contagens descritivas de publicações "
        "usam identificadores distintos.\n\nA Figura 2 apresenta as estatísticas dos dados "
        "compartilhados.\n\nOs autores são identificados por seus ORCIDs verdadeiros. O número "
        "de autores por grupo de nome ambíguo é calculado independentemente em cada grupo, "
        "e uma publicação pode aparecer em mais de um grupo."
    )
    grouped_settings = {}
    for method in data.methods:
        summary = summaries[method]
        key = tuple(summary.get(field, "Não registrado") for field in (
            "input", "seed", "records", "n_ambiguous_group", "title_threshold", "venue_threshold"
        ))
        grouped_settings.setdefault(key, []).append(method)
    descriptions = []
    for (source, seed, records, groups, title, venue), methods in grouped_settings.items():
        names = methods[0] if len(methods) == 1 else ", ".join(methods[:-1]) + " e " + methods[-1]
        verb = "foi avaliado" if len(methods) == 1 else "foram avaliados"
        descriptions.append(
            f"{names} {verb} usando `{source}`, com {numeric(records)} registros de publicações "
            f"e uma amostra solicitada de {numeric(groups)} grupos completos de nomes ambíguos. "
            f"A amostragem dos grupos usou semente aleatória {numeric(seed)}. Os limiares de "
            f"similaridade de cosseno para título e veículo foram definidos como {numeric(title)} "
            f"e {numeric(venue)}, respectivamente. A fusão lexical exigiu nomes compatíveis e "
            "similaridade superior a pelo menos um desses limiares."
        )
    settings = "#### 4.2.1 Subconjunto experimental e limiares lexicais\n\n" + "\n\n".join(descriptions)
    settings += (
        "\n\nA semente de amostragem determina os blocos de nomes selecionados e não representa "
        "uma semente de treinamento neural. A amostragem é feita sem reposição a partir dos nomes "
        "de grupos ordenados alfabeticamente, usando o gerador pseudoaleatório do Python com "
        "semente definida. Todos os registros de cada grupo selecionado são preservados na ordem "
        "original de entrada. Cada método opera independentemente sobre os registros selecionados, "
        "e as pontuações relatadas são recalculadas sobre a base alinhada de avaliação. Nenhum "
        "codificador ou modelo generativo é ajustado nessa base. As configurações relatadas "
        "descrevem as execuções salvas, e não alterações posteriores nos scripts. A seleção dos "
        "limiares não está documentada como um experimento separado de validação."
    )
    semantic = summaries.get("HHC-SE", {})
    gm = summaries.get("HHC-GM", {})
    candidate_score = {"centroid": "centroide", "hybrid": "híbrida"}.get(
        gm.get("candidate_score"), gm.get("candidate_score", "Não registrado")
    )
    settings += (
        "\n\n#### 4.2.2 Representações semânticas e recuperação de candidatos\n\n"
        f"O HHC-SE usou o codificador pré-treinado `{setting(semantic, 'model')}` em "
        f"{device(semantic)}, com limiar adicional de fusão semântica de "
        f"{setting(semantic, 'semantic_threshold')}. Esse limiar é um corte de similaridade "
        "de cosseno, não uma probabilidade estimada de identidade de autoria. "
        f"O HHC-GM usou `{setting(gm, 'embedding_model')}` para recuperação de candidatos "
        f"com pontuação por {candidate_score}. Os pares candidatos foram mantidos com corte "
        f"de similaridade de {setting(gm, 'semantic_candidate_threshold')} e no máximo "
        f"{setting(gm, 'semantic_top_k')} vizinhos por cluster; zero indica ausência de limite "
        "de vizinhos. Um par era elegível quando selecionado por qualquer um dos clusters. "
        "A recuperação identificava pares para avaliação pelo LLM e não autorizava fusões "
        f"diretamente. O codificador de recuperação processou lotes de {setting(gm, 'embedding_batch_size')} "
        f"registros com comprimento máximo de sequência de {setting(gm, 'embedding_max_length')} tokens. "
    )
    if "batch_size" in semantic and "max_length" in semantic:
        settings += (
            "Para o HHC-SE, o tamanho do lote e o comprimento máximo foram "
            f"{setting(semantic, 'batch_size')} e {setting(semantic, 'max_length')}, respectivamente."
        )
    else:
        settings += "Os metadados do HHC-SE não registram o tamanho do lote nem o comprimento máximo do codificador."
    quantization = {True: "ativada", False: "desativada"}.get(gm.get("cpu_int8"), "não registrada")
    settings += (
        "\n\n#### 4.2.3 Modelo generativo e critérios de decisão\n\n"
        f"A avaliação de pares de clusters usou `{setting(gm, 'model')}`, na revisão "
        f"`{setting(gm, 'model_revision')}`, em {device(gm)}. Cada prompt incluiu no máximo "
        f"{setting(gm, 'max_records_per_cluster')} publicações selecionadas por cluster. "
        f"A entrada foi limitada a {setting(gm, 'max_input_tokens')} tokens e a geração a "
        f"{setting(gm, 'max_new_tokens')} novos tokens. O limite de tentativas adicionais "
        f"para respostas malformadas foi {setting(gm, 'llm_retries')}. "
        "A inferência usou decodificação gulosa (`do_sample=False`), com o modo de raciocínio "
        "do Qwen desativado. Nenhuma temperatura de amostragem foi fornecida explicitamente "
        "a `generate`; a temperatura não é aplicada porque a amostragem de tokens está "
        "desativada. Portanto, a configuração é descrita como decodificação gulosa, sem "
        "afirmar que `temperature=0` foi configurado explicitamente.\n\nUma resposta positiva "
        "válida autorizava uma fusão apenas quando a probabilidade declarada de identidade "
        f"de autoria era de pelo menos {setting(gm, 'llm_confidence_threshold')}. "
        f"O orçamento foi de {setting(gm, 'max_llm_comparisons_per_group')} avaliações de pares "
        "de clusters por bloco de nomes; zero indica orçamento ilimitado. Novas tentativas "
        "podem aumentar o número de chamadas de inferência sem aumentar esse orçamento de pares. "
        f"A quantização int8 em CPU estava {quantization}, com "
        f"{setting(gm, 'cpu_threads')} threads de CPU configuradas."
    )
    settings += "\n\n#### 4.2.4 Ambiente computacional\n\n" + build_machine_configuration(data, output_dir)
    prompt_path = Path(output_dir) / Path(gm.get("prompts_output", "hhc_gm_prompts.json")).name
    settings += "\n\n#### 4.2.5 Protocolo de prompt e resposta do LLM\n\n"
    settings += (
        '**Configuração zero-shot.** Cada chamada contém as instruções de decisão, os campos '
        'exigidos para a resposta JSON e as evidências dos clusters `A` e `B`. Não são fornecidos '
        'pares anteriores acompanhados de decisões corretas, como ocorreria em uma configuração '
        'few-shot. As publicações recuperadas e selecionadas compõem os dados da comparação atual; '
        'não são demonstrações rotuladas. A descrição do formato JSON também não constitui um '
        'exemplo resolvido.\n'
        '\n'
        'O template de chat recebe uma única mensagem de usuário, sem histórico de julgamentos '
        'anteriores. Os rótulos verdadeiros não entram no prompt e são usados apenas na avaliação e'
        ' na auditoria. Nas novas tentativas por resposta inválida, acrescenta-se somente um '
        'lembrete sobre o formato JSON, sem fornecer a resposta correta. O modelo é usado com seus '
        'pesos pré-treinados, sem ajuste fino neste experimento. Zero-shot descreve, portanto, a '
        'ausência de demonstrações no contexto de inferência; não implica ausência de treinamento '
        'prévio nem garante que conteúdos relacionados estejam ausentes do pré-treinamento.\n'
        '\n'
        'A entrada preenchida e a resposta ilustrativa apresentadas abaixo documentam o protocolo '
        'do artigo. A resposta ilustrativa não é acrescentada aos prompts enviados ao modelo e não '
        'transforma o experimento em one-shot ou few-shot.\n'
        '\n'
    )
    if prompt_path.exists():
        entries = json.loads(prompt_path.read_text(encoding="utf-8"))
        prompts = [entry["prompt"] for entry in entries if isinstance(entry.get("prompt"), str)]
        if prompts:
            instructions = prompts[0].split("\nInput:\n", 1)[0]
            variants = {prompt.split("\nInput:\n", 1)[0] for prompt in prompts}
            settings += (
                "O texto de instrução abaixo foi extraído literalmente do primeiro prompt salvo "
                f"no [log de prompts da execução]({link(prompt_path)}), em vez de reconstruído "
                f"a partir dos valores atuais do código. O log contém {numeric(len(prompts))} "
                f"prompts enviados e {numeric(len(variants))} variantes de prefixo de instrução. "
                "Os prompts registrados são preservados no idioma original para manter o conteúdo "
                "do experimento.\n\n```text\n" + instructions.replace("```", "` ` `") + "\n```\n\n"
                "Cada instrução é seguida por `Input:` e por um conteúdo JSON delimitado por "
                "`<clusters>`, contendo os clusters `A` e `B`. Cada um contém `focal_author_names`, "
                "`total_publications` e as `publications` selecionadas, com `title`, `venue` e "
                "`collaborators`. `name_evidence` registra a compatibilidade dos nomes e a presença "
                "de nomes completos. O conteúdo termina com `</clusters>` e `Answer:`. Prompts "
                "completos preenchidos são artefatos complementares opcionais.\n\n"
                "Os campos de resposta são `same_author` (booleano), `same_author_probability` "
                "(número de 0 a 1) e `reason` (evidência). O corte de 0,5 do prompt define a "
                "consistência entre o booleano e a probabilidade. O limiar de fusão mais alto do "
                "experimento é aplicado depois. O campo separado `label` do log contém o rótulo "
                "verdadeiro para auditoria e nunca é enviado como parte do prompt."
            )
            populated = None
            for prompt in sorted(prompts, key=len):
                try:
                    payload = json.loads(prompt.split("<clusters>\n", 1)[1].split("\n</clusters>", 1)[0])
                except (IndexError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict) and all(key in payload for key in ("A", "B", "name_evidence")):
                    populated = payload
                    break
            if populated is not None:
                settings += (
                    "\n\nA seguir está a entrada completa do menor prompt registrado com estrutura "
                    "válida. Ela sucede as instruções acima. O exemplo foi selecionado por sua "
                    "brevidade, não como evidência de acurácia do modelo.\n\n```text\nInput:\n<clusters>\n"
                    + json.dumps(populated, ensure_ascii=False, indent=2, sort_keys=True).replace("```", "` ` `")
                    + "\n</clusters>\nAnswer:\n```\n\nUma resposta sintaticamente válida tem a forma "
                    "abaixo. Ela é ilustrativa e não corresponde a um julgamento observado para o "
                    "exemplo anterior.\n\n```json\n"
                    + json.dumps({
                        "same_author": True,
                        "same_author_probability": 0.9,
                        "reason": "Nomes focais coincidentes e uma conexão específica de pesquisa sustentam a identidade.",
                    }, ensure_ascii=False, indent=2)
                    + "\n```\n\nO booleano e a probabilidade devem concordar com a regra de decisão "
                    "de 0,5 do prompt. Um julgamento positivo abaixo do limiar experimental de fusão "
                    "mantém os clusters separados. Respostas malformadas são repetidas dentro do "
                    "limite informado, e comparações não resolvidas ou grandes demais mantêm os "
                    "clusters separados. Para o Qwen, o prompt é fornecido como mensagem de usuário "
                    "por meio do template de chat do modelo, com o modo de raciocínio desativado."
                )
        else:
            settings += "O log salvo não contém prompts enviados. Nenhum prompt específico da execução está disponível."
    else:
        settings += "Nenhum log de prompts salvo foi encontrado para esta execução. Seu prompt exato não pode ser verificado nesses artefatos."
    settings += (
        '\n'
        '\n'
        '#### 4.2.6 Protocolo de avaliação\n'
        '\n'
        'O F1 B-cubed é a métrica principal. Seja N o número de ocorrências de publicações avaliadas,'
        ' C(i) o cluster previsto que contém a ocorrência i e G(i) o cluster de seu autor verdadeiro.'
        ' A precisão e a revocação por registro são calculadas e depois recebem média sobre todas as '
        'N ocorrências, com peso igual para cada ocorrência, em vez de cada autor:\n'
        '\n'
        '$$\n'
        'P_{B^3}=\\frac{1}{N}\\sum_{i=1}^{N}\\frac{|C(i)\\cap G(i)|}{|C(i)|},\\qquad '
        'R_{B^3}=\\frac{1}{N}\\sum_{i=1}^{N}\\frac{|C(i)\\cap G(i)|}{|G(i)|}.\n'
        '$$\n'
        '\n'
        '$$\n'
        'F_{B^3}=\\frac{2P_{B^3}R_{B^3}}{P_{B^3}+R_{B^3}}.\n'
        '$$\n'
        '\n'
        'Por exemplo, se um cluster previsto contém três registros, dos quais dois pertencem ao mesmo'
        ' autor real, esses dois registros recebem precisão 2/3 cada. Se esse autor tem quatro '
        'registros na base avaliada, a revocação de cada um desses registros é 2/4. O F1 B-cubed '
        'final é a média harmônica da precisão e da revocação já agregadas, não uma média dos F1 '
        'individuais.\n'
        '\n'
        'As métricas pairwise fornecem diagnósticos secundários. Um verdadeiro positivo (TP) é um par'
        ' com o mesmo autor verdadeiro e o mesmo cluster previsto. Um falso positivo (FP) une autores'
        ' diferentes, e um falso negativo (FN) separa registros do mesmo autor. Um verdadeiro '
        'negativo (TN) separa corretamente autores diferentes. A precisão pairwise é TP/(TP+FP), a '
        'revocação é TP/(TP+FN), e o F1 é sua média harmônica. A implementação atribui valor um à '
        'precisão ou à revocação quando o respectivo denominador é zero. As métricas globais pairwise'
        ' e B-cubed usam a identidade verdadeira do autor em toda a base alinhada. A análise de '
        'mudanças nos erros restringe os pares ao mesmo bloco de nomes ambíguos e compara cada '
        'extensão ao HHC sobre pares idênticos de registros.\n'
        '\n'
        'Dentro de cada bloco de nomes, a pureza média dos clusters (ACP) corresponde à média por '
        'registro da precisão de cluster, e a pureza média dos autores (AAP) corresponde à média por '
        'registro da revocação de autor. Para o bloco b, K(b)=sqrt(ACP(b) × AAP(b)). Os valores '
        'relatados de ACP, AAP e K são médias aritméticas entre blocos, com peso igual para cada '
        'bloco. Valores maiores indicam melhor desempenho em todas as métricas de agrupamento '
        'desta seção; para o escore de Brier da Seção 5.2, valores menores indicam melhores '
        'predições probabilísticas.\n'
        '\n'
        'As unidades de agregação diferem entre as métricas. Um grupo é um bloco de nomes ambíguos '
        '(`ambiguous_name`), não um cluster previsto. O B-cubed usa média global por registro: um '
        'bloco com 100 ocorrências contribui dez vezes mais que um bloco com 10. As métricas pairwise'
        ' usam contagens globais de pares, não uma média das métricas por bloco; um conjunto de n '
        'registros contém n(n−1)/2 pares não ordenados, de modo que grandes clusters verdadeiros ou '
        'previstos podem influenciar desproporcionalmente o resultado. ACP, AAP e K usam média macro '
        'entre blocos: blocos com 100 e 10 ocorrências têm o mesmo peso final. Em particular, o K '
        'relatado é a média das raízes quadradas calculadas em cada bloco, não a raiz quadrada do '
        'produto das médias relatadas de ACP e AAP.\n'
        '\n'
        'Para B blocos, as médias macro são:\n'
        '\n'
        '$$\n'
        'ACP=\\frac{1}{B}\\sum_{b=1}^{B}ACP_b,\\qquad AAP=\\frac{1}{B}\\sum_{b=1}^{B}AAP_b,\\qquad '
        'K=\\frac{1}{B}\\sum_{b=1}^{B}\\sqrt{ACP_b\\,AAP_b}.\n'
        '$$\n'
        '\n'
        'Todas as métricas comparativas são recalculadas sobre as mesmas ocorrências de publicações '
        'alinhadas para os três métodos, preservando ocorrências duplicadas. Os F1 relatados são '
        'calculados a partir da precisão e da revocação agregadas, e não pela média dos F1 entre '
        'blocos de nomes. Os gráficos de mudanças nos erros somam pares de publicações corrigidos e '
        'introduzidos dentro dos blocos. Os tempos são totais das execuções salvas, não médias por '
        'grupo nem médias de execuções repetidas.\n'
        '\n'
        '#### 4.2.7 Escopo e limites de reprodutibilidade\n'
        '\n'
        'O estudo relata uma execução por método no subconjunto selecionado. Não foram calculados '
        'intervalos de confiança com sementes repetidas nem realizados testes de significância. A '
        'revisão `main` do modelo é mutável, e os hashes exatos dos checkpoints não foram '
        'preservados. Configurações históricas ausentes não são reconstruídas a partir dos valores '
        'atuais ou de scripts experimentais posteriores. Um estudo posterior plenamente reproduzível '
        'deve registrar todas as configurações de inferência e o hardware, fixar as revisões dos '
        'modelos e selecionar limiares em grupos de validação separados antes de avaliar a base final'
        ' de teste. Os resultados atuais descrevem este experimento sem estabelecer superioridade '
        'geral.'
    )
    results = (
        f"Os valores de F1 B-cubed foram {score_list}.\n\n"
        "As mudanças no F1 B-cubed em relação ao HHC, em pontos percentuais absolutos, foram "
        + ", ".join(
            f"{numeric((scores[method]['b3_f1'] - scores[baseline]['b3_f1']) * 100, '+.3f')} "
            f"para {method}" for method in data.methods if method != baseline
        )
        + ". A Figura 3 apresenta os resultados B-cubed, a Figura 6 resume todas as métricas "
        "e a Figura 7 quantifica as mudanças nos erros de pares.\n\n"
        "Uma fusão incorreta corrigida separa autores reais diferentes. Uma separação incorreta "
        "corrigida une registros do mesmo autor. Erros introduzidos são regressões nesses mesmos "
        "pares. Essas são contagens de pares, não de autores ou de fusões de clusters. Melhorias "
        "na evidência semântica não garantem a preservação da sequência exata de fusões do HHC. "
        "O HHC-GM parte dos clusters completos do HHC e apenas os une; portanto, não pode reparar "
        "uma fusão incorreta do método de referência.\n\n### 5.1 Tempo de execução\n\n"
        "A Figura 8 compara os tempos relatados usando escala logarítmica. O HHC inclui "
        "carregamento dos dados, pré-processamento, agrupamento, avaliação e exportação das "
        "predições. O HHC-SE também inclui o cálculo dos embeddings, e o HHC-GM apresenta a "
        "duração medida de sua execução registrada. Ambos os métodos neurais excluem a preparação "
        "e o download dos modelos. A gravação do JSON de métricas ocorre após a medição. Esses "
        "tempos refletem as execuções registradas, não um benchmark com hardware controlado.\n\n"
        "Os tempos medidos foram "
        + ", ".join(f"{format_duration(summaries[method]['runtime_seconds'])} para {method}" for method in data.methods)
        + "."
    )
    if "llm_model_calls" in gm:
        results += (
            f" A execução GM realizou {setting(gm, 'llm_model_calls')} chamadas ao modelo, "
            f"com {setting(gm, 'llm_merges')} fusões de clusters aceitas."
        )
    diagnostics = summaries.get("HHC-GM", {}).get("decision_diagnostics", {})
    if diagnostics:
        results += (
            "\n\n### 5.2 Auditoria dos julgamentos do LLM\n\n"
            f"A avaliação de {setting(diagnostics, 'labeled_valid_comparisons')} julgamentos "
            "válidos e rotulados de pares de clusters encontrou "
            f"{setting(diagnostics, 'true_positive')} correspondências corretas, "
            f"{setting(diagnostics, 'false_positive')} correspondências incorretas, "
            f"{setting(diagnostics, 'true_negative')} separações corretas e "
            f"{setting(diagnostics, 'false_negative')} correspondências não identificadas. "
        )
        brier = diagnostics.get("brier_score")
        if brier is not None:
            results += f"O escore de Brier foi {numeric(brier, '.4f')}. "
        results += (
            "Esses julgamentos se referem a pares de clusters recuperados e diferem dos erros "
            "de pares de publicações da Figura 7. A seleção de candidatos e as fusões anteriores "
            "condicionam essa amostra de auditoria."
        )
        tp, fn = diagnostics.get("true_positive", 0), diagnostics.get("false_negative", 0)
        if tp + fn:
            results += (
                f" No limiar configurado, o julgador aceita {numeric(tp)} das {numeric(tp + fn)} "
                f"comparações rotuladas como sendo do mesmo autor (revocação de {numeric(tp / (tp + fn), '.1%')}), "
                "indicando oportunidades de fusão não aproveitadas entre os candidatos avaliados."
            )
    if diagnostics.get("brier_score") is not None:
        results += (
            "\n\nO **escore de Brier** mede o erro das probabilidades declaradas pelo LLM. "
            "Para cada comparação válida e rotulada j, seja p_j o valor de "
            "`same_author_probability` e y_j o rótulo verdadeiro: 1 para o mesmo autor e "
            "0 para autores diferentes. Com M comparações, calcula-se:\n\n"
            "$$\nBS=\\frac{1}{M}\\sum_{j=1}^{M}(p_j-y_j)^2.\n$$\n\n"
            "Cada comparação tem o mesmo peso; não se calcula uma média por grupo de nome "
            "ambíguo nem por par de publicações. Apenas respostas válidas com rótulo booleano "
            "entram no cálculo. O escore usa a probabilidade original, antes da aplicação do "
            f"limiar de fusão de {setting(gm, 'llm_confidence_threshold')}; alterar somente esse "
            "limiar não altera o Brier sobre as "
            "mesmas comparações e probabilidades.\n\n"
            "Nesta formulação binária, o escore varia de 0 a 1, e **quanto menor, melhor**. "
            "Zero significa probabilidades perfeitamente corretas; um corresponde ao erro "
            "máximo, com certeza na classe errada em todas as comparações. Por exemplo, "
            "p=0,9 produz erro 0,01 quando y=1, mas 0,81 quando y=0. Prever p=0,5 em todas "
            "as comparações produz Brier 0,25, qualquer que seja o rótulo. Assim, erros "
            "confiantes recebem penalidade maior.\n\n"
            f"O valor observado de {numeric(diagnostics['brier_score'], '.4f')} é a média "
            "dos erros quadráticos, não uma taxa de erro nem uma acurácia. "
        )
        zero_bins = [
            bucket for bucket in diagnostics.get("confidence_bins", [])
            if bucket.get("count", 0) and bucket.get("mean_probability") == 0
        ]
        if zero_bins:
            zero_count = sum(bucket["count"] for bucket in zero_bins)
            zero_positives = round(sum(
                bucket["count"] * bucket["observed_same_author_rate"] for bucket in zero_bins
            ))
            results += (
                f"Nos diagnósticos salvos, {numeric(zero_count)} comparações receberam p=0, "
                f"embora {numeric(zero_positives)} fossem do mesmo autor. Cada uma dessas "
                "correspondências recebeu penalidade 1, contribuindo fortemente para o escore alto. "
            )
        results += (
            "O Brier avalia a qualidade probabilística global e depende também da capacidade "
            "de distinguir as classes e da proporção de positivos; isoladamente, não mede "
            "apenas calibração. Como a auditoria usa candidatos selecionados durante o "
            "agrupamento, seu valor descreve essa amostra e não demonstra calibração ou "
            "desempenho probabilístico em uma base independente."
        )
    if not gm.get("retrieval_diagnostics"):
        results += "\n\nA cobertura da recuperação de candidatos não foi registrada na execução GM atual."
    results += "\n\nPredições e metadados salvos: " + ", ".join(
        f"[predições {method}]({link(Path(output_dir) / (method.lower().replace('-', '_') + '_predictions.csv'))}), "
        f"[métricas {method}]({link(Path(output_dir) / (method.lower().replace('-', '_') + '_metrics.json'))})"
        for method in data.methods
    ) + "."
    figure_blocks = {
        filename: f"**Figura {number}. {caption}.**\n\n![{caption}]({link(Path(report_dir) / filename)})"
        for number, (filename, caption) in enumerate(REPORT_FIGURES.items(), start=1)
    }
    methodology = (
        "A Figura 1 resume as etapas dos três métodos, incluindo seus critérios de fusão "
        "e condições de parada.\n\n" + figure_blocks.pop("methodology-pt_BR.png")
    )
    figures = "\n\n".join(figure_blocks.values())
    changes = build_error_changes(data)
    conclusion = (
        f"Na base comum atual, o {best} obtém o maior F1 B-cubed "
        f"({numeric(scores[best]['b3_f1'], '.4f')}). "
        + " ".join(
            f"O {method} corrige {numeric(int(row['Separações incorretas corrigidas']))} separações incorretas "
            f"e {numeric(int(row['Fusões incorretas corrigidas']))} fusões incorretas, enquanto introduz "
            f"{numeric(int(row['Separações incorretas introduzidas']))} separações incorretas e "
            f"{numeric(int(row['Fusões incorretas introduzidas']))} fusões incorretas em relação ao {baseline}."
            for method, row in changes.iterrows()
        )
        + " Esses resultados apoiam a avaliação da recuperação semântica e dos julgamentos "
        "generativos como evidências adicionais, sem presumir que similaridade temática implique "
        "identidade. Um estudo mais robusto requer grupos separados de validação e teste, múltiplas "
        "sementes, intervalos de confiança por blocos de nomes, checkpoints fixados e medições de "
        "tempo com hardware controlado. Também é necessário um mecanismo de divisão para corrigir "
        "fusões incorretas do método de referência."
    )
    return dict(zip(PAPER_SECTIONS, (abstract, methodology, dataset, settings, results, figures, conclusion), strict=True))


def update_readme_report(
    data: ReportData,
    readme_path: str | Path = "README-pt_BR.md",
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
            raise ValueError(f"O README deve conter exatamente um {start} e um {end}")
        left, right = content.index(start) + len(start), content.index(end)
        if left > right:
            raise ValueError(f"Os marcadores do README estão invertidos para {name}")
        spans.append((left, right, name))
    ordered = sorted(spans)
    if any(
        left < previous_right
        for (_, previous_right, _), (left, _, _) in zip(ordered, ordered[1:])
    ):
        raise ValueError("As seções do artigo no README não podem se sobrepor")
    for left, right, name in reversed(ordered):
        content = content[:left] + "\n\n" + sections[name] + "\n\n" + content[right:]
    path.write_text(content, encoding="utf-8")
    return path


# English text and layouts share the data model and language-neutral helpers above.

_EN_METRICS = (
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


def _en_load_report_data(output_dir: str | Path) -> ReportData:
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


def _en_source_records(
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


def _en_build_group_examples_html(
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
    examples = _en_source_records(
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


_EN_PAIR_OUTCOMES = {
    "TP": "True positive — same author, correctly merged",
    "TN": "True negative — different authors, correctly separated",
    "FP": "False positive — different authors, incorrectly merged",
    "FN": "False negative — same author, incorrectly separated",
}


def _en_select_model_examples(
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
        method: {outcome: {"count": 0, "pairs": []} for outcome in _EN_PAIR_OUTCOMES}
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


def _en_display_model_examples(
    data: ReportData,
    *,
    seed: int = 42,
    examples_per_outcome: int = 1,
    input_path: str | Path | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> None:
    """Display source records illustrating each model's pairwise successes/errors."""
    from IPython.display import HTML, display

    results = _en_select_model_examples(
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
        source = _en_source_records(data, data.common.iloc[positions], input_path)
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
                f"<h3>{escape(_EN_PAIR_OUTCOMES[outcome])}</h3>"
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


def _en_build_error_changes(data: ReportData) -> pd.DataFrame:
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


def _en_plot_error_changes(
    data: ReportData,
    output_path: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> Path:
    """Plot error corrections and regressions on the shared evaluation cohort."""
    changes = _en_build_error_changes(data)
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


def _en_format_duration(seconds: float) -> str:
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


def _en_plot_metrics(
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


def _en_plot_pairwise(
    data: ReportData,
    output_path: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> Path:
    """Save pairwise precision, recall, and F1."""
    return _en_plot_metrics(
        data,
        output_path,
        metric_keys=("pairwise_precision", "pairwise_recall", "pairwise_f1"),
        labels=("Precision", "Recall", "F1"),
        title="Pairwise metrics",
        style=style,
        show=show,
    )


def _en_plot_bcubed(
    data: ReportData,
    output_path: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> Path:
    """Save B-cubed precision, recall, and F1."""
    return _en_plot_metrics(
        data,
        output_path,
        metric_keys=("b3_precision", "b3_recall", "b3_f1"),
        labels=("Precision", "Recall", "F1"),
        title="B-cubed metrics",
        style=style,
        show=show,
    )


def _en_plot_purity(
    data: ReportData,
    output_path: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> Path:
    """Save average cluster purity, average author purity, and K."""
    return _en_plot_metrics(
        data,
        output_path,
        metric_keys=("average_cluster_purity", "average_author_purity", "k_metric"),
        labels=("ACP", "AAP", "K"),
        title="Purity and K metrics",
        style=style,
        show=show,
    )


def _en_build_data_statistics_rows(data: ReportData) -> tuple[list[str], list[list[str]]]:
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


def _en_build_summary_rows(
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
            {m: _en_format_duration(summaries[m]["runtime_seconds"]) for m in methods},
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
    for label, key in _EN_METRICS:
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


def _en_plot_data_statistics(
    data: ReportData,
    output_path: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> Path:
    """Save a standalone table describing the shared evaluation cohort."""
    statistics_columns, statistics_rows = _en_build_data_statistics_rows(data)
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


def _en_plot_summary_table(
    data: ReportData,
    output_path: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> Path:
    """Save experiment scores and extension deltas against HHC."""
    columns, rows, changes = _en_build_summary_rows(data)
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


def _en_plot_execution_time(
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
                label = _en_format_duration(duration)
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


def _en_plot_methodology(
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


def _en_generate_reports(
    data: ReportData,
    report_dir: str | Path,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    show: bool = False,
) -> dict[str, Path]:
    """Generate all standard report images and return their paths."""
    plots = {
        "data_statistics.png": _en_plot_data_statistics,
        "pairwise_metrics.png": _en_plot_pairwise,
        "bcubed_metrics.png": _en_plot_bcubed,
        "purity_k_metrics.png": _en_plot_purity,
        "reports_table.png": _en_plot_summary_table,
        "error_changes.png": _en_plot_error_changes,
        "execution_time.png": _en_plot_execution_time,
        "methodology.png": _en_plot_methodology,
    }
    return {
        name: plot(data, Path(report_dir) / name, style=style, show=show)
        for name, plot in plots.items()
    }


_EN_REPORT_FIGURES = {
    "methodology.png": "Method workflows for HHC, HHC-SE, and HHC-GM. HHC-SE interleaves lexical and semantic merging; HHC-GM adds retrieval and generative judgments after complete HHC",
    "data_statistics.png": "Shared data: totals and min/max/average distributions",
    "bcubed_metrics.png": "B-cubed precision, recall, and F1 on aligned records (primary metric)",
    "pairwise_metrics.png": "Pairwise precision, recall, and F1 on aligned records (secondary metric)",
    "purity_k_metrics.png": "Macro-averaged group purity (ACP, AAP) and K",
    "reports_table.png": "Complete scores, run times, and changes against HHC",
    "error_changes.png": "Within-name false merges and false splits corrected or introduced against HHC",
    "execution_time.png": "Reported execution time in seconds (logarithmic axis). Neural model setup is excluded. Durations reflect the recorded executions and are not a controlled hardware benchmark",
}


def _en_build_machine_configuration(
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


def _en_build_machine_configuration_html(
    data: ReportData,
    output_dir: str | Path = "outputs",
    *,
    style: PlotStyle = DEFAULT_STYLE,
) -> str:
    """Render the machine table using the report palette and typography."""
    lines = _en_build_machine_configuration(data, output_dir).splitlines()
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for index, line in enumerate(lines)
        if index != 1
    ]
    return _configuration_table_html(rows, "Machine configuration", style)


def _en_build_model_parameters_html(
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


def _en_build_paper_sections(
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
        "Inference used greedy decoding (`do_sample=False`), with thinking disabled for Qwen. No sampling temperature was explicitly supplied to `generate`; temperature is not applied because token sampling is disabled. This should be reported as greedy decoding, rather than as an explicitly configured `temperature=0`. "
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
        + _en_build_machine_configuration(data, output_dir)
    )
    prompt_path = (
        Path(output_dir) / Path(gm.get("prompts_output", "hhc_gm_prompts.json")).name
    )
    settings += "\n\n#### 4.2.5 LLM prompt and response protocol\n\n"
    settings += (
        '**Zero-shot setup.** Each call contains the decision instructions, the required JSON response '
        'fields, and evidence from clusters `A` and `B`. No previous pairs accompanied by correct '
        'decisions are supplied, as they would be in a few-shot setup. Retrieved and selected '
        'publications are the data of the current comparison, not labeled demonstrations. Describing the'
        ' JSON format is also not a solved example.\n\nThe chat template receives a single user message, '
        'without a history of previous judgments. Ground-truth labels do not enter the prompt and are '
        'used only for evaluation and auditing. Retries after invalid responses add only a reminder '
        'about the JSON format, without supplying the correct answer. The model uses its pretrained '
        'weights without fine-tuning in this experiment. Zero-shot therefore describes the absence of '
        'demonstrations in the inference context; it does not imply an absence of prior training or '
        'guarantee that related content was absent from pretraining.\n\nThe populated input and '
        "illustrative response shown below document the paper's protocol. The illustrative response is "
        'not appended to the prompts sent to the model and does not make the experiment one-shot or '
        'few-shot.\n\n'
    )
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
        "to each block. Higher values indicate better performance for all clustering metrics in this section; the Brier score in Section 5.2 instead uses lower values to indicate better probabilistic predictions.\n\n"
        'The aggregation units differ across metrics. A group means an ambiguous-name block '
        '(`ambiguous_name`), not a predicted cluster. B-cubed uses a global record average: a block with'
        ' 100 occurrences contributes ten times as much as one with 10. Pairwise uses global pair '
        'counts, not a mean of block-level scores; a set of n records contains n(n−1)/2 unordered pairs,'
        ' so large author or predicted clusters can have a disproportionate influence. ACP, AAP, and K '
        'use a macro average over blocks: blocks with 100 and 10 occurrences receive the same final '
        'weight. In particular, the reported K is the mean of the block-specific square roots, not the '
        'square root of the product of the reported mean ACP and mean AAP.\n\nFor B blocks, the macro '
        'averages are:\n\n$$\nACP=\\frac{1}{B}\\sum_{b=1}^{B}ACP_b,\\qquad '
        'AAP=\\frac{1}{B}\\sum_{b=1}^{B}AAP_b,\\qquad K=\\frac{1}{B}\\sum_{b=1}^{B}\\sqrt{ACP_b\\,AAP_b}.\n$$\n\n'
        'All comparative scores are recomputed from the same aligned publication occurrences for the '
        'three methods, preserving duplicate occurrences. The reported F1 values are calculated from '
        'aggregated precision and recall, rather than averaged across name blocks. Error-change plots '
        'sum corrected and introduced publication pairs within name blocks. Execution times are totals '
        'for the saved runs, not averages over groups or repeated runs.\n\n'
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
    changes = _en_build_error_changes(data)
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
            f"{_en_format_duration(summaries[method]['runtime_seconds'])} for {method}"
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
            "and differ from the publication-pair errors in Figure 7. "
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
        if brier is not None:
            results += (
                '\n\nThe **Brier score** measures the error of the probabilities stated by the LLM. For each valid'
                ' labeled comparison j, let p_j be `same_author_probability` and y_j the ground-truth label: 1 '
                'for the same author and 0 for different authors. With M comparisons, it is calculated as:\n\n$$\n'
                'BS=\\frac{1}{M}\\sum_{j=1}^{M}(p_j-y_j)^2.\n$$\n\nEach comparison has equal weight; this is not a '
                'mean over ambiguous-name groups or publication pairs. Only valid responses with a boolean '
                'ground-truth label enter the calculation. The score uses the original probability, before '
                'applying the {gm.get("llm_min_probability", 0.7)} merge threshold; changing only that threshold'
                ' does not change Brier on the same comparisons and probabilities.\n\nIn this binary formulation, '
                'the score ranges from 0 to 1, and **lower is better**. Zero means perfectly correct '
                'probabilities; one is the maximum error, with certainty in the wrong class on every comparison.'
                ' For example, p=0.9 incurs error 0.01 when y=1, but 0.81 when y=0. Predicting p=0.5 for every '
                'comparison produces Brier 0.25 regardless of the labels. Confident errors therefore receive a '
                'larger penalty.\n\nThe observed value of {brier:.4f} is the mean squared probability error, not '
                'an error rate or accuracy. '
            ).replace('{gm.get("llm_min_probability", 0.7)}', str(gm.get('llm_confidence_threshold', 'Not recorded'))).replace('{brier:.4f}', f'{brier:.4f}')
            for bucket in diagnostics.get("confidence_bins", []):
                count = bucket.get("count", 0)
                rate = bucket.get("observed_same_author_rate")
                if count and bucket.get("mean_probability") == 0 and rate is not None:
                    positives = round(count * rate)
                    results += (
                        f"In the saved diagnostics, {count} comparisons received p=0 even though "
                        f"{positives} concerned the same author. Each of these missed matches incurred "
                        "a penalty of 1, contributing strongly to the high score. "
                    )
            results += (
                'Brier evaluates overall probabilistic quality and also depends on the ability to distinguish '
                'classes and on the positive-class proportion; it is not solely a measure of calibration. '
                'Because the audit uses candidates selected during clustering, its value describes this sample '
                'and does not establish calibration or probabilistic performance on an independent dataset.'
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
        for number, (filename, caption) in enumerate(_EN_REPORT_FIGURES.items(), start=1)
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


def _en_update_readme_report(
    data: ReportData,
    readme_path: str | Path = "README.md",
    *,
    report_dir: str | Path = "figs",
    output_dir: str | Path = "outputs",
) -> Path:
    """Replace only explicit paper tags; fail before writing if tags are malformed."""
    path = Path(readme_path)
    content = path.read_text(encoding="utf-8")
    sections = _en_build_paper_sections(
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


def _with_report_language(portuguese, english):
    """Select localized report text without changing process-wide state."""
    @wraps(portuguese)
    def localized(*args, language="pt_BR", **kwargs):
        if language not in ("en", "pt_BR"):
            raise ValueError("language must be 'en' or 'pt_BR'")
        implementation = english if language == "en" else portuguese
        return implementation(*args, **kwargs)
    return localized


build_data_statistics_rows = _with_report_language(build_data_statistics_rows, _en_build_data_statistics_rows)
build_error_changes = _with_report_language(build_error_changes, _en_build_error_changes)
build_group_examples_html = _with_report_language(build_group_examples_html, _en_build_group_examples_html)
build_machine_configuration = _with_report_language(build_machine_configuration, _en_build_machine_configuration)
build_machine_configuration_html = _with_report_language(build_machine_configuration_html, _en_build_machine_configuration_html)
build_model_parameters_html = _with_report_language(build_model_parameters_html, _en_build_model_parameters_html)
build_paper_sections = _with_report_language(build_paper_sections, _en_build_paper_sections)
build_summary_rows = _with_report_language(build_summary_rows, _en_build_summary_rows)
display_model_examples = _with_report_language(display_model_examples, _en_display_model_examples)
format_duration = _with_report_language(format_duration, _en_format_duration)
generate_reports = _with_report_language(generate_reports, _en_generate_reports)
load_report_data = _with_report_language(load_report_data, _en_load_report_data)
plot_bcubed = _with_report_language(plot_bcubed, _en_plot_bcubed)
plot_data_statistics = _with_report_language(plot_data_statistics, _en_plot_data_statistics)
plot_error_changes = _with_report_language(plot_error_changes, _en_plot_error_changes)
plot_execution_time = _with_report_language(plot_execution_time, _en_plot_execution_time)
plot_methodology = _with_report_language(plot_methodology, _en_plot_methodology)
plot_metrics = _with_report_language(plot_metrics, _en_plot_metrics)
plot_pairwise = _with_report_language(plot_pairwise, _en_plot_pairwise)
plot_purity = _with_report_language(plot_purity, _en_plot_purity)
plot_summary_table = _with_report_language(plot_summary_table, _en_plot_summary_table)
select_model_examples = _with_report_language(select_model_examples, _en_select_model_examples)
update_readme_report = _with_report_language(update_readme_report, _en_update_readme_report)


def _en_display_group_examples(
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
            _en_build_group_examples_html(
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


def _en_display_machine_configuration(
    data: ReportData,
    output_dir: str | Path = "outputs",
    *,
    style: PlotStyle = DEFAULT_STYLE,
) -> None:
    """Display the styled machine configuration in a notebook."""
    from IPython.display import HTML, display

    display(HTML(_en_build_machine_configuration_html(data, output_dir, style=style)))


def _en_display_model_parameters(
    data: ReportData, *, style: PlotStyle = DEFAULT_STYLE
) -> None:
    """Display the saved parameter tables in a notebook."""
    from IPython.display import HTML, display

    display(HTML(_en_build_model_parameters_html(data, style=style)))

display_group_examples = _with_report_language(display_group_examples, _en_display_group_examples)
display_machine_configuration = _with_report_language(display_machine_configuration, _en_display_machine_configuration)
display_model_parameters = _with_report_language(display_model_parameters, _en_display_model_parameters)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera gráficos, tabelas e artigo dos experimentos HHC em português brasileiro.",
        add_help=False,
    )
    parser._optionals.title = "Opções"
    parser.add_argument("-h", "--help", action="help", help="Mostrar esta ajuda e sair")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Diretório dos resultados salvos")
    parser.add_argument("--report-dir", type=Path, default=Path("figs"),
                        help="Diretório das figuras geradas")
    parser.add_argument("--show", action="store_true", help="Exibir as figuras geradas")
    parser.add_argument("--readme", type=Path, default=Path("README-pt_BR.md"),
                        help="Artigo a atualizar (padrão: README-pt_BR.md)")
    parser.add_argument(
        "--no-update-readme",
        action="store_true",
        help="Gerar figuras sem atualizar os marcadores do artigo",
    )
    parser.add_argument("--language", choices=("en", "pt_BR"), default="pt_BR")
    args = parser.parse_args()
    if args.language == "en" and args.readme == Path("README-pt_BR.md"):
        args.readme = Path("README.md")
    data = load_report_data(args.output_dir)
    for path in generate_reports(data, args.report_dir, show=args.show, language=args.language).values():
        print(path)
    if not args.no_update_readme:
        print(
            update_readme_report(
                data,
                args.readme,
                report_dir=args.report_dir,
                output_dir=args.output_dir,
                language=args.language,
            )
        )


if __name__ == "__main__":
    main()
