"""HHC-SE: Heuristic-Based Hierarchical Clustering with Semantic Evidence.

The classic HHC first stage and its title/venue merge rules are preserved.
HHC-SE adds cosine similarity between pretrained semantic paper embeddings as
one additional second-stage merge condition. SemCSE is the default encoder,
but the method name is independent of the selected embedding model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional
from tqdm import tqdm

try:
    from src import hhc
except ImportError:  # Supports ``python src/hhc_se.py``.
    import hhc  # type: ignore[no-redef]


DEFAULT_MODEL = "CLAUSE-Bielefeld/SemCSE_cosine"


@dataclass(slots=True)
class SemanticCluster:
    classic: hhc.Cluster
    embedding_sum: torch.Tensor

    @classmethod
    def from_classic(
        cls, cluster: hhc.Cluster, record_embeddings: torch.Tensor
    ) -> SemanticCluster:
        indices = [record.index for record in cluster.records]
        return cls(cluster, record_embeddings[indices].sum(dim=0))

    def similarity(self, other: SemanticCluster) -> float:
        return functional.cosine_similarity(
            self.embedding_sum.unsqueeze(0),
            other.embedding_sum.unsqueeze(0),
        ).item()

    def merge(self, other: SemanticCluster) -> None:
        self.classic.merge(other.classic)
        self.embedding_sum.add_(other.embedding_sum)


def document_text(row: dict[str, str]) -> str:
    """Build the scientific text passed to the encoder without using labels."""
    title = row["title"].strip()
    venue = row["venue"].strip()
    return f"Title: {title}\nVenue: {venue}" if venue else f"Title: {title}"


def embedding_fingerprint(
    rows: list[dict[str, str]], model_name: str, max_length: int
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{model_name}\0{max_length}\0".encode())
    for row in rows:
        digest.update(row["paper_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(document_text(row).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return requested


def encode_rows(
    rows: list[dict[str, str]],
    model_name: str,
    model_cache: Path,
    batch_size: int,
    max_length: int,
    device: str,
    cache_path: Path | None,
    rebuild_cache: bool,
    no_progress: bool,
) -> tuple[torch.Tensor, bool, float, float]:
    """Return embeddings, cache status, embedding time, and model setup time."""
    fingerprint = embedding_fingerprint(rows, model_name, max_length)
    if cache_path is not None and cache_path.exists() and not rebuild_cache:
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (
            isinstance(cached, dict)
            and cached.get("fingerprint") == fingerprint
            and cached.get("model") == model_name
            and isinstance(cached.get("embeddings"), torch.Tensor)
            and isinstance(cached.get("embedding_seconds"), (int, float))
        ):
            return (
                cached["embeddings"].float(),
                True,
                float(cached["embedding_seconds"]),
                0.0,
            )

    model_setup_started_at = time.perf_counter()
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformer dependencies are missing. Install them with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    model_cache.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache)
    model = AutoModel.from_pretrained(model_name, cache_dir=model_cache).to(device)
    model.eval()
    model_setup_seconds = time.perf_counter() - model_setup_started_at

    embedding_started_at = time.perf_counter()
    texts = [document_text(row) for row in rows]
    batches: list[torch.Tensor] = []
    starts = range(0, len(texts), batch_size)
    with torch.inference_mode():
        for start in tqdm(
            starts,
            total=(len(texts) + batch_size - 1) // batch_size,
            desc="Encoding semantic evidence",
            unit="batch",
            disable=no_progress,
        ):
            tokens = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            # This is the pooling used by the official SemCSE training code:
            # the [CLS] token from the final hidden layer.
            embeddings = model(**tokens).last_hidden_state[:, 0]
            embeddings = functional.normalize(embeddings, p=2, dim=-1)
            batches.append(embeddings.cpu())

    matrix = torch.cat(batches, dim=0).float()
    embedding_seconds = time.perf_counter() - embedding_started_at
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fingerprint": fingerprint,
                "model": model_name,
                "max_length": max_length,
                "embedding_seconds": embedding_seconds,
                "embeddings": matrix.half(),
            },
            cache_path,
        )
    return matrix, False, embedding_seconds, model_setup_seconds


def semantic_second_step(
    classic_clusters: list[hhc.Cluster],
    record_embeddings: torch.Tensor,
    title_threshold: float,
    venue_threshold: float,
    semantic_threshold: float,
) -> list[hhc.Cluster]:
    """Apply classic merge rules or the additional semantic cosine rule."""
    clusters = [
        SemanticCluster.from_classic(cluster, record_embeddings)
        for cluster in classic_clusters
    ]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                left = clusters[i]
                right = clusters[j]
                names_match = hhc.names_similar(
                    left.classic.names[0], right.classic.names[0]
                )
                classic_match = (
                    hhc.cosine(left.classic.title_terms, right.classic.title_terms)
                    > title_threshold
                    or hhc.cosine(left.classic.venue_terms, right.classic.venue_terms)
                    > venue_threshold
                )
                semantic_match = left.similarity(right) > semantic_threshold
                if names_match and (classic_match or semantic_match):
                    left.merge(right)
                    clusters.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1
    return [cluster.classic for cluster in clusters]


def write_predictions(
    path: Path, rows: list[dict[str, str]], predictions: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["paper_id", "ambiguous_name", "label", "predicted_cluster"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, prediction in zip(rows, predictions, strict=True):
            writer.writerow(
                {
                    "paper_id": row["paper_id"],
                    "ambiguous_name": row["ambiguous_name"],
                    "label": row.get("label", ""),
                    "predicted_cluster": prediction,
                }
            )


def run(args: argparse.Namespace) -> dict[str, float | int | str | bool]:
    started_at = time.perf_counter()
    rows = hhc.read_rows(args.input, args.n_ambiguous_group, args.seed)
    device = resolve_device(args.device)

    embeddings, cache_hit, embedding_seconds, model_setup_seconds = encode_rows(
        rows,
        args.model,
        args.model_cache,
        args.batch_size,
        args.max_length,
        device,
        args.embedding_cache,
        args.rebuild_cache,
        args.no_progress,
    )
    groups: dict[str, list[hhc.Record]] = defaultdict(list)
    for index, row in enumerate(
        tqdm(
            rows, desc="Preparing HHC records", unit="record", disable=args.no_progress
        )
    ):
        record = hhc.make_record(index, row)
        groups[record.ambiguous_name].append(record)

    predictions = [""] * len(rows)
    for ambiguous_name, records in tqdm(
        groups.items(),
        total=len(groups),
        desc="HHC-SE clustering",
        unit="group",
        disable=args.no_progress,
    ):
        seed_clusters = hhc.first_step(records)
        clusters = semantic_second_step(
            seed_clusters,
            embeddings,
            args.title_threshold,
            args.venue_threshold,
            args.semantic_threshold,
        )
        for number, cluster in enumerate(clusters, start=1):
            cluster_id = f"{ambiguous_name}::{number:04d}"
            for record in cluster.records:
                predictions[record.index] = cluster_id

    write_predictions(args.output, rows, predictions)
    summary: dict[str, float | int | str | bool] = {
        "method": "HHC-SE",
        "model": args.model,
        "model_cache": str(args.model_cache),
        "input": str(args.input),
        "output": str(args.output),
        "ambiguous_groups": len(groups),
        "n_ambiguous_group": args.n_ambiguous_group,
        "seed": args.seed,
        "title_threshold": args.title_threshold,
        "venue_threshold": args.venue_threshold,
        "semantic_threshold": args.semantic_threshold,
        "device": device,
        "embedding_cache_hit": cache_hit,
        "embedding_seconds": round(embedding_seconds, 3),
        "model_setup_seconds_excluded": round(model_setup_seconds, 3),
    }
    if rows and "label" in rows[0]:
        summary.update(hhc.evaluate(rows, predictions))
    current_run_seconds = time.perf_counter() - started_at - model_setup_seconds
    summary["current_run_seconds"] = round(current_run_seconds, 3)
    summary["runtime_seconds"] = round(
        current_run_seconds + (embedding_seconds if cache_hit else 0.0), 3
    )
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs/hhc_se_predictions.csv"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("outputs/hhc_se_metrics.json"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path("models"),
        help="directory for downloaded Hugging Face models (default: models)",
    )
    parser.add_argument("--semantic-threshold", type=float, default=0.80)
    parser.add_argument("--title-threshold", type=float, default=0.30)
    parser.add_argument("--venue-threshold", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("outputs/semcse_cosine_embeddings.pt"),
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    hhc.add_group_selection_args(parser)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    hhc.validate_group_selection_args(parser, args)
    if not -1 <= args.semantic_threshold <= 1:
        parser.error("--semantic-threshold must be between -1 and 1")
    for name in ("title_threshold", "venue_threshold"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.batch_size < 1 or args.max_length < 1:
        parser.error("--batch-size and --max-length must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
