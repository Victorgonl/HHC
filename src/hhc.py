import argparse
import ast
import csv
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import snowballstemmer
from tqdm import tqdm

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "about",
    "after",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "between",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "over",
    "study",
    "the",
    "their",
    "through",
    "to",
    "toward",
    "towards",
    "under",
    "using",
    "via",
    "was",
    "we",
    "were",
    "with",
    "without",
}
STEMMER = snowballstemmer.stemmer("english")


def ascii_lower(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    return "".join(c for c in value if not unicodedata.combining(c)).lower()


def name_tokens(name: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(ascii_lower(name)))


def token_compatible(left: str, right: str) -> bool:
    return left == right or left[0] == right[0] and (len(left) == 1 or len(right) == 1)


def names_similar(left: str, right: str) -> bool:
    """Fragment-style, order-independent name comparison.

    Initials are compatible with full tokens (``R Bettati`` / ``Riccardo Bettati``),
    while conflicting initials or full tokens are not.  Order independence handles
    both ``S Rajakumar`` and ``Rajakumar S`` in this dataset.
    """
    a, b = name_tokens(left), name_tokens(right)
    if not a or not b:
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    used: set[int] = set()
    for token in sorted(short, key=len, reverse=True):
        match = next(
            (
                i
                for i, other in enumerate(long)
                if i not in used and token_compatible(token, other)
            ),
            None,
        )
        if match is None:
            return False
        used.add(match)
    return True


def coauthor_keys(name: str) -> set[str]:
    """Return keys that make initials and full-name forms meet efficiently."""
    tokens = name_tokens(name)
    if not tokens:
        return set()
    if len(tokens) == 1:
        return {f"only:{tokens[0]}"}
    keys = {"exact:" + ":".join(sorted(tokens))}
    for i, token in enumerate(tokens):
        if len(token) > 1:
            other_initials = "".join(
                sorted(t[0] for j, t in enumerate(tokens) if j != i)
            )
            keys.add(f"part:{token}:{other_initials}")
    return keys


def text_terms(text: str) -> Counter[str]:
    words = [
        w
        for w in TOKEN_RE.findall(ascii_lower(text))
        if w not in STOP_WORDS and len(w) > 1
    ]
    return Counter(STEMMER.stemWords(words))


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = left.keys() & right.keys()
    dot = sum(left[t] * right[t] for t in common)
    if not dot:
        return 0.0
    norm_left = math.sqrt(sum(v * v for v in left.values()))
    norm_right = math.sqrt(sum(v * v for v in right.values()))
    return dot / (norm_left * norm_right)


@dataclass(slots=True)
class Record:
    index: int
    ambiguous_name: str
    author_name: str
    coauthor_keys: set[str]
    title_terms: Counter[str]
    venue_terms: Counter[str]

    @property
    def long_name(self) -> bool:
        tokens = name_tokens(self.author_name)
        return len(tokens) > 1 and sum(len(t) > 1 for t in tokens) > 1


@dataclass(slots=True)
class Cluster:
    records: list[Record] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    coauthors: set[str] = field(default_factory=set)
    title_terms: Counter[str] = field(default_factory=Counter)
    venue_terms: Counter[str] = field(default_factory=Counter)

    @classmethod
    def from_record(cls, record: Record) -> "Cluster":
        return cls(
            records=[record],
            names=[record.author_name],
            coauthors=set(record.coauthor_keys),
            title_terms=record.title_terms.copy(),
            venue_terms=record.venue_terms.copy(),
        )

    def add_record(self, record: Record) -> None:
        self.records.append(record)
        self.names.append(record.author_name)
        self.coauthors.update(record.coauthor_keys)
        self.title_terms.update(record.title_terms)
        self.venue_terms.update(record.venue_terms)

    def merge(self, other: "Cluster") -> None:
        self.records.extend(other.records)
        self.names.extend(other.names)
        self.coauthors.update(other.coauthors)
        self.title_terms.update(other.title_terms)
        self.venue_terms.update(other.venue_terms)


def parse_coauthors(raw: str, author_name: str) -> set[str]:
    try:
        names = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid coauthors value {raw!r}") from exc
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError(f"coauthors must be a list of strings, got {raw!r}")
    keys: set[str] = set()
    for name in names:
        # The dataset includes the focal author in its coauthors list.  Excluding it
        # is essential; otherwise every same-name record would share a "coauthor".
        if not names_similar(name, author_name):
            keys.update(coauthor_keys(name))
    return keys


def make_record(index: int, row: dict[str, str]) -> Record:
    return Record(
        index=index,
        ambiguous_name=row["ambiguous_name"],
        author_name=row["author_name"],
        coauthor_keys=parse_coauthors(row["coauthors"], row["author_name"]),
        title_terms=text_terms(row["title"]),
        venue_terms=text_terms(row["venue"]),
    )


def first_step(records: list[Record]) -> list[Cluster]:
    """Build conservative clusters from compatible names and a shared coauthor."""
    clusters: list[Cluster] = []
    coauthor_index: dict[str, set[int]] = defaultdict(set)
    ordered = [r for r in records if r.long_name] + [
        r for r in records if not r.long_name
    ]
    for record in ordered:
        candidates: set[int] = set()
        for key in record.coauthor_keys:
            candidates.update(coauthor_index.get(key, ()))
        chosen = next(
            (
                i
                for i in sorted(candidates)
                if names_similar(record.author_name, clusters[i].names[0])
                and record.coauthor_keys & clusters[i].coauthors
            ),
            None,
        )
        if chosen is None:
            chosen = len(clusters)
            clusters.append(Cluster.from_record(record))
        else:
            clusters[chosen].add_record(record)
        for key in record.coauthor_keys:
            coauthor_index[key].add(chosen)
    return clusters


def second_step(
    clusters: list[Cluster], title_threshold: float, venue_threshold: float
) -> list[Cluster]:
    """Agglomerate clusters until no qualifying title/venue pair remains."""
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                left, right = clusters[i], clusters[j]
                compatible = names_similar(left.names[0], right.names[0])
                merge = compatible and (
                    cosine(left.title_terms, right.title_terms) > title_threshold
                    or cosine(left.venue_terms, right.venue_terms) > venue_threshold
                )
                if merge:
                    left.merge(right)
                    clusters.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1
    return clusters


def cluster_group(
    records: list[Record], title_threshold: float, venue_threshold: float
) -> list[Cluster]:
    return second_step(first_step(records), title_threshold, venue_threshold)


def choose2(n: int) -> int:
    return n * (n - 1) // 2


def k_metrics(
    rows: list[dict[str, str]], predictions: list[str]
) -> tuple[float, float, float]:
    """Return macro-averaged ACP, AAP, and K as evaluated in the paper."""
    group_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        group_indices[row.get("ambiguous_name", "__all__")].append(index)

    acp_values: list[float] = []
    aap_values: list[float] = []
    k_values: list[float] = []
    for indices in group_indices.values():
        truth_sizes = Counter(rows[i]["label"] for i in indices)
        pred_sizes = Counter(predictions[i] for i in indices)
        cells = Counter((rows[i]["label"], predictions[i]) for i in indices)
        count = len(indices)
        acp = sum(n * n / pred_sizes[p] for (truth, p), n in cells.items()) / count
        aap = sum(n * n / truth_sizes[truth] for (truth, p), n in cells.items()) / count
        acp_values.append(acp)
        aap_values.append(aap)
        k_values.append(math.sqrt(acp * aap))

    group_count = len(group_indices)
    return (
        sum(acp_values) / group_count,
        sum(aap_values) / group_count,
        sum(k_values) / group_count,
    )


def evaluate(
    rows: list[dict[str, str]], predictions: list[str]
) -> dict[str, float | int]:
    """Calculate pairwise and B-cubed scores without influencing clustering."""
    truth_sizes = Counter(row["label"] for row in rows)
    pred_sizes = Counter(predictions)
    cells = Counter(
        (row["label"], pred) for row, pred in zip(rows, predictions, strict=True)
    )
    true_positive_pairs = sum(choose2(n) for n in cells.values())
    predicted_pairs = sum(choose2(n) for n in pred_sizes.values())
    true_pairs = sum(choose2(n) for n in truth_sizes.values())
    pair_precision = true_positive_pairs / predicted_pairs if predicted_pairs else 1.0
    pair_recall = true_positive_pairs / true_pairs if true_pairs else 1.0
    pair_f1 = (
        2 * pair_precision * pair_recall / (pair_precision + pair_recall)
        if pair_precision + pair_recall
        else 0.0
    )
    count = len(rows)
    b3_precision = sum(n * n / pred_sizes[p] for (t, p), n in cells.items()) / count
    b3_recall = sum(n * n / truth_sizes[t] for (t, p), n in cells.items()) / count
    b3_f1 = 2 * b3_precision * b3_recall / (b3_precision + b3_recall)
    average_cluster_purity, average_author_purity, k_metric = k_metrics(
        rows, predictions
    )
    return {
        "records": count,
        "true_authors": len(truth_sizes),
        "predicted_clusters": len(pred_sizes),
        "pairwise_precision": pair_precision,
        "pairwise_recall": pair_recall,
        "pairwise_f1": pair_f1,
        "b3_precision": b3_precision,
        "b3_recall": b3_recall,
        "b3_f1": b3_f1,
        "average_cluster_purity": average_cluster_purity,
        "average_author_purity": average_author_purity,
        "k_metric": k_metric,
    }


def read_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    required = {
        "ambiguous_name",
        "paper_id",
        "author_name",
        "coauthors",
        "title",
        "venue",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
        rows = []
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    started_at = time.perf_counter()
    rows = read_rows(args.input, args.limit)
    groups: dict[str, list[Record]] = defaultdict(list)
    for index, row in enumerate(
        tqdm(rows, desc="Preparing records", unit="record", disable=args.no_progress)
    ):
        record = make_record(index, row)
        groups[record.ambiguous_name].append(record)

    predictions = [""] * len(rows)
    for ambiguous_name, records in tqdm(
        groups.items(),
        total=len(groups),
        desc="Clustering names",
        unit="group",
        disable=args.no_progress,
    ):
        clusters = cluster_group(records, args.title_threshold, args.venue_threshold)
        for number, cluster in enumerate(clusters, start=1):
            cluster_id = f"{ambiguous_name}::{number:04d}"
            for record in cluster.records:
                predictions[record.index] = cluster_id

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
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

    summary: dict[str, float | int | str] = {
        "input": str(args.input),
        "output": str(args.output),
        "ambiguous_groups": len(groups),
        "title_threshold": args.title_threshold,
        "venue_threshold": args.venue_threshold,
    }
    if rows and "label" in rows[0]:
        summary.update(evaluate(rows, predictions))
    summary["runtime_seconds"] = round(time.perf_counter() - started_at, 3)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("outputs/hhc_predictions.csv")
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("outputs/hhc_metrics.json"),
        help="metrics JSON path (default: outputs/hhc_metrics.json)",
    )
    parser.add_argument("--title-threshold", type=float, default=0.30)
    parser.add_argument("--venue-threshold", type=float, default=0.50)
    parser.add_argument(
        "--limit", type=int, help="process only the first N records (smoke tests)"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="disable tqdm progress bars"
    )
    args = parser.parse_args(argv)
    for name in ("title_threshold", "venue_threshold"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
