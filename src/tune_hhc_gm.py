"""Sweep GM settings on a user-supplied validation set, reusing model decisions.

Example: python -m src.tune_hhc_gm validation.csv --model Qwen/Qwen3-1.7B
Never pass the final test set as validation input.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

try:
    from src import hhc_gm
except ImportError:
    import hhc_gm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gm_validation"))
    parser.add_argument("--confidence-grid", type=float, nargs="+", default=[0.7, 0.8, 0.9])
    parser.add_argument("--semantic-grid", type=float, nargs="+", default=[0.4, 0.55])
    parser.add_argument("--top-k-grid", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--minimum-precision", type=float, default=0.98)
    options, extra = parser.parse_known_args()
    if not 0 <= options.minimum_precision <= 1:
        parser.error("--minimum-precision must be between 0 and 1")
    options.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (confidence, semantic, top_k) in enumerate(itertools.product(
        options.confidence_grid, options.semantic_grid, options.top_k_grid
    )):
        prefix = options.output_dir / f"trial_{i:03d}"
        args = hhc_gm.parse_args([str(options.validation_input), *extra,
            "--llm-confidence-threshold", str(confidence),
            "--semantic-candidate-threshold", str(semantic), "--semantic-top-k", str(top_k),
            "--output", str(prefix.with_suffix(".csv")),
            "--metrics-output", str(prefix.with_suffix(".json")),
            "--prompts-output", str(prefix) + "_prompts.json",
            "--decisions-output", str(prefix) + "_decisions.jsonl", "--retrieval-diagnostics"])
        result = hhc_gm.run(args)
        if "pairwise_f1" not in result:
            raise ValueError("Validation input requires ground-truth labels")
        results.append(result)
        (options.output_dir / "sweep.json").write_text(json.dumps(results, indent=2) + "\n")
    eligible = [r for r in results if r["pairwise_precision"] >= options.minimum_precision]
    best = max(eligible, key=lambda r: (r["pairwise_f1"], r["b3_f1"])) if eligible else None
    selection = {"minimum_precision": options.minimum_precision,
                 "best_validation_run": best,
                 "note": "Freeze these settings before evaluating a separate test set."}
    (options.output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
