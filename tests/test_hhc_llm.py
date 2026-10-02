from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import hhc, hhc_gm  # noqa: E402


def make_row(index: int, name: str, title: str) -> dict[str, str]:
    return {
        "ambiguous_name": "j_smith",
        "paper_id": f"paper-{index}",
        "author_name": name,
        "coauthors": repr([name]),
        "title": title,
        "venue": "",
        "label": "unused",
    }


class FakeJudge:
    def __init__(self, decision: hhc_gm.LLMDecision) -> None:
        self.decision = decision
        self.calls = 0

    def cache_key(self, left, right, rows) -> str:
        indices = sorted(
            record.index
            for cluster in (left, right)
            for record in cluster.records
        )
        return json.dumps(indices)

    def compare(self, left, right, rows):
        self.calls += 1
        return self.decision, False


class CharacterTokenizer:
    def __call__(self, text):
        return {"input_ids": list(text)}


class DecisionParsingTests(unittest.TestCase):
    def test_parses_json_inside_markdown(self) -> None:
        decision = hhc_gm.parse_decision(
            '```json\n{"same_author": true, "confidence": 0.94, "reason": "match"}\n```'
        )
        self.assertTrue(decision.same_author)
        self.assertEqual(decision.confidence, 0.94)

    def test_rejects_string_boolean(self) -> None:
        with self.assertRaises(ValueError):
            hhc_gm.parse_decision(
                '{"same_author": "true", "confidence": 0.94, "reason": "match"}'
            )


class LLMStageTests(unittest.TestCase):
    def test_oversized_prompt_automatically_reduces_records(self) -> None:
        rows = [
            make_row(0, "J Smith", "A" * 100),
            make_row(1, "J Smith", "B" * 100),
            make_row(2, "John Smith", "C" * 100),
            make_row(3, "John Smith", "D" * 100),
        ]
        left = hhc.Cluster.from_record(hhc.make_record(0, rows[0]))
        left.add_record(hhc.make_record(1, rows[1]))
        right = hhc.Cluster.from_record(hhc.make_record(2, rows[2]))
        right.add_record(hhc.make_record(3, rows[3]))
        one_record_prompt = hhc_gm.build_prompt(left, right, rows, max_records=1)

        judge = object.__new__(hhc_gm.LocalTransformersJudge)
        judge.max_records = 2
        judge.max_input_tokens = len(one_record_prompt)
        judge.retries = 0
        judge.prompt_reductions = 0
        judge.prompt_cache = {}
        judge.tokenizer = CharacterTokenizer()
        judge._render = lambda prompt: prompt

        prompt = judge._prompt(left, right, rows)

        self.assertEqual(prompt, one_record_prompt)
        self.assertEqual(judge.prompt_reductions, 1)

    def test_prompt_summary_excludes_ground_truth_label(self) -> None:
        rows = [make_row(0, "John Smith", "Graph inference")]
        cluster = hhc.Cluster.from_record(hhc.make_record(0, rows[0]))

        summary = hhc_gm.cluster_summary(cluster, rows, max_records=8)

        self.assertNotIn("label", json.dumps(summary))

    def test_merges_compatible_names_on_confident_decision(self) -> None:
        rows = [
            make_row(0, "J Smith", "Graph inference"),
            make_row(1, "John Smith", "Marine ecology"),
        ]
        clusters = [
            hhc.Cluster.from_record(hhc.make_record(i, row))
            for i, row in enumerate(rows)
        ]
        judge = FakeJudge(hhc_gm.LLMDecision(True, 0.95, "supporting evidence"))

        result, stats = hhc_gm.llm_second_step(
            clusters, rows, judge, 0.9, 0.0, 25
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(stats.merges, 1)
        self.assertEqual(stats.model_calls, 1)

    def test_does_not_merge_below_confidence_threshold(self) -> None:
        rows = [
            make_row(0, "J Smith", "Graph inference"),
            make_row(1, "John Smith", "Marine ecology"),
        ]
        clusters = [
            hhc.Cluster.from_record(hhc.make_record(i, row))
            for i, row in enumerate(rows)
        ]
        judge = FakeJudge(hhc_gm.LLMDecision(True, 0.70, "uncertain"))

        result, stats = hhc_gm.llm_second_step(
            clusters, rows, judge, 0.9, 0.0, 25
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(stats.merges, 0)

    def test_never_asks_about_incompatible_names(self) -> None:
        rows = [
            make_row(0, "John Smith", "Graph inference"),
            make_row(1, "Jane Smith", "Graph inference"),
        ]
        clusters = [
            hhc.Cluster.from_record(hhc.make_record(i, row))
            for i, row in enumerate(rows)
        ]
        judge = FakeJudge(hhc_gm.LLMDecision(True, 1.0, "incorrect"))

        result, stats = hhc_gm.llm_second_step(
            clusters, rows, judge, 0.9, 0.0, 25
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(stats.comparisons, 0)
        self.assertEqual(judge.calls, 0)


if __name__ == "__main__":
    unittest.main()
