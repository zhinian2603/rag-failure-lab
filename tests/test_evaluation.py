import json
import unittest
from pathlib import Path

from rag_failure_lab.evaluation import compare, evaluate
from rag_failure_lab.models import parse_dataset
from rag_failure_lab.report import render_report
from rag_failure_lab.retrieval import Retriever
from rag_failure_lab.replay import capture_retrieval_traces


DATA = Path(__file__).parents[1] / "src" / "rag_failure_lab" / "data" / "northstar-support.json"


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = parse_dataset(json.loads(DATA.read_text(encoding="utf-8")))

    def test_bm25_recovers_a_keyword_retrieval_miss(self):
        baseline = evaluate(self.dataset, mode="keyword", top_k=1)
        candidate = evaluate(self.dataset, mode="bm25", top_k=1)
        before = {row["id"]: row for row in baseline["cases"]}
        after = {row["id"]: row for row in candidate["cases"]}
        self.assertEqual(before["C09"]["diagnosis"], "retrieval_miss")
        self.assertEqual(after["C09"]["diagnosis"], "pass")
        self.assertEqual(compare(baseline, candidate)["counts"], {"unchanged": 11, "improved": 1})

    def test_failure_stages_are_separate(self):
        run = evaluate(self.dataset, mode="bm25", top_k=1)
        by_id = {row["id"]: row for row in run["cases"]}
        self.assertEqual(by_id["C04"]["diagnosis"], "answer_gap")
        self.assertEqual(by_id["C12"]["diagnosis"], "citation_error")
        self.assertEqual(run["summary"]["retrieval_recall_at_k"], 1.0)

    def test_rankings_are_deterministic(self):
        search = Retriever(self.dataset.documents, "bm25")
        first = search.search("Where is EU customer data hosted?", 1)
        second = search.search("Where is EU customer data hosted?", 1)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["doc_id"], "regions")

    def test_rejects_broken_gold_ids(self):
        payload = json.loads(DATA.read_text(encoding="utf-8"))
        payload["cases"][0]["required_doc_ids"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "unknown gold document"):
            parse_dataset(payload)

    def test_report_escapes_user_supplied_text(self):
        payload = json.loads(DATA.read_text(encoding="utf-8"))
        payload["cases"][0]["question"] = "<script>alert(1)</script>"
        run = evaluate(parse_dataset(payload), top_k=1)
        html = render_report(run)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>alert(1)</script>", html)

    def test_recorded_trace_mode_separates_failure_stages(self):
        path = DATA.with_name("recorded-traces.json")
        dataset = parse_dataset(json.loads(path.read_text(encoding="utf-8")))
        run = evaluate(dataset, mode="recorded", top_k=2)
        by_id = {row["id"]: row["diagnosis"] for row in run["cases"]}
        self.assertEqual(by_id, {
            "T01": "pass",
            "T02": "retrieval_miss",
            "T03": "citation_error",
            "T04": "answer_gap",
        })

    def test_captured_rankings_replay_to_same_diagnoses(self):
        direct = evaluate(self.dataset, mode="bm25", top_k=1)
        captured = capture_retrieval_traces(self.dataset, mode="bm25", top_k=1)
        replayed = evaluate(captured, mode="recorded", top_k=1)
        self.assertEqual(
            [row["diagnosis"] for row in direct["cases"]],
            [row["diagnosis"] for row in replayed["cases"]],
        )
        self.assertEqual(direct["summary"]["retrieval_recall_at_k"], replayed["summary"]["retrieval_recall_at_k"])

    def test_empty_captured_ranking_is_a_retrieval_miss(self):
        payload = json.loads(DATA.read_text(encoding="utf-8"))
        payload["cases"][0]["question"] = "unmatched-zzzxxyy"
        dataset = parse_dataset(payload)
        captured = capture_retrieval_traces(dataset, mode="bm25", top_k=1)
        self.assertEqual(captured.cases[0].retrieved_doc_ids, ())
        replayed = evaluate(captured, mode="recorded", top_k=1)
        self.assertEqual(replayed["cases"][0]["diagnosis"], "retrieval_miss")


if __name__ == "__main__":
    unittest.main()
