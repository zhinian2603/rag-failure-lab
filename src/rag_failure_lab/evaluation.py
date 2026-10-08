"""Separate retrieval, citation, answer completeness, and lexical support."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .models import Case, Dataset
from .retrieval import Retriever, tokens


GROUNDING_THRESHOLD = 0.25


def _coverage(case: Case) -> float:
    answer = case.answer.casefold()
    return sum(term.casefold() in answer for term in case.expected_terms) / len(case.expected_terms)


def _grounding(case: Case, documents_by_id: dict[str, str]) -> float:
    answer_words = set(tokens(case.answer))
    if not answer_words or not case.citations:
        return 0.0
    evidence_words: set[str] = set()
    for citation in case.citations:
        evidence_words.update(tokens(documents_by_id.get(citation, "")))
    return len(answer_words & evidence_words) / len(answer_words)


def evaluate(dataset: Dataset, *, mode: str = "bm25", top_k: int = 3) -> dict[str, Any]:
    if mode not in {"keyword", "bm25", "recorded"}:
        raise ValueError("mode must be keyword, bm25, or recorded")
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20")
    retriever = Retriever(dataset.documents, mode) if mode != "recorded" else None
    document_text = {doc.id: f"{doc.title} {doc.text}" for doc in dataset.documents}
    document_titles = {doc.id: doc.title for doc in dataset.documents}
    rows: list[dict[str, Any]] = []

    for case in dataset.cases:
        if mode == "recorded":
            if case.retrieved_doc_ids is None:
                raise ValueError(f"case {case.id} has no retrieved_doc_ids for recorded mode")
            hits = [
                {"doc_id": doc_id, "title": document_titles.get(doc_id, "Unknown source"), "score": None}
                for doc_id in case.retrieved_doc_ids[:top_k]
            ]
        else:
            assert retriever is not None
            hits = retriever.search(case.question, top_k)
        retrieved = [str(hit["doc_id"]) for hit in hits]
        required = set(case.required_doc_ids)
        recall = len(required.intersection(retrieved)) / len(required)
        first_relevant_rank = next(
            (rank for rank, doc_id in enumerate(retrieved, 1) if doc_id in required), None
        )
        reciprocal_rank = 1 / first_relevant_rank if first_relevant_rank else 0.0
        citations = set(case.citations)
        citations_valid = (
            bool(citations)
            and citations.issubset(retrieved)
            and citations.issubset(document_text)
            and required.issubset(citations)
        )
        term_coverage = _coverage(case)
        grounding = _grounding(case, document_text)

        if recall < 1.0:
            diagnosis = "retrieval_miss"
        elif not citations_valid:
            diagnosis = "citation_error"
        elif term_coverage < 1.0:
            diagnosis = "answer_gap"
        elif grounding < GROUNDING_THRESHOLD:
            diagnosis = "weak_evidence"
        else:
            diagnosis = "pass"

        rows.append(
            {
                "id": case.id,
                "question": case.question,
                "answer": case.answer,
                "expected_answer": case.expected_answer,
                "required_doc_ids": list(case.required_doc_ids),
                "citations": list(case.citations),
                "retrieved": hits,
                "recall_at_k": round(recall, 4),
                "reciprocal_rank": round(reciprocal_rank, 4),
                "citations_valid": citations_valid,
                "expected_term_coverage": round(term_coverage, 4),
                "lexical_support": round(grounding, 4),
                "diagnosis": diagnosis,
            }
        )

    count = len(rows)
    counts = Counter(row["diagnosis"] for row in rows)
    config = {"mode": mode, "top_k": top_k, "grounding_threshold": GROUNDING_THRESHOLD}
    fingerprint = hashlib.sha256(
        json.dumps({"dataset": dataset.to_dict(), "config": config}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "id": fingerprint,
        "dataset": dataset.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "summary": {
            "case_count": count,
            "retrieval_recall_at_k": round(sum(row["recall_at_k"] for row in rows) / count, 4),
            "retrieval_mrr": round(sum(row["reciprocal_rank"] for row in rows) / count, 4),
            "valid_citation_rate": round(sum(row["citations_valid"] for row in rows) / count, 4),
            "answer_completeness": round(sum(row["expected_term_coverage"] for row in rows) / count, 4),
            "average_lexical_support": round(sum(row["lexical_support"] for row in rows) / count, 4),
            "passed_rate": round(counts["pass"] / count, 4),
            "diagnoses": dict(counts),
        },
        "cases": rows,
    }


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare the same case IDs; refuse a misleading cross-dataset comparison."""
    if baseline["dataset"] != candidate["dataset"]:
        raise ValueError("runs must use the same dataset")
    before = {row["id"]: row for row in baseline["cases"]}
    after = {row["id"]: row for row in candidate["cases"]}
    if before.keys() != after.keys():
        raise ValueError("runs must contain the same case IDs")
    if any(before[case_id]["question"] != after[case_id]["question"] for case_id in before):
        raise ValueError("runs must contain the same questions")
    rows: list[dict[str, Any]] = []
    for case_id in sorted(before):
        old, new = before[case_id], after[case_id]
        delta = round(new["recall_at_k"] - old["recall_at_k"], 4)
        if delta > 0:
            movement = "improved"
        elif delta < 0:
            movement = "regressed"
        elif old["diagnosis"] != "pass" and new["diagnosis"] == "pass":
            movement = "improved"
        elif old["diagnosis"] == "pass" and new["diagnosis"] != "pass":
            movement = "regressed"
        else:
            movement = "unchanged"
        rows.append(
            {
                "id": case_id,
                "question": new["question"],
                "baseline_diagnosis": old["diagnosis"],
                "candidate_diagnosis": new["diagnosis"],
                "recall_delta": delta,
                "movement": movement,
            }
        )
    counts = Counter(row["movement"] for row in rows)
    return {
        "baseline_id": baseline["id"],
        "candidate_id": candidate["id"],
        "baseline_mode": baseline["config"]["mode"],
        "candidate_mode": candidate["config"]["mode"],
        "recall_delta": round(
            candidate["summary"]["retrieval_recall_at_k"] - baseline["summary"]["retrieval_recall_at_k"],
            4,
        ),
        "counts": dict(counts),
        "cases": rows,
    }
