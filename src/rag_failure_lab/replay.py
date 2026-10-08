"""Capture real rankings from the bundled retrievers as portable trace input."""

from __future__ import annotations

from dataclasses import replace

from .models import Dataset
from .retrieval import Retriever


def capture_retrieval_traces(dataset: Dataset, *, mode: str = "bm25", top_k: int = 3) -> Dataset:
    """Run retrieval once and store its ranked IDs with each labeled case.

    Answers and citations remain the authored fixture values. This command
    captures actual retriever output; it does not synthesize model responses.
    """
    retriever = Retriever(dataset.documents, mode)
    cases = tuple(
        replace(
            case,
            retrieved_doc_ids=tuple(str(hit["doc_id"]) for hit in retriever.search(case.question, top_k)),
        )
        for case in dataset.cases
    )
    return replace(dataset, cases=cases)
