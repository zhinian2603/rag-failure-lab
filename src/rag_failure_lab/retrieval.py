"""Two inspectable lexical retrievers; no embedding API or hidden model call."""

from __future__ import annotations

import math
import re
from collections import Counter

from .models import Document


STOP_WORDS = frozenset(
    "a an and are as at be by can do for from how i in is it of on or our the to what when where which who with you".split()
)


def tokens(text: str) -> list[str]:
    """English demo tokenizer. Multilingual tokenization is an extension point."""
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOP_WORDS]


class Retriever:
    def __init__(self, documents: tuple[Document, ...], mode: str = "bm25") -> None:
        if mode not in {"keyword", "bm25"}:
            raise ValueError("mode must be keyword or bm25")
        self.documents = documents
        self.mode = mode
        self.body_terms = [Counter(tokens(doc.text)) for doc in documents]
        self.title_terms = [Counter(tokens(doc.title)) for doc in documents]
        self.lengths = [sum(terms.values()) for terms in self.body_terms]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.document_frequency = Counter(
            term for terms in self.body_terms for term in terms
        )

    def search(self, question: str, top_k: int = 3) -> list[dict[str, object]]:
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")
        query = set(tokens(question))
        results: list[dict[str, object]] = []
        for index, doc in enumerate(self.documents):
            if self.mode == "keyword":
                score = float(sum(self.body_terms[index].get(term, 0) for term in query))
            else:
                score = self._bm25(index, query)
            if score > 0:
                results.append({"doc_id": doc.id, "title": doc.title, "score": round(score, 4)})
        results.sort(key=lambda item: (-float(item["score"]), str(item["doc_id"])))
        return results[:top_k]

    def _bm25(self, index: int, query: set[str]) -> float:
        k1 = 1.5
        b = 0.75
        score = 0.0
        total_docs = len(self.documents)
        length_ratio = self.lengths[index] / max(1.0, self.average_length)
        for term in query:
            frequency = self.body_terms[index].get(term, 0)
            title_frequency = self.title_terms[index].get(term, 0)
            if not frequency and not title_frequency:
                continue
            # A title match is a transparent, fixed signal; it is not learned.
            frequency += 2 * title_frequency
            df = self.document_frequency.get(term, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            score += idf * frequency * (k1 + 1) / (
                frequency + k1 * (1 - b + b * length_ratio)
            )
        return score
