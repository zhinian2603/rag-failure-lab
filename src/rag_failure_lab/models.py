"""Validated, deliberately small data contract for a RAG evaluation run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    answer: str
    expected_answer: str
    required_doc_ids: tuple[str, ...]
    citations: tuple[str, ...]
    expected_terms: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Dataset:
    name: str
    documents: tuple[Document, ...]
    cases: tuple[Case, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _strings(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(f"{label} must be a list of strings")
    return tuple(_string(item, f"{label} item") for item in value)


def parse_dataset(data: Any) -> Dataset:
    """Reject malformed fixtures before metrics can give a misleading result.

    Citations are allowed to reference unknown IDs: detecting that failure is
    one of the diagnostic tasks. Gold document IDs must exist in the corpus.
    """
    if not isinstance(data, dict):
        raise ValueError("dataset must be a JSON object")
    name = _string(data.get("name"), "name")
    raw_docs = data.get("documents")
    raw_cases = data.get("cases")
    if not isinstance(raw_docs, list) or not raw_docs:
        raise ValueError("documents must be a non-empty list")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases must be a non-empty list")
    if len(raw_docs) > 1000 or len(raw_cases) > 1000:
        raise ValueError("demo limit: at most 1000 documents and 1000 cases")

    documents: list[Document] = []
    doc_ids: set[str] = set()
    for index, item in enumerate(raw_docs):
        if not isinstance(item, dict):
            raise ValueError(f"documents[{index}] must be an object")
        doc = Document(
            id=_string(item.get("id"), f"documents[{index}].id"),
            title=_string(item.get("title"), f"documents[{index}].title"),
            text=_string(item.get("text"), f"documents[{index}].text"),
        )
        if doc.id in doc_ids:
            raise ValueError(f"duplicate document ID: {doc.id}")
        doc_ids.add(doc.id)
        documents.append(doc)

    cases: list[Case] = []
    case_ids: set[str] = set()
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise ValueError(f"cases[{index}] must be an object")
        case = Case(
            id=_string(item.get("id"), f"cases[{index}].id"),
            question=_string(item.get("question"), f"cases[{index}].question"),
            answer=_string(item.get("answer"), f"cases[{index}].answer"),
            expected_answer=_string(item.get("expected_answer"), f"cases[{index}].expected_answer"),
            required_doc_ids=_strings(item.get("required_doc_ids"), f"cases[{index}].required_doc_ids"),
            citations=_strings(item.get("citations", []), f"cases[{index}].citations", allow_empty=True),
            expected_terms=_strings(item.get("expected_terms"), f"cases[{index}].expected_terms"),
            retrieved_doc_ids=(
                _strings(item["retrieved_doc_ids"], f"cases[{index}].retrieved_doc_ids", allow_empty=True)
                if "retrieved_doc_ids" in item else None
            ),
        )
        if case.id in case_ids:
            raise ValueError(f"duplicate case ID: {case.id}")
        missing = set(case.required_doc_ids) - doc_ids
        if missing:
            raise ValueError(f"case {case.id} has unknown gold document IDs: {sorted(missing)}")
        case_ids.add(case.id)
        cases.append(case)

    return Dataset(name=name, documents=tuple(documents), cases=tuple(cases))
