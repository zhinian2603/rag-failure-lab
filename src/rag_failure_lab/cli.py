"""Command-line entry point for repeatable runs and a local demo server."""

from __future__ import annotations

import argparse
import json
import logging
from importlib.resources import files
from pathlib import Path

from .api import serve
from .evaluation import compare, evaluate
from .models import Dataset, parse_dataset
from .report import render_report
from .replay import capture_retrieval_traces
from .storage import RunStore


def demo_dataset_path() -> Path:
    return Path(str(files("rag_failure_lab").joinpath("data/northstar-support.json")))


def load_dataset(path: Path) -> Dataset:
    return parse_dataset(json.loads(path.read_text(encoding="utf-8")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-failure-lab", description="Reproducible RAG failure triage")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate_cmd = sub.add_parser("evaluate", help="evaluate a JSON dataset and write a report")
    evaluate_cmd.add_argument("--dataset", type=Path, default=demo_dataset_path())
    evaluate_cmd.add_argument("--mode", choices=("keyword", "bm25", "recorded"), default="bm25")
    evaluate_cmd.add_argument("--top-k", type=int, default=1)
    evaluate_cmd.add_argument("--compare", choices=("keyword", "bm25", "recorded"), help="run a baseline mode for comparison")
    evaluate_cmd.add_argument("--html", type=Path, default=Path("demo-report.html"))
    evaluate_cmd.add_argument("--json", type=Path, help="optional machine-readable output")
    replay_cmd = sub.add_parser("capture-traces", help="run a retriever and save its ranked document IDs")
    replay_cmd.add_argument("--dataset", type=Path, default=demo_dataset_path())
    replay_cmd.add_argument("--mode", choices=("keyword", "bm25"), default="bm25")
    replay_cmd.add_argument("--top-k", type=int, default=3)
    replay_cmd.add_argument("--output", type=Path, default=Path("docs/replayed-traces.json"))
    serve_cmd = sub.add_parser("serve", help="serve a local report and JSON API")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8080)
    serve_cmd.add_argument("--db", type=Path, default=Path(".state/runs.db"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        dataset = load_dataset(args.dataset)
        run = evaluate(dataset, mode=args.mode, top_k=args.top_k)
        baseline = evaluate(dataset, mode=args.compare, top_k=args.top_k) if args.compare else None
        comparison = compare(baseline, run) if baseline else None
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(render_report(run, comparison), encoding="utf-8")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps({"run": run, "comparison": comparison}, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(args.html), "run_id": run["id"], "summary": run["summary"]}, indent=2))
        return 0
    if args.command == "capture-traces":
        dataset = capture_retrieval_traces(load_dataset(args.dataset), mode=args.mode, top_k=args.top_k)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dataset.to_dict(), indent=2), encoding="utf-8")
        print(json.dumps({"trace_file": str(args.output), "case_count": len(dataset.cases), "mode": args.mode}))
        return 0
    if args.command == "serve":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        args.db.parent.mkdir(parents=True, exist_ok=True)
        store = RunStore(args.db)
        dataset = load_dataset(demo_dataset_path())
        baseline = evaluate(dataset, mode="keyword", top_k=1)
        candidate = evaluate(dataset, mode="bm25", top_k=1)
        store.save(baseline)
        store.save(candidate)
        print(f"RAG Failure Lab: http://{args.host}:{args.port}", flush=True)
        serve(args.host, args.port, store, dataset, candidate["id"], compare(baseline, candidate))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
