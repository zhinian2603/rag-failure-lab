"""Single-file, shareable HTML report with no remote scripts or assets."""

from __future__ import annotations

from html import escape
from typing import Any


LABELS = {
    "pass": "Passed",
    "retrieval_miss": "Retrieval miss",
    "citation_error": "Citation error",
    "answer_gap": "Answer gap",
    "weak_evidence": "Weak evidence",
}


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _card(label: str, value: str, detail: str, accent: str = "") -> str:
    return (
        f'<div class="metric {accent}"><span>{escape(label)}</span>'
        f'<strong>{escape(value)}</strong><small>{escape(detail)}</small></div>'
    )


def render_report(run: dict[str, Any], comparison: dict[str, Any] | None = None) -> str:
    summary = run["summary"]
    diagnoses = summary["diagnoses"]
    cards = "".join(
        [
            _card("Retrieval recall", _pct(summary["retrieval_recall_at_k"]), f"gold sources in top {run['config']['top_k']}", "teal"),
            _card("Valid citations", _pct(summary["valid_citation_rate"]), "source IDs and retrieval checked"),
            _card("Answer completeness", _pct(summary["answer_completeness"]), "expected terms present"),
            _card("Cases passed", _pct(summary["passed_rate"]), f"{summary['case_count']} synthetic cases", "violet"),
        ]
    )
    diagnosis_cards = "".join(
        f'<div class="diagnosis"><span class="dot {escape(key)}"></span>'
        f'<span>{escape(LABELS[key])}</span><strong>{diagnoses.get(key, 0)}</strong></div>'
        for key in ("pass", "retrieval_miss", "citation_error", "answer_gap", "weak_evidence")
    )
    rows: list[str] = []
    for case in run["cases"]:
        diagnosis = str(case["diagnosis"])
        retrieved = ", ".join(str(hit["doc_id"]) for hit in case["retrieved"]) or "none"
        rows.append(
            f'<details class="case"><summary><span class="case-id">{escape(case["id"])}</span>'
            f'<span class="question">{escape(case["question"])}</span>'
            f'<span class="pill {escape(diagnosis)}">{escape(LABELS[diagnosis])}</span>'
            f'<span class="chevron">⌄</span></summary><div class="case-detail">'
            f'<div><b>Recorded answer</b><p>{escape(case["answer"])}</p></div>'
            f'<div><b>Expected answer</b><p>{escape(case["expected_answer"])}</p></div>'
            f'<div><b>Retrieved documents</b><p>{escape(retrieved)}</p></div>'
            f'<div><b>Gold documents</b><p>{escape(", ".join(case["required_doc_ids"]))}</p></div>'
            f'<div><b>Recall@k</b><p>{_pct(case["recall_at_k"])}</p></div>'
            f'<div><b>Lexical support</b><p>{_pct(case["lexical_support"])}</p></div>'
            "</div></details>"
        )

    compare_html = ""
    if comparison:
        delta = comparison["recall_delta"]
        sign = "+" if delta > 0 else ""
        moves = comparison["counts"]
        compare_rows = "".join(
            f'<tr><td>{escape(row["id"])}</td><td>{escape(row["question"])}</td>'
            f'<td><span class="pill {escape(row["baseline_diagnosis"])}">{escape(LABELS[row["baseline_diagnosis"]])}</span></td>'
            f'<td><span class="pill {escape(row["candidate_diagnosis"])}">{escape(LABELS[row["candidate_diagnosis"]])}</span></td>'
            f'<td class="movement {escape(row["movement"])}">{escape(row["movement"].title())}</td></tr>'
            for row in comparison["cases"]
        )
        compare_html = (
            '<section class="panel compare"><div class="section-head"><div><div class="eyebrow">REGRESSION VIEW</div>'
            '<h2>Baseline → candidate</h2></div>'
            f'<span class="delta">{sign}{_pct(delta)} recall</span></div>'
            f'<p class="muted">{escape(comparison["baseline_mode"])} → {escape(comparison["candidate_mode"])}'
            f' · {moves.get("improved", 0)} improved · {moves.get("regressed", 0)} regressed · {moves.get("unchanged", 0)} unchanged</p>'
            '<div class="table-wrap"><table><thead><tr><th>ID</th><th>Question</th><th>Baseline</th><th>Candidate</th><th>Change</th></tr></thead>'
            f'<tbody>{compare_rows}</tbody></table></div></section>'
        )

    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG Failure Lab — {escape(run["dataset"])}</title>
<style>
:root{{--ink:#162338;--muted:#627185;--line:#dfe7ec;--bg:#f5f8fa;--teal:#0b8e87;--navy:#142943;--violet:#6958b7;--amber:#c98918;--red:#b94d4b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}}
.topbar{{background:#10243b;color:#fff;padding:17px max(28px,calc((100vw - 1180px)/2));display:flex;align-items:center;justify-content:space-between}}
.brand{{font-weight:800;letter-spacing:-.03em;font-size:19px}}.brand i{{font-style:normal;color:#36d0bf}}.topbar small{{color:#b4c7d5}}
main{{max-width:1180px;margin:auto;padding:48px 28px 80px}}.eyebrow{{font-size:11px;font-weight:800;color:var(--teal);letter-spacing:.16em}}
h1{{font-size:clamp(34px,4vw,55px);line-height:1.08;letter-spacing:-.05em;margin:12px 0 14px;max-width:840px}}
.lead{{color:var(--muted);font-size:18px;max-width:740px;margin:0 0 28px}}.meta{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:38px}}
.meta span{{border:1px solid var(--line);background:#fff;border-radius:999px;padding:7px 13px;color:#506479;font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}}.metric{{background:#fff;border:1px solid var(--line);border-radius:16px;padding:21px;min-height:148px}}
.metric span,.metric small{{display:block;color:var(--muted)}}.metric span{{font-size:12px;font-weight:700}}.metric strong{{display:block;font-size:35px;letter-spacing:-.055em;margin:10px 0 2px}}
.metric.teal strong{{color:var(--teal)}}.metric.violet strong{{color:var(--violet)}}.metric small{{font-size:11px}}
.two{{display:grid;grid-template-columns:1.1fr .9fr;gap:16px;margin-bottom:24px}}.panel{{background:#fff;border:1px solid var(--line);border-radius:18px;padding:27px}}
.panel h2{{font-size:22px;letter-spacing:-.03em;margin:5px 0 17px}}.diagnosis{{display:flex;align-items:center;gap:10px;border-top:1px solid #edf1f4;padding:11px 0}}
.diagnosis strong{{margin-left:auto;font-size:19px}}.dot{{width:9px;height:9px;border-radius:50%;background:var(--amber)}}.dot.pass{{background:var(--teal)}}.dot.citation_error{{background:var(--red)}}
.muted{{color:var(--muted)}}.note{{background:#f1f7f7;border-left:3px solid var(--teal);padding:13px 17px;border-radius:0 8px 8px 0;font-size:13px;color:#3c5a62}}
.section-head{{display:flex;align-items:center;justify-content:space-between;gap:15px}}.delta{{background:#e5f6f1;color:#08786f;padding:8px 12px;border-radius:999px;font-weight:800;white-space:nowrap}}
.compare{{margin-bottom:24px}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th{{text-align:left;color:var(--muted);font-size:11px;letter-spacing:.08em;text-transform:uppercase}}
th,td{{padding:11px 12px;border-bottom:1px solid #edf1f4}}td:nth-child(2){{min-width:210px}}.movement.improved{{color:var(--teal);font-weight:800}}.movement.regressed{{color:var(--red);font-weight:800}}
.case{{border-top:1px solid #edf1f4}}.case summary{{list-style:none;cursor:pointer;display:flex;align-items:center;gap:13px;padding:15px 0}}.case summary::-webkit-details-marker{{display:none}}
.case-id{{color:var(--muted);font:700 12px ui-monospace,SFMono-Regular,Consolas,monospace}}.question{{flex:1;font-weight:600}}.chevron{{color:var(--muted)}}
.pill{{display:inline-block;border-radius:999px;padding:3px 9px;background:#fff1dc;color:#916311;font-size:11px;font-weight:700;white-space:nowrap}}.pill.pass{{background:#e4f5f0;color:#08786f}}.pill.citation_error{{background:#fde9e7;color:#a63d3a}}.pill.answer_gap{{background:#f1eaff;color:#7051a9}}.pill.weak_evidence{{background:#eaf0fa;color:#4d6799}}
.case-detail{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;padding:4px 0 19px 42px;font-size:13px}}.case-detail b{{color:#425970}}.case-detail p{{margin:3px 0 0;color:var(--muted)}}
footer{{border-top:1px solid var(--line);margin-top:30px;padding-top:15px;color:var(--muted);font-size:12px}}
@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,1fr)}}.two{{grid-template-columns:1fr}}}}@media(max-width:530px){{main{{padding:32px 16px}}.grid{{grid-template-columns:1fr 1fr}}.metric{{padding:14px;min-height:125px}}.metric strong{{font-size:27px}}.case-detail{{grid-template-columns:1fr;padding-left:0}}.topbar{{padding:15px}}
.case summary{{flex-wrap:wrap}}.question{{flex-basis:75%}}}}
</style></head><body>
<header class="topbar"><div class="brand">RAG <i>Failure</i> Lab</div><small>Transparent diagnostics · reproducible runs</small></header>
<main><div class="eyebrow">EVALUATION REPORT / SYNTHETIC DEMO</div><h1>Find where the answer pipeline breaks.</h1>
<p class="lead">A compact view of retrieval misses, citation errors, incomplete answers and evidence signals — with every case open for inspection.</p>
<div class="meta"><span>Dataset: {escape(run["dataset"])}</span><span>Retriever: {escape(run["config"]["mode"].upper())}</span><span>Top K: {run["config"]["top_k"]}</span><span>Run ID: {escape(run["id"])}</span></div>
<div class="grid">{cards}</div><div class="two"><section class="panel"><div class="eyebrow">FAILURE MIX</div><h2>What needs attention</h2>{diagnosis_cards}</section>
<section class="panel"><div class="eyebrow">INTERPRETATION</div><h2>Measure each stage separately</h2><p class="muted">Gold document recall checks retrieval. Citation validation checks whether cited sources were retrieved and include the required evidence. Expected terms flag missing answer elements.</p>
<div class="note"><b>Scope note.</b> Lexical support is a transparent token-overlap heuristic. It cannot prove factual grounding or semantic correctness; use human review or a calibrated judge for high-stakes content.</div></section></div>
{compare_html}<section class="panel"><div class="section-head"><div><div class="eyebrow">CASE EXPLORER</div><h2>Inspect every outcome</h2></div><span class="muted">{summary["case_count"]} cases</span></div>{''.join(rows)}</section>
<footer>Built from synthetic documentation and recorded answers. No customer data, credentials or paid model calls are used.</footer></main></body></html>'''
