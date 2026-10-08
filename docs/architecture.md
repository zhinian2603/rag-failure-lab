# Design notes

## Why the primary input is a labeled case

A believable quality claim needs a frozen evaluation set. Each case declares the question, recorded answer, expected answer, required source IDs, and expected terms. This keeps the benchmark from changing silently as prompt or retrieval logic changes. The bundled examples are synthetic and deliberately include defects.

## Why recorded traces matter

Replacing a customer's retriever with the demo BM25 would hide the actual production failure. `recorded` mode accepts the ranked document IDs captured by the customer's pipeline. The same evaluator can then separate a source miss from a citation mismatch or an incomplete answer. This is the intended first integration point in a paid diagnostic engagement.

The bundled replay is reproducible without customer access: `capture-traces` runs our own BM25 or keyword retriever over author-created documents and writes its actual ranked IDs to a JSON file. A zero-hit query produces an empty ranking and is correctly classified as a retrieval miss. Answers and citations remain authored test fixtures; no model output or employer data is implied.

## Failure ordering

The primary diagnosis is selected in this order: missing gold source, bad citation, missing answer term, weak lexical evidence, pass. This avoids calling a generated answer the primary problem when the necessary source was never retrieved. The report still shows all raw signals so an engineer can challenge the chosen label.

## Reliability decisions

- Dataset IDs are validated before evaluation. Gold source IDs must exist; bad cited IDs remain in the case so they can be diagnosed.
- `top_k` is bounded to 1–20; input lists are bounded to 1000 items each; HTTP request bodies are capped at 2 MiB.
- Retrieval sorting has a stable ID tie-break, enabling repeatable comparisons.
- Run IDs hash the dataset plus evaluation settings. SQLite uses a primary key and `INSERT OR IGNORE`, making repeated submissions idempotent.
- Connections are closed after every operation. The local server handles concurrent requests without sharing a connection across threads.
- HTML escapes every case field. The report loads no third-party scripts or images.
- The server binds to `127.0.0.1` by default and sends `Cache-Control: no-store`.

## Known limits

- The tokenizer is English-oriented; Chinese and other languages need their own segmentation strategy.
- Expected-term matching can miss paraphrases and may be satisfied by a negated phrase. It is a cheap regression signal, not a semantic score.
- Lexical support can pass a wrong answer that reuses source vocabulary or flag a correct paraphrase. It must be reviewed with human labels or a calibrated judge.
- The local API is a demo. There is no authentication, tenant isolation, queue, retention policy or observability stack for hosted use.
- The sample comparison is intentionally small. It demonstrates the workflow, not a universal BM25 advantage.
