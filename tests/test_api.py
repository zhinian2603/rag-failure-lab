import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from rag_failure_lab.api import make_handler
from rag_failure_lab.evaluation import compare, evaluate
from rag_failure_lab.models import parse_dataset
from rag_failure_lab.storage import RunStore


DATA = Path(__file__).parents[1] / "src" / "rag_failure_lab" / "data" / "northstar-support.json"


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dataset_dict = json.loads(DATA.read_text(encoding="utf-8"))
        dataset = parse_dataset(self.dataset_dict)
        self.store = RunStore(Path(self.temp.name) / "runs.db")
        base = evaluate(dataset, mode="keyword", top_k=1)
        candidate = evaluate(dataset, mode="bm25", top_k=1)
        self.store.save(base)
        self.store.save(candidate)
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(self.store, dataset, candidate["id"], compare(base, candidate))
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def test_demo_report_and_health(self):
        with urlopen(self.base_url + "/") as response:
            page = response.read().decode()
            self.assertIn("Find where the answer pipeline breaks", page)
            self.assertIn("Baseline → candidate", page)
        with urlopen(self.base_url + "/healthz") as response:
            self.assertEqual(json.load(response), {"status": "ok"})

    def test_create_run_is_idempotent(self):
        data = json.dumps({"dataset": self.dataset_dict, "mode": "bm25", "top_k": 1}).encode()
        request = Request(self.base_url + "/api/runs", data=data, headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            first = json.load(response)
            self.assertEqual(response.status, 201)
        with urlopen(request) as response:
            second = json.load(response)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(self.store.list()), 2)

    def test_invalid_dataset_is_rejected(self):
        request = Request(self.base_url + "/api/runs", data=b'{"dataset":{}}', headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 400)

    def test_invalid_mode_is_client_error(self):
        data = json.dumps({"dataset": self.dataset_dict, "mode": []}).encode()
        request = Request(self.base_url + "/api/runs", data=data, headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
