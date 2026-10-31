"""모델 다운로드 없이 입력 경계와 사전 출력의 무결성 확인."""

import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from encoder import EmbeddingError, MAX_TEXT_CHARS, Settings, validate_items, validate_vectors
from precompute import batches, generate


class FakeEncoder:
    def metadata(self):
        return {"modelId": "test-model", "revision": "a" * 40, "dimension": 2,
                "normalized": True, "fingerprint": "test-fingerprint"}

    def embed(self, role, items):
        return {"items": [{"id": item["id"], "embedding": [1.0, 0.0]} for item in items]}


def record(item_id="page-1", text="검증용 짧은 문장"):
    return {"id": item_id, "bookId": "book-1", "segmentId": item_id,
            "pageNo": 1, "text": text,
            "contentHash": hashlib.sha256(text.encode()).hexdigest()}


class ContractTests(unittest.TestCase):
    def test_invalid_ids_and_text_are_rejected_without_echoing_body(self):
        for items in ([{"id": "a", "text": " "}],
                      [{"id": "a", "text": "본문"}, {"id": "a", "text": "본문"}],
                      [{"id": "a", "text": "x" * (MAX_TEXT_CHARS + 1)}]):
            with self.subTest(count=len(items)), self.assertRaises(EmbeddingError) as failure:
                validate_items("query", items)
            self.assertNotIn("본문", str(failure.exception))

    def test_vectors_must_be_finite_normalized_and_match_dimension(self):
        for vectors in ([[math.nan, 0]], [[math.inf, 0]], [[0, 0]], [[1, 0, 0]], [[2, 0]], [[True, 0]]):
            with self.subTest(vectors=vectors), self.assertRaises(EmbeddingError):
                validate_vectors(vectors, 1, 2)
        validate_vectors([[1.0, 0.0]], 1, 2)

    def test_revision_is_required_and_cpu_is_explicit(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            Settings.from_env()
        with patch.dict("os.environ", {"EMBEDDING_MODEL_REVISION": "a" * 40,
                                       "EMBEDDING_DEVICE": "cpu"}, clear=True):
            self.assertEqual(Settings.from_env().device, "cpu")

    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.jsonl"
            row = record()
            row["text"] = "바뀐 문장"
            source.write_text(json.dumps(row) + "\n")
            with self.assertRaises(EmbeddingError) as failure:
                list(batches(source, "document"))
            self.assertEqual(failure.exception.code, "HASH_MISMATCH")

    def test_failure_does_not_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "input.jsonl", Path(tmp) / "output"
            # 첫 batch 처리 후 다음 batch에서 중복 ID 발생
            rows = [record(f"page-{i}") for i in range(8)] + [record("page-0")]
            source.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaises(EmbeddingError):
                generate(source, output, "document", FakeEncoder())
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(tmp).glob(".embedding-*")))

    def test_success_keeps_ids_metadata_and_hash_but_omits_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "input.jsonl", Path(tmp) / "output"
            row = record()
            source.write_text(json.dumps(row))
            self.assertEqual(generate(source, output, "document", FakeEncoder()), 1)
            result = json.loads((output / "vectors.jsonl").read_text())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(result["id"], row["id"])
            self.assertEqual(result["contentHash"], row["contentHash"])
            self.assertEqual(result["pageNo"], 1)
            self.assertNotIn("text", result)
            self.assertEqual(manifest["recordCount"], 1)
            with self.assertRaises(EmbeddingError):
                generate(source, output, "document", FakeEncoder())

    def test_any_is_not_a_precomputed_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "cards.jsonl"
            row = record()
            row.pop("bookId")
            row.pop("segmentId")
            row.pop("pageNo")
            row["cardCode"] = "ANY"
            source.write_text(json.dumps(row))
            with self.assertRaises(EmbeddingError) as failure:
                list(batches(source, "query"))
            self.assertEqual(failure.exception.code, "INVALID_CARD")


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
                     "HTTP 확인에는 fastapi와 httpx 설치 필요")
class ApiContractTests(unittest.TestCase):
    def setUp(self):
        import app as api
        from fastapi.testclient import TestClient
        from types import SimpleNamespace

        class ApiEncoder(FakeEncoder):
            def __init__(self, settings):
                self.model = SimpleNamespace(device="cuda:0")

            def embed(self, role, items):
                validate_items(role, items)
                if items[0]["id"] == "failure":
                    raise RuntimeError("출력되면 안 되는 사용자 입력")
                return {**self.metadata(), "role": role, **super().embed(role, items)}

        self.environment = patch.dict("os.environ", {
            "EMBEDDING_MODEL_REVISION": "a" * 40, "EMBEDDING_API_KEY": "test-only-key",
        }, clear=True)
        self.encoder = patch.object(api, "Encoder", ApiEncoder)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.encoder.start()
        self.addCleanup(self.encoder.stop)
        self.client = TestClient(api.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.headers = {"X-API-Key": "test-only-key"}

    def test_authentication_and_health(self):
        self.assertEqual(self.client.get("/health").status_code, 401)
        health = self.client.get("/health", headers=self.headers)
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["ready"])
        self.assertNotIn("api_key", health.json())

    def test_request_contract_and_validation_never_echo_input(self):
        payload = {"role": "query", "items": [{"id": "prompt-1", "text": "짧은 확인 문장"}]}
        result = self.client.post("/embeddings", headers=self.headers, json=payload)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["items"][0]["id"], "prompt-1")
        self.assertNotIn("짧은 확인 문장", result.text)
        payload["items"][0]["unexpected"] = "노출되면 안 되는 입력"
        invalid = self.client.post("/embeddings", headers=self.headers, json=payload)
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.json()["code"], "INVALID_REQUEST")
        self.assertNotIn("노출되면 안 되는 입력", invalid.text)
        payload["items"][0].pop("unexpected")
        payload["items"] *= 2
        duplicate = self.client.post("/embeddings", headers=self.headers, json=payload)
        self.assertEqual(duplicate.status_code, 422)
        self.assertEqual(duplicate.json()["code"], "DUPLICATE_ID")

    def test_model_failure_returns_sanitized_503(self):
        result = self.client.post("/embeddings", headers=self.headers, json={
            "role": "query", "items": [{"id": "failure", "text": "사용자 문장"}]})
        self.assertEqual(result.status_code, 503)
        self.assertEqual(result.json()["code"], "MODEL_EXECUTION_FAILED")
        self.assertNotIn("사용자", result.text)


if __name__ == "__main__":
    unittest.main()
