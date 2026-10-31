"""모델 없이 카드 입력과 사전 벡터 산출물 검증."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import precompute
from encoder import EmbeddingError, configuration_fingerprint
from precompute import batches, generate
from test_contract import FakeEncoder, record
from verify_embeddings import verify

CARDS = Path(__file__).parent / "config" / "cards-v1.jsonl"


class PrecomputeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.output = self.root / "input.jsonl", self.root / "documents"
        self.source.write_text("\n".join(json.dumps(record(f"page-{i}")) for i in range(2)),
                               encoding="utf-8")

    def test_three_cards_and_documents_use_compatible_model_metadata(self):
        generate(self.source, self.output, "document", FakeEncoder())
        self.assertEqual(verify(self.source, self.output), 2)
        cards_output = self.root / "cards"
        self.assertEqual(generate(CARDS, cards_output, "query", FakeEncoder()), 3)
        self.assertEqual(verify(CARDS, cards_output, self.output / "manifest.json"), 3)
        rows = [row for batch in batches(CARDS, "query") for row in batch]
        self.assertEqual({row["cardCode"] for row in rows}, {"WARM", "HUMOR", "THOUGHTFUL"})

    def test_missing_duplicate_or_invalid_card_code_rejected(self):
        rows = [row for batch in batches(CARDS, "query") for row in batch]
        variants = [rows[:2], [rows[0], {**rows[0], "id": "different-id"}, rows[2]],
                    [{**rows[0], "cardCode": ["WARM"]}, *rows[1:]]]
        for index, variant in enumerate(variants):
            path = self.root / f"cards-{index}.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in variant), encoding="utf-8")
            with self.subTest(index=index), self.assertRaises(EmbeddingError):
                list(batches(path, "query"))

    def test_wrong_response_ids_or_vectors_never_publish_output(self):
        encoder = FakeEncoder()
        correct = encoder.embed("document", [record("page-0"), record("page-1")])
        variants = [{"items": list(reversed(correct["items"]))}, {"items": []},
                    {"items": [{**item, "embedding": [0.0, 0.0]} for item in correct["items"]]}]
        for response in variants:
            with self.subTest(), patch.object(encoder, "embed", return_value=response):
                with self.assertRaises(EmbeddingError):
                    generate(self.source, self.output, "document", encoder)
            self.assertFalse(self.output.exists())
            self.assertFalse(list(self.root.glob(".embedding-*")))

    def test_tampered_vectors_counts_and_source_mapping_rejected(self):
        generate(self.source, self.output, "document", FakeEncoder())
        path = self.output / "vectors.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        variants = [rows[:1], rows + [rows[0]], list(reversed(rows))]
        for field, value in (("contentHash", "0" * 64), ("pageNo", True), ("role", "query"),
                             ("text", "합성 비공개 본문"), ("embedding", [1.0, 0.0, 0.0]),
                             ("embedding", [float("nan"), 0.0]), ("embedding", [2.0, 0.0])):
            mutated = copy.deepcopy(rows)
            mutated[0][field] = value
            variants.append(mutated)
        for index, variant in enumerate(variants):
            path.write_text("\n".join(json.dumps(row) for row in variant), encoding="utf-8")
            with self.subTest(index=index), self.assertRaises(EmbeddingError) as failure:
                verify(self.source, self.output)
            self.assertNotIn("합성 비공개", str(failure.exception))

    def test_manifest_fingerprint_counts_gpu_and_model_compatibility(self):
        generate(self.source, self.output, "document", FakeEncoder())
        path = self.output / "manifest.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for field, value in (("fingerprint", "0" * 64), ("dimension", 3), ("recordCount", 3),
                             ("normalized", False), ("gpuReady", True)):
            path.write_text(json.dumps({**original, field: value}), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(EmbeddingError):
                verify(self.source, self.output)
        path.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaises(EmbeddingError) as failure:
            verify(self.source, self.output, require_gpu=True)
        self.assertEqual(failure.exception.code, "GPU_REQUIRED")
        reference = {**original, "revision": "b" * 40}
        reference["fingerprint"] = configuration_fingerprint(reference)
        other = self.root / "other-manifest.json"
        other.write_text(json.dumps(reference), encoding="utf-8")
        with self.assertRaises(EmbeddingError) as failure:
            verify(self.source, self.output, other)
        self.assertEqual(failure.exception.code, "INCOMPATIBLE_MODEL")

    def test_validate_only_does_not_load_model_or_need_environment(self):
        output = io.StringIO()
        args = ["precompute.py", "--role", "query", "--input", str(CARDS), "--validate-only"]
        with (patch("sys.argv", args), patch.dict("os.environ", {}, clear=True),
              patch.object(precompute, "Encoder", side_effect=AssertionError("모델 로딩 금지")),
              contextlib.redirect_stdout(output)):
            precompute.main()
        self.assertIn("입력 검증 완료: 3개", output.getvalue())


if __name__ == "__main__":
    unittest.main()
