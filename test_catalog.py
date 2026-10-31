"""합성 본문으로 원본 위치·해시·출력 원자성 확인."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from encoder import EmbeddingError
from precompute import batches, generate
from prepare_catalog import METADATA_FIELDS, prepare, sha256, validate_catalog
from test_contract import FakeEncoder


def sample():
    pages = ["합성 첫 페이지 😀\n", "합성 두 번째 페이지"]
    text = "\n\n".join(pages)
    book = {key: "검증용 메타데이터" for key in METADATA_FIELDS}
    book.update(id="sample-book", text=text, pages=pages, pageCount=len(pages),
                pageJoiner="\n\n", textSHA256=sha256(text), translationNote="",
                chapters=[{"id": "chapter-1", "title": "합성 챕터", "start": 0,
                           "end": 2, "available": True, "pageIds": ["page-a", "page-b"]}])
    return {"version": "test-v1", "bookCount": 1, "books": [book]}


class CatalogTests(unittest.TestCase):
    def test_conversion_keeps_page_text_utf16_positions_and_source_metadata(self):
        catalog = sample()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.json"
            source.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
            output = Path(tmp) / "prepared"
            manifest = prepare(source, output)
            rows = [row for batch in batches(output / "segments.jsonl", "document") for row in batch]
            metadata = json.loads((output / "catalog-metadata.json").read_text(encoding="utf-8"))
            book = catalog["books"][0]
            self.assertEqual([row["text"] for row in rows], book["pages"])
            self.assertEqual([row["pageNo"] for row in rows], [1, 2])
            self.assertEqual(rows[0]["endPosition"], len(book["pages"][0]) + 1)
            for row in rows:
                page = book["pages"][row["pageNo"] - 1].encode("utf-16-le")
                restored = page[row["startPosition"] * 2:row["endPosition"] * 2].decode("utf-16-le")
                self.assertEqual(sha256(restored), row["contentHash"])
            self.assertEqual(metadata["books"][0]["chapters"], book["chapters"])
            self.assertEqual(metadata["books"][0]["licenseURL"], book["licenseURL"])
            self.assertNotIn("text", metadata["books"][0])
            self.assertNotIn("pages", metadata["books"][0])
            self.assertEqual(manifest["segmentCount"], 2)
            self.assertEqual(manifest["modelTokenValidation"], "pending-precompute")
            second = Path(tmp) / "prepared-again"
            prepare(source, second)
            self.assertEqual((output / "segments.jsonl").read_bytes(),
                             (second / "segments.jsonl").read_bytes())
            vectors = Path(tmp) / "vectors"
            self.assertEqual(generate(output / "segments.jsonl", vectors, "document", FakeEncoder()), 2)
            generated = [json.loads(line) for line in (vectors / "vectors.jsonl").read_text().splitlines()]
            self.assertEqual(generated[0]["id"], rows[0]["id"])
            self.assertNotIn("text", generated[0])
            with self.assertRaises(EmbeddingError):
                prepare(source, output)

    def test_bad_structure_hash_pages_and_chapters_are_rejected_without_body(self):
        bad_books = []
        for field, value in (("pageCount", 3), ("pageCount", True), ("textSHA256", "0" * 64),
                             ("text", "숨겨야 하는 합성 본문"), ("pages", ["", "두 번째"]),
                             ("chapters", [{"id": "a", "title": "합성", "start": -1,
                                             "end": 2, "available": True}])):
            data = sample()
            data["books"][0][field] = value
            bad_books.append(data)
        data = sample()
        data["books"].append(copy.deepcopy(data["books"][0]))
        data["bookCount"] = 2
        bad_books.append(data)
        data = sample()
        data["books"][0]["chapters"][0]["pageIds"] = ["duplicate", "duplicate"]
        bad_books.append(data)
        for data in bad_books:
            with self.subTest(), self.assertRaises(EmbeddingError) as failure:
                validate_catalog(data)
            self.assertNotIn("숨겨야 하는", str(failure.exception))

    def test_nested_chapters_allowed_but_crossing_and_reverse_order_rejected(self):
        data = sample()
        data["books"][0].update(licenseURL=None, licenseLabel=None, credits=None)
        parent = data["books"][0]["chapters"][0]
        child = {**parent, "id": "child", "start": 0, "end": 1, "pageIds": ["page-a"]}
        data["books"][0]["chapters"].append(child)
        validate_catalog(data)
        data["books"][0]["chapters"] = [child, parent]
        with self.assertRaises(EmbeddingError):
            validate_catalog(data)

    def test_write_failure_removes_staging_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source.json", Path(tmp) / "output"
            source.write_text(json.dumps(sample()), encoding="utf-8")
            with patch.object(Path, "write_text", side_effect=OSError("검증용 저장 실패")):
                with self.assertRaises(OSError):
                    prepare(source, output)
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(tmp).glob(".catalog-*")))


if __name__ == "__main__":
    unittest.main()
