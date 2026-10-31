"""원본 페이지를 유지하는 도서 검증과 사전 임베딩 입력 생성."""

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from encoder import EmbeddingError, validate_items

METADATA_FIELDS = (
    "id", "title", "author", "category", "workType", "language", "edition",
    "editionLabel", "txtFile", "sourceURL", "sourceName", "licenseURL",
    "licenseLabel", "credits", "translationNote",
)
NULLABLE_FIELDS = {"licenseURL", "licenseLabel", "credits"}


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def invalid(book_no, rule, page_no=None):
    # 원문·잘못된 필드 값을 포함하지 않는 검증 오류
    location = f"{book_no}번째 작품"
    if page_no is not None:
        location += f" {page_no}페이지"
    raise EmbeddingError("INVALID_CATALOG", f"{location}: {rule}")


def validate_catalog(catalog):
    if (not isinstance(catalog, dict)
            or not isinstance(catalog.get("version"), str) or not catalog["version"].strip()
            or not isinstance(catalog.get("books"), list) or not catalog["books"]
            or type(catalog.get("bookCount")) is not int
            or catalog["bookCount"] != len(catalog["books"])):
        raise EmbeddingError("INVALID_CATALOG", "version·bookCount·books 구조를 확인해주세요.")
    seen = set()
    for book_no, book in enumerate(catalog["books"], 1):
        if (not isinstance(book, dict) or not set(METADATA_FIELDS) <= book.keys()
                or any(not isinstance(book[k], str) for k in METADATA_FIELDS if k not in NULLABLE_FIELDS)
                or any(book[k] is not None and not isinstance(book[k], str) for k in NULLABLE_FIELDS)
                or any(not book[k].strip() for k in METADATA_FIELDS
                       if k not in NULLABLE_FIELDS and k != "translationNote")):
            invalid(book_no, "필수 메타데이터 누락 또는 잘못된 자료형")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", book["id"]):
            invalid(book_no, "외부 작품 ID는 영문·숫자·밑줄·하이픈 1~64자")
        if book["id"] in seen:
            invalid(book_no, "중복 작품 ID")
        seen.add(book["id"])
        pages = book.get("pages")
        if (not isinstance(pages, list) or not pages
                or type(book.get("pageCount")) is not int or book["pageCount"] != len(pages)
                or not isinstance(book.get("pageJoiner"), str)
                or not isinstance(book.get("text"), str)):
            invalid(book_no, "본문·페이지 구조 또는 쪽수 불일치")
        for page_no, page in enumerate(pages, 1):
            try:
                validate_items("document", [{"id": "page", "text": page}])
            except EmbeddingError:
                invalid(book_no, "빈 페이지·인코딩·문자 상한 확인", page_no)
        if book["pageJoiner"].join(pages) != book["text"]:
            invalid(book_no, "페이지 재결합 결과와 전체 본문 불일치")
        if book.get("textSHA256") != sha256(book["text"]):
            invalid(book_no, "전체 본문 SHA-256 불일치")
        chapters = book.get("chapters")
        if not isinstance(chapters, list):
            invalid(book_no, "chapters 자료형 확인")
        chapter_ids, parents, last_start = set(), [], -1
        for chapter in chapters:
            if (not isinstance(chapter, dict)
                    or any(not isinstance(chapter.get(k), str) or not chapter[k].strip()
                           for k in ("id", "title"))
                    or type(chapter.get("available")) is not bool
                    or type(chapter.get("start")) is not int
                    or type(chapter.get("end")) is not int):
                invalid(book_no, "챕터 필드 확인")
            start, end = chapter["start"], chapter["end"]
            if (chapter["id"] in chapter_ids or not 0 <= start < end <= len(pages)
                    or start < last_start):
                invalid(book_no, "챕터 ID·0-based 범위·정렬 확인")
            chapter_ids.add(chapter["id"])
            last_start = start
            while parents and start >= parents[-1]:
                parents.pop()
            if parents and end > parents[-1]:
                invalid(book_no, "챕터 범위는 분리되거나 완전히 포함되어야 함")
            parents.append(end)
            if "pageIds" in chapter:
                ids = chapter["pageIds"]
                if (not isinstance(ids, list) or len(ids) != end - start
                        or any(not isinstance(pid, str) or not pid.strip() for pid in ids)
                        or len(set(ids)) != len(ids)):
                    invalid(book_no, "챕터 pageIds 자료형·개수·중복 확인")
    return {"bookCount": len(catalog["books"]),
            "pageCount": sum(len(book["pages"]) for book in catalog["books"])}


def load_catalog(path):
    try:
        source = path.read_bytes()
        catalog = json.loads(source.decode("utf-8-sig"))
    except (ValueError, UnicodeError):
        raise EmbeddingError("INVALID_CATALOG", "UTF-8 JSON 형식을 확인해주세요.") from None
    validate_catalog(catalog)
    return catalog, hashlib.sha256(source).hexdigest()


def prepare(input_path, output_dir):
    # ponytail: 현재 자료는 페이지당 최대 1,093자. 토큰 초과 발생 시 내부 구간 분할 추가
    if output_dir.exists():
        raise EmbeddingError("OUTPUT_EXISTS", "출력 경로가 이미 있습니다. 새 경로를 사용해주세요.")
    catalog, source_hash = load_catalog(input_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".catalog-", dir=output_dir.parent))
    try:
        count = 0
        with (staging / "segments.jsonl").open("w", encoding="utf-8") as stream:
            for book in catalog["books"]:
                for page_no, text in enumerate(book["pages"], 1):
                    digest = sha256(text)
                    segment_id = f"{book['id']}:p{page_no}:{digest[:16]}"
                    record = {"id": segment_id, "bookId": book["id"],
                              "segmentId": segment_id, "pageNo": page_no,
                              "startPosition": 0, "endPosition": len(text.encode("utf-16-le")) // 2,
                              "text": text, "contentHash": digest}
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    count += 1
        # 출처·번역 안내·챕터 원본 보존, 본문 중복 저장 제외
        metadata = {**catalog, "books": [
            {k: value for k, value in book.items() if k not in ("text", "pages")}
            for book in catalog["books"]
        ]}
        manifest = {"schemaVersion": 1, "sourceSHA256": source_hash,
                    "sourceVersion": catalog["version"], **validate_catalog(catalog),
                    "segmentCount": count, "mappingVersion": "provided-pages-utf16-v1",
                    "pageIndexBase": 1, "chapterIndexBase": 0,
                    "positionUnit": "UTF-16", "endPositionExclusive": True,
                    "modelTokenValidation": "pending-precompute",
                    "rightsApproval": "not-evaluated",
                    "createdAt": datetime.now(timezone.utc).isoformat()}
        for filename, value in (("catalog-metadata.json", metadata), ("manifest.json", manifest)):
            (staging / filename).write_text(
                json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        if output_dir.exists():
            raise EmbeddingError("OUTPUT_EXISTS", "출력 경로가 이미 있습니다.")
        staging.rename(output_dir)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description="본문 출력 없이 books.json 검증·JSONL 변환")
    parser.add_argument("--input", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate-only", action="store_true")
    action.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.validate_only:
            catalog, _ = load_catalog(args.input)
            result = validate_catalog(catalog)
        else:
            result = prepare(args.input, args.output_dir)
        print(f"도서 검증 완료: {result['bookCount']}개 작품, {result['pageCount']}페이지")
    except EmbeddingError as error:
        print(f"{error.code}: {error.message}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as error:
        print(f"도서 준비 실패: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
