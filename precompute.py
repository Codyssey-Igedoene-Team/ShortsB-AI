"""본문을 출력하지 않는 JSONL 사전 임베딩."""

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from encoder import Encoder, EmbeddingError, MAX_ITEMS, Settings, validate_items, validate_vectors

FIELDS = {"id", "bookId", "segmentId", "cardCode", "pageNo", "startPosition", "endPosition", "text", "contentHash"}
CARD_CODES = {"WARM", "HUMOR", "THOUGHTFUL"}


def batches(path, role):
    # 안정 ID·해시 검증과 파일 전체 중복 차단
    seen, cards, batch = set(), set(), []
    with path.open(encoding="utf-8") as stream:
        while line := stream.readline(1_048_577):
            if len(line) > 1_048_576:
                raise EmbeddingError("INVALID_RECORD", "JSONL 한 행이 너무 큽니다.")
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError):
                raise EmbeddingError("INVALID_RECORD", "JSONL 형식을 확인해주세요.") from None
            allowed = FIELDS if role == "document" else {"id", "cardCode", "text", "contentHash"}
            if not isinstance(record, dict) or set(record) - allowed:
                raise EmbeddingError("INVALID_RECORD", "허용되지 않는 레코드 구조입니다.")
            validate_items(role, [record])
            if record["id"] in seen:
                raise EmbeddingError("DUPLICATE_ID", "파일 내 원본 ID가 중복됩니다.")
            seen.add(record["id"])
            digest = record.get("contentHash")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise EmbeddingError("INVALID_HASH", "SHA-256 필드가 필요합니다.")
            if hashlib.sha256(record["text"].encode("utf-8")).hexdigest() != digest:
                raise EmbeddingError("HASH_MISMATCH", "입력 해시가 일치하지 않습니다.")
            if role == "document":
                if any(not isinstance(record.get(k), str) or not record[k].strip()
                       for k in ("bookId", "segmentId")):
                    raise EmbeddingError("INVALID_SOURCE", "본문 원본 ID가 필요합니다.")
                if record.get("cardCode") is not None:
                    raise EmbeddingError("INVALID_SOURCE", "본문에 카드 코드를 넣을 수 없습니다.")
            else:
                code = record.get("cardCode")
                if not isinstance(code, str) or code not in CARD_CODES:
                    raise EmbeddingError("INVALID_CARD", "사전 query는 고정 카드 3종만 허용합니다.")
                if code in cards:
                    raise EmbeddingError("DUPLICATE_CARD", "카드 코드는 파일 내 한 번만 사용할 수 있습니다.")
                cards.add(code)
            page = record.get("pageNo")
            if page is not None and (type(page) is not int or page < 1):
                raise EmbeddingError("INVALID_POSITION", "pageNo는 1부터 시작합니다.")
            start, end = record.get("startPosition"), record.get("endPosition")
            if start is not None or end is not None:
                if page is None or type(start) is not int or type(end) is not int or not 0 <= start < end:
                    raise EmbeddingError("INVALID_POSITION", "원본 페이지 범위를 확인해주세요.")
            batch.append(record)
            if len(batch) == MAX_ITEMS:
                yield batch
                batch = []
    if role == "query" and cards != CARD_CODES:
        raise EmbeddingError("INCOMPLETE_CARDS", "WARM·HUMOR·THOUGHTFUL 카드가 각각 하나씩 필요합니다.")
    if batch:
        yield batch


def generate(input_path, output_dir, role, encoder):
    # 완성된 디렉터리만 공개하고 기존 결과 덮어쓰기 방지
    if output_dir.exists():
        raise EmbeddingError("OUTPUT_EXISTS", "출력 경로가 이미 있습니다. 새 경로를 사용해주세요.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".embedding-", dir=output_dir.parent))
    try:
        count = 0
        metadata = encoder.metadata()
        with (staging / "vectors.jsonl").open("w", encoding="utf-8") as out:
            for batch in batches(input_path, role):
                result = encoder.embed(role, batch)
                # 원본 순서와 모델 응답의 ID·벡터 대조
                items = result.get("items") if isinstance(result, dict) else None
                if (not isinstance(items, list) or len(items) != len(batch)
                        or any(not isinstance(item, dict) or item.get("id") != record["id"]
                               or not isinstance(item.get("embedding"), list)
                               for record, item in zip(batch, items, strict=True))):
                    raise EmbeddingError("INVALID_MODEL_OUTPUT", "모델 응답 ID·개수가 일치하지 않습니다.", 500)
                validate_vectors([item["embedding"] for item in items], len(batch), metadata["dimension"])
                for record, item in zip(batch, items, strict=True):
                    row = {key: value for key, value in record.items() if key != "text"}
                    row.update(role=role, embedding=item["embedding"])
                    out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    count += 1
        if not count:
            raise EmbeddingError("EMPTY_INPUT", "입력 레코드가 없습니다.")
        manifest = {**metadata, "schemaVersion": 1, "role": role,
                    "recordCount": count, "createdAt": datetime.now(timezone.utc).isoformat()}
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if output_dir.exists():
            raise EmbeddingError("OUTPUT_EXISTS", "출력 경로가 이미 있습니다.")
        staging.rename(output_dir)
        return count
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description="ID·해시를 검증하는 사전 임베딩")
    parser.add_argument("--input", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output-dir", type=Path)
    action.add_argument("--validate-only", action="store_true")
    parser.add_argument("--role", choices=("query", "document"), required=True)
    args = parser.parse_args()
    try:
        if not args.input.is_file() or (args.output_dir and args.output_dir.exists()):
            raise EmbeddingError("INVALID_PATH", "입력 파일과 새 출력 경로를 확인해주세요.")
        if args.validate_only:
            count = sum(len(batch) for batch in batches(args.input, args.role))
            if not count:
                raise EmbeddingError("EMPTY_INPUT", "입력 레코드가 없습니다.")
            print(f"입력 검증 완료: {count}개 (모델 토큰·벡터 검증 전)")
            return
        count = generate(args.input, args.output_dir, args.role, Encoder(Settings.from_env()))
        print(f"사전 임베딩 완료: {count}개")
    except EmbeddingError as error:
        print(f"{error.code}: {error.message}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as error:
        # 라이브러리 오류에도 본문·토큰·인증값 노출 방지
        print(f"사전 처리 실패: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
