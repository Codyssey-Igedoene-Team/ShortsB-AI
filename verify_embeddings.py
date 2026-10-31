"""원본 입력과 사전 임베딩 산출물의 읽기 전용 검증."""

import argparse
import json
import re
import sys
from itertools import zip_longest
from pathlib import Path

from encoder import (
    EmbeddingError, MODEL_CONFIG_FIELDS, configuration_fingerprint, validate_vectors,
)
from precompute import batches


def load_manifest(path, require_gpu=False):
    try:
        if path.stat().st_size > 1_048_576:
            raise ValueError
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or not set(MODEL_CONFIG_FIELDS) <= metadata.keys():
            raise ValueError
        if (type(metadata.get("schemaVersion")) is not int or metadata["schemaVersion"] != 1
                or metadata.get("role") not in ("document", "query")
                or type(metadata.get("recordCount")) is not int or metadata["recordCount"] < 1
                or type(metadata["dimension"]) is not int or metadata["dimension"] < 1
                or type(metadata["maxTokens"]) is not int or metadata["maxTokens"] < 1
                or metadata["normalized"] is not True):
            raise ValueError
        if (any(not isinstance(metadata[k], str) or not metadata[k].strip()
                for k in ("modelId", "revision", "preprocessingVersion", "encoder"))
                or not re.fullmatch(r"[0-9a-f]{40}", metadata["revision"])
                or any(not isinstance(metadata[k], str) for k in ("queryPrefix", "documentPrefix"))):
            raise ValueError
        libraries = metadata["libraries"]
        if (not isinstance(libraries, dict)
                or set(libraries) != {"torch", "sentence-transformers", "transformers", "numpy"}
                or any(not isinstance(v, str) or not v.strip() for v in libraries.values())
                or metadata.get("fingerprint") != configuration_fingerprint(metadata)):
            raise ValueError
        device = metadata.get("device")
        if (not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device)
                or type(metadata.get("gpuReady")) is not bool
                or metadata["gpuReady"] != device.startswith("cuda")):
            raise ValueError
    except (ValueError, TypeError, UnicodeError, KeyError):
        raise EmbeddingError("INVALID_MANIFEST", "モデル情報・設定 fingerprint を確認してください。") from None
    if require_gpu and not metadata["gpuReady"]:
        raise EmbeddingError("GPU_REQUIRED", "GPU で生成した結果が必要です。")
    return metadata


def vector_rows(path):
    with path.open(encoding="utf-8") as stream:
        while line := stream.readline(1_048_577):
            try:
                if len(line) > 1_048_576:
                    raise ValueError
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError
            except (ValueError, UnicodeError):
                raise EmbeddingError("INVALID_VECTOR_FILE", "벡터 파일 형식을 확인해주세요.") from None
            yield row


def verify(input_path, output_dir, against_manifest=None, require_gpu=False):
    metadata = load_manifest(output_dir / "manifest.json", require_gpu)
    if against_manifest is not None:
        reference = load_manifest(against_manifest, require_gpu)
        if any(metadata[key] != reference[key] for key in (*MODEL_CONFIG_FIELDS, "fingerprint")):
            raise EmbeddingError("INCOMPATIBLE_MODEL", "본문·카드의 모델 설정이 다릅니다.")
    records = (record for batch in batches(input_path, metadata["role"]) for record in batch)
    count = 0
    for record, row in zip_longest(records, vector_rows(output_dir / "vectors.jsonl")):
        if record is None or row is None:
            raise EmbeddingError("OUTPUT_MISMATCH", "입력과 벡터의 개수가 다릅니다.")
        expected = {key: value for key, value in record.items() if key != "text"}
        if (set(row) != set(expected) | {"role", "embedding"}
                or row["role"] != metadata["role"]
                or any(type(row[key]) is not type(value) or row[key] != value
                       for key, value in expected.items())
                or not isinstance(row["embedding"], list)):
            raise EmbeddingError("OUTPUT_MISMATCH", "원본 ID·순서·위치·해시·역할이 일치하지 않습니다.")
        validate_vectors([row["embedding"]], 1, metadata["dimension"])
        count += 1
    if count != metadata["recordCount"]:
        raise EmbeddingError("OUTPUT_MISMATCH", "manifest의 레코드 수가 일치하지 않습니다.")
    return count


def main():
    parser = argparse.ArgumentParser(description="본문 출력 없이 원본·벡터·모델 호환성 검증")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--against-manifest", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    try:
        count = verify(args.input, args.output_dir, args.against_manifest, args.require_gpu)
        print(f"임베딩 산출물 검증 완료: {count}개")
    except EmbeddingError as error:
        print(f"{error.code}: {error.message}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as error:
        print(f"산출물 검증 실패: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
