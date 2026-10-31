"""KURE-v1 공통 임베딩과 입력·출력 검증."""

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from importlib.metadata import version

MAX_ITEMS = 8
MAX_TEXT_CHARS = 32768
MODEL_CONFIG_FIELDS = (
    "modelId", "revision", "dimension", "normalized", "maxTokens",
    "preprocessingVersion", "encoder", "queryPrefix", "documentPrefix", "libraries",
)


def configuration_fingerprint(metadata):
    config = {key: metadata[key] for key in MODEL_CONFIG_FIELDS}
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class EmbeddingError(Exception):
    # 본문을 포함하지 않는 오류 계약
    def __init__(self, code, message, status=422):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


@dataclass(frozen=True)
class Settings:
    model_id: str
    revision: str
    device: str
    cache_dir: str | None
    api_key: str

    @classmethod
    def from_env(cls):
        revision = os.getenv("EMBEDDING_MODEL_REVISION", "").strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("EMBEDDING_MODEL_REVISION에 고정된 40자리 commit SHA가 필요합니다.")
        device = (os.getenv("EMBEDDING_DEVICE") or "cuda").strip()
        if device != "cpu" and not re.fullmatch(r"cuda(?::[0-9]+)?", device):
            raise ValueError("EMBEDDING_DEVICE는 cpu 또는 cuda[:번호] 형식입니다.")
        return cls(
            (os.getenv("EMBEDDING_MODEL_ID") or "nlpai-lab/KURE-v1").strip(),
            revision, device, os.getenv("MODEL_CACHE_DIR") or None,
            os.getenv("EMBEDDING_API_KEY", ""),
        )


def validate_items(role, items):
    # 사전 처리와 API가 공유하는 입력 경계
    if role not in ("query", "document"):
        raise EmbeddingError("INVALID_ROLE", "role을 확인해주세요.")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS:
        raise EmbeddingError("INVALID_BATCH", f"항목 수는 1~{MAX_ITEMS}개입니다.")
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise EmbeddingError("INVALID_ITEM", "항목 구조를 확인해주세요.")
        item_id, text = item.get("id"), item.get("text")
        if not isinstance(item_id, str) or not item_id.strip() or len(item_id) > 128:
            raise EmbeddingError("INVALID_ID", "항목 ID를 확인해주세요.")
        if item_id in seen:
            raise EmbeddingError("DUPLICATE_ID", "요청 내 ID가 중복됩니다.")
        seen.add(item_id)
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("INVALID_TEXT", "비어 있지 않은 텍스트가 필요합니다.")
        if len(text) > MAX_TEXT_CHARS:
            raise EmbeddingError("INPUT_TOO_LONG", "입력 길이 제한을 초과했습니다.")
        try:
            text.encode("utf-8")
        except UnicodeError:
            raise EmbeddingError("INVALID_TEXT", "유효한 UTF-8 텍스트가 필요합니다.") from None


def validate_vectors(vectors, count, dimension):
    # 차원·유한 실수·정규화 검증
    if len(vectors) != count:
        raise EmbeddingError("INVALID_MODEL_OUTPUT", "벡터 개수가 일치하지 않습니다.", 500)
    for vector in vectors:
        if len(vector) != dimension or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) for value in vector
        ):
            raise EmbeddingError("INVALID_MODEL_OUTPUT", "벡터 형식을 확인할 수 없습니다.", 500)
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise EmbeddingError("INVALID_MODEL_OUTPUT", "정규화된 벡터가 아닙니다.", 500)


class Encoder:
    # 고정 revision 모델 1회 로딩과 단일 GPU 추론
    def __init__(self, settings):
        import torch
        from sentence_transformers import SentenceTransformer

        self._torch = torch
        self._lock = threading.Lock()
        if settings.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("GPU를 사용할 수 없습니다. CPU로 자동 전환하지 않습니다.")
        self.model = SentenceTransformer(
            settings.model_id, revision=settings.revision, device=settings.device,
            cache_folder=settings.cache_dir, trust_remote_code=False,
        )
        self.model.float()
        self.model.eval()
        if self.model.device.type != settings.device.split(":")[0]:
            raise RuntimeError("요청한 실행 장치와 실제 모델 장치가 다릅니다.")
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.max_tokens = self.model.max_seq_length
        self.settings = settings
        config = {
            "modelId": settings.model_id, "revision": settings.revision,
            "dimension": self.dimension, "normalized": True,
            "maxTokens": self.max_tokens, "preprocessingVersion": "verbatim-v1",
            "encoder": "sentence-transformers-encode-float32",
            "queryPrefix": "", "documentPrefix": "",
            "libraries": {name: version(name) for name in
                          ("torch", "sentence-transformers", "transformers", "numpy")},
        }
        self.config = config
        self.fingerprint = configuration_fingerprint(config)
        self.embed("query", [{"id": "warmup", "text": "임베딩 서버 준비 확인"}])

    def metadata(self):
        return {**self.config, "fingerprint": self.fingerprint,
                "device": str(self.model.device),
                "gpuReady": self.model.device.type == "cuda"}

    def embed(self, role, items):
        validate_items(role, items)
        # ponytail: 단일 추론 제한. 동시 처리 수가 필요하면 GPU 메모리 측정 후 확장
        if not self._lock.acquire(blocking=False):
            raise EmbeddingError("MODEL_BUSY", "다른 임베딩 요청을 처리 중입니다.", 429)
        try:
            texts = [item["text"] for item in items]
            # 모델의 자동 잘림 전 토큰 수 확인
            for text in texts:
                tokens = self.model.tokenizer(text, add_special_tokens=True, truncation=False,
                                              return_attention_mask=False,
                                              return_token_type_ids=False)["input_ids"]
                if len(tokens) > self.max_tokens:
                    raise EmbeddingError("INPUT_TOO_LONG", "모델 입력 토큰 상한을 초과했습니다.")
            try:
                vectors = self.model.encode(
                    texts, batch_size=1, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False, prompt="",
                ).tolist()
            except self._torch.cuda.OutOfMemoryError:
                self._torch.cuda.empty_cache()
                raise EmbeddingError("GPU_MEMORY_EXHAUSTED", "GPU 메모리가 부족합니다.", 503) from None
            validate_vectors(vectors, len(items), self.dimension)
            return {**self.metadata(), "role": role,
                    "items": [{"id": item["id"], "embedding": vector}
                              for item, vector in zip(items, vectors, strict=True)]}
        finally:
            self._lock.release()
