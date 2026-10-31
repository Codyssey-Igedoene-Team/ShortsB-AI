"""Spring 내부 호출용 임베딩 API."""

import hmac
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from encoder import Encoder, EmbeddingError, MAX_ITEMS, MAX_TEXT_CHARS, Settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    # 준비 완료 전에 모델 로딩과 warm-up 수행
    settings = Settings.from_env()
    app.state.settings = settings
    app.state.encoder = Encoder(settings)
    log.info("model_ready device=%s", app.state.encoder.model.device)
    yield
    del app.state.encoder


def authenticate(request: Request, x_api_key: str | None = Header(default=None)):
    expected = request.app.state.settings.api_key
    if expected and (not x_api_key or not hmac.compare_digest(
        expected.encode("utf-8"), x_api_key.encode("utf-8")
    )):
        raise EmbeddingError("AUTHENTICATION_REQUIRED", "내부 API 인증이 필요합니다.", 401)


app = FastAPI(title="ShortsB 임베딩 API", version="0.1.0", lifespan=lifespan,
              dependencies=[Depends(authenticate)])


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1, max_length=128, description="원본과 연결할 요청 내 고유 ID")
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS, description="임베딩할 텍스트")


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["query", "document"]
    items: list[Item] = Field(min_length=1, max_length=MAX_ITEMS)


@app.exception_handler(EmbeddingError)
async def embedding_error(request, exception):
    headers = {"Retry-After": "1"} if exception.status == 429 else None
    return JSONResponse(status_code=exception.status,
                        content={"code": exception.code, "message": exception.message}, headers=headers)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exception):
    # 기본 검증 오류에 포함될 수 있는 본문·프롬프트 출력 방지
    return JSONResponse(status_code=422, content={
        "code": "INVALID_REQUEST", "message": "요청 형식과 입력 제한을 확인해주세요."})


@app.exception_handler(Exception)
async def unexpected_error(request, exception):
    log.error("request_failed type=%s", type(exception).__name__)
    return JSONResponse(status_code=500, content={
        "code": "INTERNAL_ERROR", "message": "임베딩 요청을 처리할 수 없습니다."})


@app.get("/health", summary="모델 준비 상태 조회", description="모델 warm-up 이후 접근 가능. gpuReady와 revision·fingerprint를 확인한다.")
def health(request: Request):
    return {"ready": True, **request.app.state.encoder.metadata()}


@app.post("/embeddings", summary="텍스트 임베딩", description="본문은 document, 카드·프롬프트는 query. 원문·ID 순서 유지, 토큰 초과 거절. batch 최대 8개이며 동시 추론 초과는 429.")
def embeddings(body: EmbeddingRequest, request: Request):
    try:
        return request.app.state.encoder.embed(body.role, [item.model_dump() for item in body.items])
    except EmbeddingError:
        raise
    except Exception as error:
        # 서버 기본 traceback에도 원문을 남기지 않도록 안전한 오류로 변환
        log.error("inference_failed type=%s", type(error).__name__)
        raise EmbeddingError("MODEL_EXECUTION_FAILED", "모델 실행에 실패했습니다.", 503) from None
