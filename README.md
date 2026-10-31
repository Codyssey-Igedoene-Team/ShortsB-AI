# ShortsB-AI

KURE-v1을 직접 실행하는 FastAPI 임베딩 서버와 JSONL 사전 생성 도구.

## 현재 범위

- app.py: GET /health, POST /embeddings
- encoder.py: 고정 revision 모델 로딩·warm-up·원문 유지·토큰/벡터 검증
- precompute.py: 본문 document·고정 카드 query 사전 생성, 해시/중복 확인, 원문 없는 벡터 파일 출력
- test_contract.py: 모델 다운로드 없이 입력 경계·부분 출력 방지 확인

Spring은 인증·DB·유사도·카드/프롬프트 가중 조합·균등 무작위·대비 근거 검증·페이지·전체 독서 기록을 담당한다. FastAPI는 텍스트를 벡터로 바꾸며 DB 비밀번호·Google 세션을 받지 않는다. 학습·파인튜닝·런타임 자율 에이전트·생성형 설명 모델은 포함하지 않는다.

GPU 사용자의 사전 확인 결과는 Python 3.12.3, NVIDIA L4, PyTorch 2.11.0+cu128, CUDA 사용 가능, KURE-v1의 cuda:0 / (3,1024) 출력이다. 이 프로젝트의 실제 서버 기동·HTTP·전체 도서 batch 검증은 아직 별도 수행이 필요하다.

## 설치와 설정

로컬 VS Code에서는 별도 Python 가상환경을 사용한다. GPU 서버에서는 이미 확인한 환경을 재사용한다.

```bash
source /home1/ncloud/ai-venv/bin/activate
python -m pip install -r requirements.txt
```

requirements.txt는 호환 범위다. 서버에서 정상 동작을 확인한 뒤 사용 버전을 기록한다. 사전/온라인은 같은 환경을 사용하며 라이브러리 버전도 fingerprint에 포함한다. 기존 GPU torch를 재설치하거나 시스템 드라이버·CUDA·방화벽을 변경할 필요는 없다.

.env.example은 설명과 빈 변수 목록이다. 자동으로 읽지 않는다. VS Code 실행 설정 또는 셸 환경변수로 주입한다. 비밀값이 든 .env는 공유하지 않는다.

| 변수 | 동작 |
| --- | --- |
| EMBEDDING_MODEL_ID | 기본 nlpai-lab/KURE-v1. 현재 이 모델만 실행 지침을 확인함 |
| EMBEDDING_MODEL_REVISION | 필수. 40자리 commit SHA. main 같은 가변 이름 거절 |
| EMBEDDING_DEVICE | 기본 cuda. GPU 불가 시 기동 실패. cpu는 명시적인 로컬 진단용 |
| MODEL_CACHE_DIR | 선택. 미설정 시 라이브러리 기본 캐시 |
| EMBEDDING_API_KEY | 선택. 설정하면 health·embeddings에 X-API-Key 필요. 미설정 실행은 loopback/통제된 내부 연결에서만 사용 |

서버에 이미 받은 모델의 revision은 기본 캐시를 사용했다면 아래 파일에서 확인할 수 있다. 다른 캐시를 썼다면 해당 위치를 확인한다.

```bash
cat ~/.cache/huggingface/hub/models--nlpai-lab--KURE-v1/refs/main
```

캐시 ref가 없다면 다음은 공개 저장소의 현재 revision을 조회한다. 이 값을 사전/온라인에 동일하게 고정한다.

```bash
python -c "from huggingface_hub import model_info; print(model_info('nlpai-lab/KURE-v1').sha)"
```

서버 셸 설정 예시. SHA 자리에는 실제 확인한 값을 넣는다.

```bash
export EMBEDDING_MODEL_ID=nlpai-lab/KURE-v1
export EMBEDDING_MODEL_REVISION=확인한_40자리_SHA
export EMBEDDING_DEVICE=cuda
python -m uvicorn app:app --host 127.0.0.1 --port 8001 --workers 1
```

PowerShell 로컬 진단에서는 `$env:EMBEDDING_MODEL_REVISION="확인한_SHA"`, `$env:EMBEDDING_DEVICE="cpu"`처럼 주입한다. CPU 진단도 모델 전체를 다운로드하므로 초기 로컬 작업은 코드·계약 확인을 우선할 수 있다.

서버 startup에서 한 번 로딩하고 짧은 문장으로 warm-up한다. 최초 모델 다운로드가 필요할 수 있으며, GPU 메모리·모델을 중복 점유하지 않도록 worker 1개와 추론 1개를 사용한다. 사전 CLI와 온라인 서버도 같은 GPU에서 동시에 실행하지 않는다. 실제 시연 실행에 --reload를 사용하지 않는다.

## 내부 접속

제공 매뉴얼은 SSL-VPN 후 사설 IP SSH 접속을 안내한다. 포트·방화벽·GPU 드라이버·CUDA 변경은 운영진 허용 없이 수행하지 않는다. 앱 실행과 SSH forwarding의 허용 정책도 확인한다.

허용된 SSH forwarding을 사용하는 경우 PC에서 다음과 같이 연결할 수 있다. 실제 포트/사용자/사설 주소를 넣고 PC의 VPN·SSH 연결을 유지한다.

```bash
ssh -N -L 127.0.0.1:18001:127.0.0.1:8001 -p SSH_PORT USER@PRIVATE_IP
```

Spring은 PC의 http://127.0.0.1:18001로 호출한다. FastAPI의 새 외부 포트를 개방하지 않는다. 배포 Spring 서버는 자신의 허용된 접근 경로를 별도로 확인해야 한다. 토큰을 설정하면 Spring도 X-API-Key를 보낸다. 프런트는 FastAPI를 직접 호출하지 않는다.

매뉴얼에는 2026-11-01 13:00 서버 삭제 예정이라고 안내돼 있다. 변경/연장 여부는 미확인이다. 벡터·manifest·코드 등 필요한 산출물은 로컬에 보관하고 서버 영구 존속을 전제로 설계하지 않는다.

## API

Swagger: 연결된 서버의 /docs. 입력·출력은 본문 로그로 남기지 않는다. 인증키를 설정한 경우 요청의 X-API-Key 헤더에 같은 값을 전달한다. Swagger의 헤더 입력은 health·embeddings에 표시된다.

GET /health는 startup/warm-up 이후 200이며 ready, modelId, revision, fingerprint, dimension, normalized, maxTokens, device, gpuReady 등을 반환한다. CPU 진단에서는 gpuReady=false다. GPU 실행 확인은 device=cuda 및 gpuReady=true로 확인한다.

POST /embeddings 요청:

```json
{
  "role": "query",
  "items": [{"id": "prompt-1", "text": "실패한 뒤 다시 시작하는 이야기"}]
}
```

- role: query 또는 document. KURE-v1은 두 역할 모두 공식 예제의 encode를 사용하며 추가 접두사를 붙이지 않는다.
- 본문은 사전 document, 카드 3종은 사전 query, 사용자 자유 프롬프트는 온라인 query다. ANY는 임베딩하지 않으며 Spring에서 처리한다.
- 항목 1~8개, ID는 비어 있지 않은 문자열 최대 128자·요청 내 고유, text는 공백 제외 내용이 있으며 최대 32,768자. 텍스트를 strip 등으로 변경하지 않는다.
- 문자 수 제한과 별도로 실제 tokenizer의 특수 토큰 포함 길이를 확인한다. 모델 상한 초과는 422이며 자동 잘림·페이지 재분할로 숨기지 않는다.
- 응답: 모델 metadata + role + items[].id/embedding. 순서·개수 유지, L2 정규화된 실수 배열. 모델 revision·버전이 같아도 GPU 부동소수점 결과의 비트 단위 일치를 요구하지 않는다.
- 401 내부 키 오류, 422 입력·길이·중복, 429 모델 사용 중, 503 GPU 메모리 부족·모델 추론 실패, 500 잘못된 모델 출력·예상하지 못한 오류. 오류는 code/message 구조며 요청 text와 검증 상세 원문을 반환하지 않는다.
- Spring은 저장된 벡터와 fingerprint·dimension·normalized가 맞는지 확인한다. 모델·라이브러리·설정 변경 시 재생성하고 기존 활성 데이터와 혼합하지 않는다.
- 모델 준비 실패는 startup 실패다. health만 정상으로 띄우거나 CPU로 조용히 전환하지 않는다.

## 사전 생성

입력은 Spring/콘텐츠 도구가 준비한 JSONL이다. 제공 books.json 전체를 직접 읽는 가져오기·페이지/검색 구간 분할 기능은 이 서버에 포함하지 않는다. 해당 변환은 원본 페이지·DB/자료 ID를 정하는 작업과 함께 수행한다.

본문 document의 필수 필드:

- id, bookId, segmentId: 안정적인 문자열 원본 참조
- text: 정리된 원문 그대로
- contentHash: 위 text UTF-8 바이트의 SHA-256 소문자 64자리
- pageNo: 원본 페이지가 있으면 1-based로 전달
- startPosition/endPosition: 선택. pageNo 필수, UTF-16 기준 [start,end), 0 <= start < end. 실제 페이지 범위 검증은 제공 도구가 담당

카드 query는 id·text·contentHash·cardCode가 필요하다. cardCode는 WARM/HUMOR/THOUGHTFUL만 허용한다. 카드 검색 문장과 가중치 최종값은 Spring 계약에서 결정한다. 알 수 없는 필드는 거절한다. CSV 대비 후보를 이 CLI에서 자동 승인하거나 추천 결과로 생성하지 않는다.

```bash
python precompute.py --role document --input data/segments.jsonl --output-dir outputs/documents-v1
python precompute.py --role query --input data/cards.jsonl --output-dir outputs/cards-v1
```

출력 디렉터리는 새 경로여야 한다. 원본 해시·ID 중복·형식 오류면 실패하며 완성 경로를 공개하지 않는다. 성공 시 vectors.jsonl과 manifest.json을 함께 생성한다. vectors에는 원문 text를 저장하지 않고 원본 참조·해시·역할·벡터만 저장한다. manifest는 원본 개수·모델 revision·설정 fingerprint·라이브러리 버전·생성 시각을 포함한다.

사전 생성과 온라인 모델·설정은 동일하게 유지한다. 파일/manifest를 Spring에 반영하는 DB 작업은 별도다. token 상한 초과가 나오면 입력 담당자가 표시 페이지를 유지하면서 내부 구간만 분할해 안정 ID·해시·위치 매핑을 다시 전달한다.

## 확인과 남은 작업

```bash
python -m unittest -v test_contract
```

표준 라이브러리 검사는 가짜 encoder로 입력·파일 계약을 검증한다. HTTP 검사도 실행하려면 `python -m pip install httpx`로 확인 도구를 추가한다. fastapi/httpx가 없으면 HTTP 항목은 생략된다. GPU 모델 다운로드·추론 품질·HTTP 통합 확인을 대체하지 않는다. 실제 서버에서 revision 설정 → health → query/document → 사전 파일의 모델 설정 일치를 확인하고, 처리 시간·버전을 기록한다.

페이지 기준은 기존 pages 유지, 발췌량은 min(요청,전체), 신규 전체 독서는 첫 페이지다. ANY 단독은 작품 균등 선택 후 첫 페이지 발췌다. 대비는 검수한 6개 축과 양쪽 근거가 실제 발췌에 포함될 때만 제공하고 없으면 생략한다. 이 정책과 상태 보관·독서 버전 처리는 Spring 담당이다. 공유는 무기한 보류다.

본문은 대용량이므로 에이전트가 TXT를 열거나 books.json의 text/pages를 출력해 읽지 않는다. 프로그램 처리는 본문을 출력하지 않는 방식으로 수행한다. 코드 주석은 짧은 한국어 명사형으로 작성한다.

## 공식 자료

- [KURE-v1 모델 카드](https://huggingface.co/nlpai-lab/KURE-v1)
- [Sentence Transformers API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)
- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)
