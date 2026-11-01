# ShortsB-AI

KURE-v1을 직접 실행하는 FastAPI 임베딩 서버와 JSONL 사전 생성 도구.

## 현재 범위

- app.py: GET /health, POST /embeddings
- encoder.py: 고정 revision 모델 로딩·warm-up·원문 유지·토큰/벡터 검증
- prepare_catalog.py: 제공 books.json 구조·해시 검증, 기존 페이지와 UTF-16 위치를 유지한 JSONL 변환
- precompute.py: 모델 없는 입력 검사와 본문 document·고정 카드 query 사전 생성
- verify_embeddings.py: 원본 ID/위치/해시·벡터·본문/카드 모델 호환성 검사
- config/cards-v1.jsonl: 카드 3종 검색 문장 초안과 원문 해시
- test_contract.py·test_catalog.py·test_precompute.py: 모델 다운로드 없이 API·입력·위치·벡터 파일·부분 출력 방지 확인

Spring은 인증·DB·유사도·카드/프롬프트 가중 조합·균등 무작위·대비 근거 검증·페이지·전체 독서 기록을 담당한다. FastAPI는 텍스트를 벡터로 바꾸며 DB 비밀번호·Google 세션을 받지 않는다. 학습·파인튜닝·런타임 자율 에이전트·생성형 설명 모델은 포함하지 않는다.

실행 구성은 로컬 프론트·Spring·PostgreSQL과 제공 GPU 서버의 FastAPI다. 사용자가 NVIDIA L4에서 실제 FastAPI 기동, `/health` 및 문장 임베딩 200 응답, 본문 864개·카드 3개 사전 생성/산출물 검증, 로컬 DB 등록과 첫 추천·대비 추천 연결 성공을 확인했다. 프런트 전체 흐름과 모든 추천 조합의 검증 완료를 뜻하지는 않는다.

## 설치와 설정

### 확인한 실행 환경

| 항목 | 확인한 값 |
| --- | --- |
| Python | 3.12.3 |
| GPU | NVIDIA L4 |
| PyTorch | 2.11.0+cu128 |
| Sentence Transformers | 6.2.0 |
| Transformers | 5.22.1 |
| NumPy | 2.5.4 |
| 모델 | nlpai-lab/KURE-v1 |
| 모델 revision | 8b418a58414668e75532ed045c22d9ca018ae2b2 |
| 모델 fingerprint | 521e40a63c92f2444806079d8e51436cdee38b8fac6e87309e5b77d706e7206d |

서버의 기존 환경을 재사용하면 가상환경을 활성화하고 프로젝트 폴더로 이동한다. 기존 설치 환경에 패키지를 다시 설치할 필요는 없다.

```bash
cd ~/ShortsB-AI
source /home1/ncloud/ai-venv/bin/activate
```

새 Linux GPU 환경에서는 Python 3.12와 CUDA 실행이 가능한 NVIDIA 드라이버를 먼저 준비한 후 다음 순서로 설치한다. 제공 GPU 서버의 드라이버·CUDA를 임의 변경하지 않는다. 아래 CUDA 12.8 빌드는 확인한 L4 환경용이며 GTX 1050 등 모든 GPU에 적용되는 설치 방법이 아니다.

```bash
cd ~/ShortsB-AI
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install "sentence-transformers==6.2.0" "transformers==5.22.1" "numpy==2.5.4" "fastapi>=0.115,<1" "uvicorn>=0.30,<1"
```

**현재 `requirements.txt`의 `sentence-transformers>=3,<6`은 확인한 서버 버전 6.2.0과 다르다. 기존 산출물을 재사용하는 제출 재현에서는 위 버전 지정 설치를 사용하며, `pip install -r requirements.txt`를 추가로 실행하지 않는다.** 이번 수정은 README에 한정하며 requirements 파일은 변경하지 않았다. FastAPI·Uvicorn의 정확한 서버 버전은 수집되지 않았고 위 범위로 안내한다.

```bash
python -c "import numpy, torch, sentence_transformers, transformers, fastapi, uvicorn; print('패키지 준비 완료'); print('PyTorch:', torch.__version__); print('GPU 사용 가능:', torch.cuda.is_available())"
```

사전/온라인은 같은 환경을 사용한다. fingerprint에는 위 주요 라이브러리 버전도 포함되므로 패키지나 모델을 변경하면 원래 산출물과 fingerprint가 달라질 수 있다. 이때 DB fingerprint만 바꾸지 않고 실행 환경을 맞추거나 본문·카드 벡터를 모두 재생성한다.

### 환경변수와 실행

.env.example은 설명과 빈 변수 목록이다. 자동으로 읽지 않는다. VS Code 실행 설정 또는 셸 환경변수로 주입한다. 비밀값이 든 .env는 공유하지 않는다.

| 변수 | 동작 |
| --- | --- |
| EMBEDDING_MODEL_ID | 기본 nlpai-lab/KURE-v1. 현재 이 모델만 실행 지침을 확인함 |
| EMBEDDING_MODEL_REVISION | 필수. 40자리 commit SHA. main 같은 가변 이름 거절 |
| EMBEDDING_DEVICE | 기본 cuda. GPU 불가 시 기동 실패. cpu는 명시적인 로컬 진단용 |
| MODEL_CACHE_DIR | 선택. 미설정 시 라이브러리 기본 캐시 |
| EMBEDDING_API_KEY | 선택. 설정하면 health·embeddings에 X-API-Key 필요. 미설정 실행은 loopback/통제된 내부 연결에서만 사용 |

확인한 모델 버전을 아래처럼 고정한다. 모델 revision과 fingerprint는 공개 모델/설정 식별자이며 인증키가 아니다. 현재 터미널에만 적용되므로 새 SSH 접속에서는 다시 설정한다.

```bash
export EMBEDDING_MODEL_ID=nlpai-lab/KURE-v1
export EMBEDDING_MODEL_REVISION=8b418a58414668e75532ed045c22d9ca018ae2b2
export EMBEDDING_DEVICE=cuda
python -m uvicorn app:app --host 127.0.0.1 --port 8001 --workers 1
```

기동 후 GPU 서버에서 확인한다. 다른 터미널을 열거나 서버를 실행한 연결을 유지한다.

```bash
curl -s http://127.0.0.1:8001/health
```

`ready: true`, `device: cuda:0`, `gpuReady: true`와 위 revision/fingerprint를 확인한다. `address already in use`가 나오면 기존 FastAPI가 살아 있을 수 있으므로 먼저 이 health를 확인하고 다른 사람의 프로세스를 임의 종료하지 않는다. 인증키를 설정했다면 이 요청에도 같은 `X-API-Key` 헤더가 필요하다. 실제 키는 명령 이력이나 공유 문서에 남기지 않도록 주의한다.

CPU 진단은 `EMBEDDING_DEVICE=cpu`를 명시할 때 가능하지만, Spring은 CPU 응답/CPU 생성 산출물을 거부하므로 제출용 전체 연동 경로는 GPU를 사용한다.

서버 startup에서 한 번 로딩하고 짧은 문장으로 warm-up한다. 최초 모델 다운로드가 필요할 수 있으며, GPU 메모리·모델을 중복 점유하지 않도록 worker 1개와 추론 1개를 사용한다. 사전 CLI와 온라인 서버도 같은 GPU에서 동시에 실행하지 않는다. 실제 시연 실행에 --reload를 사용하지 않는다.

## 내부 접속

제공 매뉴얼은 SSL-VPN 후 사설 IP SSH 접속을 안내한다. 포트·방화벽·GPU 드라이버·CUDA 변경은 운영진 허용 없이 수행하지 않는다. 앱 실행과 SSH forwarding의 허용 정책도 확인한다.

확인한 시연은 VPN 연결 후 PuTTY로 GPU 서버에 접속해 SSH 포워딩을 설정하는 방법이다. PuTTY 창의 `Change Settings → Connection → SSH → Tunnels`에서 다음을 추가하고 `Apply`를 누른다. 이미 같은 항목이 있으면 중복 추가하지 않는다.

| 항목 | 값 |
| --- | --- |
| Source port | `18001` |
| Destination | `127.0.0.1:8001` |
| 종류 | `Local`, `Auto` |
| 다른 호스트의 로컬 포트 접속 허용 | 체크하지 않음 |

노트북에서 `http://127.0.0.1:18001/health`에 접속해 모델 정보가 나오면 연결 성공이다. Spring 실행 환경의 `APP_EMBEDDING_BASE_URL=http://127.0.0.1:18001`로 설정한다. VPN과 PuTTY를 유지하며 재접속 후 터널도 확인한다. 같은 PC에서 별도 FastAPI가 18001 포트를 사용하고 있으면 먼저 포트 충돌을 해소한다.

명령줄 SSH를 사용하는 경우 실제 접속 주소·포트·사용자로 대체한다.

```bash
ssh -N -L 127.0.0.1:18001:127.0.0.1:8001 -p SSH_PORT USER@PRIVATE_IP
```

FastAPI의 새 외부 포트는 개방하지 않는다. 내부 인증키를 사용한다면 `EMBEDDING_API_KEY`와 Spring의 `APP_EMBEDDING_API_KEY`를 같게 설정하며 프런트는 FastAPI를 직접 호출하지 않는다. 키가 없는 현재 시연 방식은 loopback·VPN·SSH 터널 안에서만 사용한다.

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

## 로컬 도서 준비

현재 카탈로그는 `data/catalog-v1/`, 사전 임베딩 산출물은 `outputs/documents-v1/`과 `outputs/cards-v1/`에 있다. **이 경로들은 `.gitignore` 제외 대상**이므로 GitHub 소스만 받은 환경에는 없을 수 있다. 임베딩 산출물은 아래 링크에서 내려받고 제공된 파일을 다음 위치에 배치한다. 묶음에 없는 원본·카탈로그는 별도로 준비한다.

- [임베딩 산출물 다운로드 — Google Drive](https://drive.google.com/file/d/1CGvm3BOziZGNqz-rcfsDLtORNrPosMtI/view?usp=sharing)

```text
data/catalog-v1/
  catalog-metadata.json
  segments.jsonl
  manifest.json
outputs/documents-v1/
  vectors.jsonl
  manifest.json
outputs/cards-v1/
  vectors.jsonl
  manifest.json
```

카탈로그가 있으면 아래 원본 변환을 건너뛴다. 산출물까지 있으면 GPU 사전 생성도 건너뛰고 산출물 검사 후 Spring DB에 등록한다. 산출물만 없으면 사전 생성 절차로 다시 만든다. 서버를 재접속할 때마다 재생성할 필요는 없다.

카탈로그도 없는 경우에는 제출팀에서 원본 `books.json`을 받아 아래 경로에 배치한다. 예시는 `ShortsB-BE`와 `ShortsB-AI`가 같은 상위 폴더에 있는 구성이다. 대소문자를 포함한 실제 폴더명에 맞춰 경로를 조정한다. 원본 없이 출처 URL만으로 동일 AI 한국어 번역판을 재현할 수 있다고 가정하지 않는다.

준비 CLI는 표준 라이브러리만 사용하며 GPU·모델·DB·환경변수가 필요하지 않다. FastAPI 요청 처리와 분리되어 실행한다.

```bash
python prepare_catalog.py --input ../ShortsB-BE/docs/shortB-15books-texts/books.json --validate-only
python prepare_catalog.py --input ../ShortsB-BE/docs/shortB-15books-texts/books.json --output-dir data/catalog-v1
```

새 출력 디렉터리를 사용한다. 이미 생성한 data/catalog-v1을 다시 만들려면 다른 버전 경로를 지정한다. 원본 books.json·TXT는 수정하지 않는다.

- segments.jsonl: 기존 pages 하나를 document 구간 하나로 변환. text·SHA-256·외부 bookId·안정 segmentId·1-based pageNo·UTF-16 [startPosition,endPosition) 보존. Java/브라우저 문자열 위치와 일치하도록 보조 평면 문자도 UTF-16 두 단위로 계산한다.
- catalog-metadata.json: 본문 text/pages를 제외한 판본·출처·번역 안내·원본 챕터 정보. chapters는 원자료의 0-based [start,end) 상태이며 업무 API에서 변환한다. licenseURL/licenseLabel/credits의 null을 임의 보충하지 않는다.
- manifest.json: 입력 파일 SHA-256·자료 버전·작품/페이지/구간 수·위치 단위·생성 시각. 모델 임베딩 manifest와 구분한다.

도서 수·외부 ID 중복·필수 메타데이터 자료형·빈 페이지·페이지 재결합·전체 해시·쪽수·챕터 범위/정렬/포함 관계·선택 pageIds를 확인한다. 로그에는 본문이나 잘못된 필드 값을 넣지 않는다. 실패하면 완성 디렉터리를 남기지 않고 기존 출력도 덮어쓰지 않는다.

현재 제공 자료는 15개 작품·864페이지이며 최대 페이지 길이는 1,093자다. 현재 CLI는 페이지 내부를 다시 나누지 않는다. 실제 토큰 상한 검증은 precompute.py의 encoder에서 수행하고 초과 시 실패한다. 긴 페이지 자료가 추가되면 표시 페이지를 유지한 내부 구간 분할을 추가한다. 구조·해시 검증은 원작 대비 전문 완전성·이용 권한·대비 근거 승인을 뜻하지 않는다. 현재 자료에 null인 권한 안내 필드가 있으므로 별도 검토가 필요하다.

## 로컬 카드 입력과 모델 없는 검사

config/cards-v1.jsonl은 문서에 있던 카드 검색 문장 초안을 실행 가능한 입력 파일로 관리한다. Git에 포함할 수 있는 설정 파일이며 data/의 생성 파일과 구분한다. 검색 품질을 확인한 최종 문장이라는 뜻은 아니다.

| 카드 코드 | 초기 검색 문장 |
| --- | --- |
| WARM | 위로와 희망, 사람 사이의 배려와 유대가 느껴지는 이야기 |
| HUMOR | 재치 있는 대화와 상황, 해학으로 웃음을 주는 이야기 |
| THOUGHTFUL | 삶과 인간관계, 선택에 관해 질문을 던지고 생각하게 하는 이야기 |

각 카드가 한 번씩 있어야 하며 ANY는 포함하지 않는다. ID에는 입력 버전과 내용 해시 일부를 포함한다. 문장을 변경하면 contentHash와 ID도 함께 갱신하고 새 버전으로 벡터를 생성한다. 카드/프롬프트 점수 가중치는 이 파일에서 정하지 않는다.

```bash
python precompute.py --role document --input data/catalog-v1/segments.jsonl --validate-only
python precompute.py --role query --input config/cards-v1.jsonl --validate-only
```

이 검사는 모델·GPU·환경변수 없이 입력의 ID·해시·위치 형식·카드 누락/중복을 확인한다. 실제 tokenizer 길이·벡터·검색 품질은 검사하지 않는다. 현재 제공 자료의 document 864개와 카드 query 3개 입력이 통과했다.

## 사전 생성

입력은 위 CLI 또는 Spring/콘텐츠 도구가 준비한 JSONL이다. FastAPI는 요청마다 books.json을 읽거나 페이지를 변경하지 않는다. 외부 bookId·segmentId와 DB 식별자의 연결은 Spring 가져오기에서 관리한다.

본문 document의 필수 필드:

- id, bookId, segmentId: 안정적인 문자열 원본 참조
- text: 정리된 원문 그대로
- contentHash: 위 text UTF-8 바이트의 SHA-256 소문자 64자리
- pageNo: 원본 페이지가 있으면 1-based로 전달
- startPosition/endPosition: 선택. pageNo 필수, UTF-16 기준 [start,end), 0 <= start < end. 실제 페이지 범위 검증은 제공 도구가 담당

카드 query는 id·text·contentHash·cardCode가 필요하다. cardCode는 WARM/HUMOR/THOUGHTFUL만 허용한다. 카드 검색 문장은 GPU 검색 결과를 보고 최종 조정하며 가중치 최종값은 Spring 계약에서 결정한다. 카드 파일에는 WARM/HUMOR/THOUGHTFUL이 각각 하나씩 있어야 한다. 알 수 없는 필드는 거절한다. CSV 대비 후보를 이 CLI에서 자동 승인하거나 추천 결과로 생성하지 않는다.

먼저 온라인 FastAPI가 해당 GPU를 점유하고 있다면 정상 종료한다. 같은 가상환경과 위 고정 환경변수를 설정한 상태에서 아래 두 명령을 하나씩 순서대로 실행한다. 본문 생성이 끝난 후 카드 생성을 시작하며, 완료 후 온라인 FastAPI를 다시 실행한다.

```bash
python precompute.py --role document --input data/catalog-v1/segments.jsonl --output-dir outputs/documents-v1
python precompute.py --role query --input config/cards-v1.jsonl --output-dir outputs/cards-v1
```

출력 디렉터리는 새 경로여야 한다. 원본 해시·ID 중복·형식 오류면 실패하며 완성 경로를 공개하지 않는다. 성공 시 vectors.jsonl과 manifest.json을 함께 생성한다. vectors에는 원문 text를 저장하지 않고 원본 참조·해시·역할·벡터만 저장한다. manifest는 원본 개수·모델 revision·설정 fingerprint·라이브러리 버전·생성 시각을 포함한다.

현재 카드 입력은 config/cards-v1.jsonl의 초안이다. GPU 결과를 검토해 최종 문구를 결정한다. ANY는 사전 생성하지 않는다.

사전 생성과 온라인 모델·설정은 동일하게 유지한다. 파일/manifest를 Spring에 반영하는 DB 작업은 별도다. token 상한 초과가 나오면 입력 담당자가 표시 페이지를 유지하면서 내부 구간만 분할해 안정 ID·해시·위치 매핑을 다시 전달한다.

## 사전 생성 결과 검사

GPU 서버에서 precompute.py 실행이 끝나면 원본 입력과 결과를 대조한다. 본문을 출력하거나 DB에 반영하지 않는 읽기 전용 검사다.

```bash
python verify_embeddings.py --input data/catalog-v1/segments.jsonl --output-dir outputs/documents-v1 --require-gpu
python verify_embeddings.py --input config/cards-v1.jsonl --output-dir outputs/cards-v1 --against-manifest outputs/documents-v1/manifest.json --require-gpu
```

- 입력/결과의 개수·ID·순서·작품/페이지/위치·해시·역할이 일치해야 한다. 결과에 원문 text가 들어 있어도 실패한다.
- 벡터 차원·유한 실수·L2 정규화와 manifest의 레코드 수를 확인한다.
- 설정 fingerprint를 다시 계산하고, 본문/카드의 모델 ID·revision·라이브러리·입력 설정을 비교한다. 일부만 다른 벡터를 함께 사용하지 않는다.
- --require-gpu는 기록된 device·gpuReady가 GPU 생성 결과인지 확인한다. CPU 합성 검사에서는 생략할 수 있으며 실제 GPU 추론·추천 품질 확인을 대체하지 않는다.

사전 생성 도구도 모델 응답의 ID·개수·벡터를 확인한 뒤 출력한다. 잘못된 응답은 완성 파일로 공개하지 않는다. 검사 성공은 벡터 파일의 일관성을 뜻하며 이용 권한·대비 근거 승인은 별도다.

## 확인과 남은 작업

```bash
python -m unittest -v test_contract test_catalog test_precompute
```

기존 합성 encoder 기반 Python/HTTP 검사 20개 통과 기록은 실제 GPU 확인과 구분한다. HTTP 항목은 추가 도구 `httpx`가 필요하며 이 문서 수정 과정에서 테스트를 실행하지 않았다. 새 테스트 코드는 작성하지 않는다.

2026-11-01 사용자의 실제 실행 보고는 다음과 같다.

- L4 GPU에서 `/health`의 CUDA·GPU 준비 상태 확인, `/embeddings` query 요청 200 응답.
- 본문 864개와 카드 3개 GPU 사전 생성 및 두 산출물 검사 성공.
- Spring 로컬 DB에 본문 벡터 864개·카드 3개·추천 설정 1건 등록 확인.
- VPN·PuTTY 터널을 통해 Spring의 프롬프트 추천과 백야 → B사감과 러브레터 대비 응답 확인.
- 전체 독서 시작·페이지 조회·위치 저장·이어 읽기, 대비 관계 45개 후보 SQL 적용 성공 보고. 모든 관계의 API 응답·전체 독서 완료·프런트 통합은 별도 확인 필요.

산출물을 로컬로 가져올 때 각 폴더의 `vectors.jsonl`과 `manifest.json`을 함께 전송한다. Spring의 `importCatalog`로 검증·등록하며 경로와 명령은 백엔드 README를 따른다. Spring에 DB 정보와 Google 설정을 주입하고 AI에는 전달하지 않는다. 신규 도서는 별도 승인 전 추천에 나오지 않는다.

페이지 기준은 기존 pages 유지, 발췌량은 min(요청,전체), 신규 전체 독서는 첫 페이지다. ANY 단독은 작품 균등 선택 후 첫 페이지 발췌다. 대비는 검수한 6개 축과 양쪽 근거가 실제 발췌에 포함될 때만 제공하고 없으면 생략한다. 이 정책과 상태 보관·독서 버전 처리는 Spring 담당이다. 공유는 무기한 보류다.

본문은 대용량이므로 에이전트가 TXT를 열거나 books.json의 text/pages를 출력해 읽지 않는다. 프로그램 처리는 본문을 출력하지 않는 방식으로 수행한다. 코드 주석은 짧은 한국어 명사형으로 작성한다.

## 공식 자료

- [KURE-v1 모델 카드](https://huggingface.co/nlpai-lab/KURE-v1)
- [Sentence Transformers API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)
- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)
