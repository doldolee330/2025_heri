# 문화유산 해설 생성 API

문화유산 ID, 관람객 유형, 질문을 받아서 맞춤형 자연어 해설을 생성하는 REST API 서버입니다.

## 개발 환경

### 운영체제
- **OS**: Linux (Ubuntu)
- **Kernel**: 5.4.0-177-generic
- **Architecture**: x86_64

### 개발 도구
- **Python**: 3.10.14
- **pip**: 23.3.1
- **Package Manager**: pip, conda (miniconda3)

### 주요 라이브러리 버전
- **Flask**: >=2.3.0
- **PyTorch**: >=2.0.0
- **Transformers**: >=4.35.0
- **Requests**: >=2.31.0
- **tqdm**: >=4.65.0 (데이터 전처리 스크립트용)

## 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd Q4_code
```

### 2. 가상환경 생성 및 활성화
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 3. requirements 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정 (선택사항)
기본적으로 코드는 상대 경로를 사용하지만, 환경 변수로 경로를 변경할 수 있습니다:

```bash
export MODEL_PATH="doldol330/2025_heri"
export EVIDENCE_FILE="data/aimuse_items_enriched_final.json"
export PROMPTS_FILE="data/prompts/prompt_all.txt"
export SOLR_QUERY_SCRIPT="/path/to/solr/db/query-db.sh"
```

**참고**: 
- `MODEL_PATH`는 HuggingFace 모델 경로 또는 로컬 모델 경로를 지정합니다.
- `SOLR_QUERY_SCRIPT`는 Solr 쿼리 스크립트 경로입니다. 기본값은 `../solr/db/query-db.sh`입니다.
- 다른 환경 변수들은 기본값이 설정되어 있어 생략 가능합니다.

### 5. 데이터 준비

이 프로젝트는 이미 전처리된 데이터 파일(`data/aimuse_items_enriched_final.json`)을 사용합니다.
이 파일은 이미 포함되어 있습니다.

데이터 전처리 과정을 재현하려면 `preprocess_data.py` 스크립트를 사용할 수 있습니다:

#### 데이터 수집 (선택사항)
```bash
python preprocess_data.py --collect --output data/aimuse_items_filtered.json
```

#### 데이터 전처리 (선택사항, Solr 서버 필요)
```bash
python preprocess_data.py --enrich \
    --input data/aimuse_items_filtered.json \
    --output data/aimuse_items_enriched_final.json \
    --top-k 3
```

**참고**: `--enrich` 옵션은 Solr 가 필요합니다. 
- Solr은 Q4_code 프로젝트 밖에 별도로 설치되어 있다고 가정합니다.
- 기본 경로는 `../../solr/db/query-db.sh`입니다.
- 다른 경로를 사용하려면 `SOLR_QUERY_SCRIPT` 환경 변수로 지정하세요.

## 사용 방법

### 서버 실행
```bash
python app.py
```

서버는 기본적으로 `http://0.0.0.0:5000`에서 실행됩니다.

### API 엔드포인트

#### 1. 헬스 체크
```bash
curl http://localhost:5000/health
```

응답:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 2. 해설 생성
```bash
curl "http://localhost:5000/generate/동원1257/탐구형/이%20유물에%20대해%20설명해%20주세요"
```

**파라미터:**
- `heritage_id`: 문화유산 ID (URL 인코딩 필요)
- `visitor_type`: 관람객 유형 (탐구형, 취미형, 의무형, 휴식형, 사교형)
- `question`: 질문 (URL 인코딩 필요)

**응답 예시:**
```json
{
  "문화유산_ID": "동원1257",
  "관람객_유형": "탐구형",
  "유물명": "청자 상감 운학문 매병",
  "질문": "이 유물에 대해 설명해 주세요",
  "답변": "...",
  "근거문서_요약": {
    "description": "...",
    "headword_doc_존재": true,
    "reference_doc_개수": 3
  },
  "status": "success"
}
```

## 프로젝트 구조

```
Q4_code/
├── app.py                 # 메인 API 서버
├── preprocess_data.py     # 데이터 전처리 스크립트 (참고용)
├── requirements.txt       # Python 의존성
├── README.md             # 이 파일
├── .gitignore            # Git 제외 파일 목록
└── data/                  # 데이터 파일
    ├── aimuse_items_enriched_final.json  # 전처리된 문화유산 데이터
    └── prompts/
        └── prompt_all.txt                # 관람객 유형별 프롬프트
```

**참고**: 
- `data/aimuse_items_enriched_final.json`은 이미 전처리된 최종 데이터 파일입니다.
- `preprocess_data.py`는 데이터 전처리 과정을 재현하기 위한 참고용 스크립트입니다.
- 실제 API 서버는 `data/aimuse_items_enriched_final.json` 파일을 직접 사용합니다.
- Solr 관련 스크립트는 프로젝트 밖의 `../../solr/db/` 경로에 있다고 가정합니다.

## 모델 정보

- **베이스 모델**: `google/gemma-3-27b-it`
- **Fine-tuned 모델**: HuggingFace에 업로드된 모델 사용
- **모델 경로**: `MODEL_PATH` 환경 변수로 지정

모델은 HuggingFace에서 다운로드하거나, 로컬 경로를 지정할 수 있습니다.

## 관람객 유형

API는 다음 5가지 관람객 유형을 지원합니다:

1. **탐구형**: 깊이 있는 정보와 전문적인 해설
2. **취미형**: 쉽고 간결한 설명
3. **의무형**: 기본적인 정보만 간단히 제공
4. **휴식형**: 감성적이고 편안한 설명
5. **사교형**: 대화형 서술로 상호작용 유도

## 문제 해결

### 모델 로드 실패
- `MODEL_PATH` 환경 변수가 올바른지 확인
- HuggingFace 토큰이 설정되어 있는지 확인
- GPU 메모리가 충분한지 확인

### 데이터 로드 실패
- `EVIDENCE_FILE` 경로가 올바른지 확인
- JSON 파일 형식이 올바른지 확인

### Solr 쿼리 실패
- Solr 서버가 실행 중인지 확인
- Solr이 `../../solr/db/` 경로에 설치되어 있는지 확인
- `SOLR_QUERY_SCRIPT` 환경 변수로 올바른 경로를 지정했는지 확인
- 스크립트 실행 권한이 있는지 확인


