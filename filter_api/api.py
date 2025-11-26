# filter_api/api.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import APIRouter, HTTPException, status
from filter_api.model import FilterQueryRequest, FilterResponse, ErrorResponse
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 전역 변수 (모델 및 토크나이저 저장)
# ============================================================

# 모델 로드 상태 및 객체 저장
TOXICITY_MODEL = None
TOXICITY_TOKENIZER = None
DEVICE = "cpu" # 초기값, initialize_toxicity_model에서 변경됨


print("DEBUG TOXIC_PATH=", os.getenv("TOXICITY_MODEL_PATH"))

# 모델 파일 경로 (필요에 따라 .env에서 가져오거나 하드코딩)
MODEL_DIR = os.getenv("TOXICITY_MODEL_PATH")
if not MODEL_DIR:
    raise RuntimeError("X TOXICITY_MODEL_PATH(.env)이 설정되지 않았습니다.")

# ============================================================
# 1) 모델 초기화 함수
# ============================================================

def initialize_toxicity_model():
    """클린봇 AI 모델과 토크나이저를 로드하고 전역 변수에 저장합니다."""
    global TOXICITY_MODEL, TOXICITY_TOKENIZER, DEVICE
    
    # 모델 로드 상태 확인 (중복 로드 방지)
    if TOXICITY_MODEL is not None:
        print("ℹ️ 유해성 필터링 모델이 이미 로드되었습니다.")
        return

    # GPU/CPU 설정
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🚀 유해성 필터링 모델 로드 중... (DEVICE: {DEVICE})")
    
    # 모델/토크나이저 로드
    TOXICITY_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
    TOXICITY_MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    TOXICITY_MODEL.to(DEVICE)
    TOXICITY_MODEL.eval() # 평가 모드 설정
    
    print("✅ 유해성 필터링 모델 로드 완료.")

# ============================================================
# 2) 예측 함수 (클린봇 로직)
# ============================================================

def predict_toxicity(text: str) -> dict:
    """
    텍스트의 유해성 확률을 예측합니다. (클린봇ai.py 로직 기반)
    
    Returns: { "정상 확률": float, "유해 확률": float }
    """
    if TOXICITY_MODEL is None or TOXICITY_TOKENIZER is None:
        # 모델이 로드되지 않은 경우 오류 발생
        raise RuntimeError("유해성 필터링 모델이 초기화되지 않았습니다.")
        
    # 토큰화
    inputs = TOXICITY_TOKENIZER(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 모델 예측
    with torch.no_grad():
        outputs = TOXICITY_MODEL(**inputs)

    logits = outputs.logits
    # Softmax를 적용하여 확률 계산. [0]은 배치 차원 제거
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0] 
    
    # 모델의 라벨 순서에 따라 확률을 반환 (0: 정상, 1: 유해라고 가정)
    return {
        "정상 확률": probs[0],
        "유해 확률": probs[1]
    }

# ============================================================
# 3) API 응답 생성 함수 (더미 데이터 제거 반영)
# ============================================================

def create_filter_response(text: str) -> dict:
    """
    API 명세에 맞는 필터링 응답을 생성합니다.
    """
    
    # 유해성 예측
    prediction = predict_toxicity(text)
    
    # toxicity는 유해 확률을 사용
    toxicity_score = prediction["유해 확률"]
    
    # blocked 결정: toxicity >= 0.7이면 True (임계값 0.7 사용)
    blocked = toxicity_score >= 0.7
    
    # 🚨 더미 데이터 로직 제거: 다른 점수들은 0으로 설정
    insult = 0.0
    profanity = 0.0
    hate = 0.0
    threat = 0.0
    
    return {
        "originQuery": text,
        "toxicity": round(toxicity_score, 4),
        "insult": insult,
        "profanity": profanity,
        "hate": hate,
        "threat": threat,
        "blocked": blocked
    }

# ============================================================
# 4) FastAPI 라우터 정의
# ============================================================

# APIRouter 인스턴스 생성
router = APIRouter(
    prefix="/api/v1/filter",
    tags=["Toxicity Filter (Cleanbot AI)"]
)

@router.post(
    "/text", 
    response_model=FilterResponse,
    responses={
        400: {"model": ErrorResponse, "description": "필수 파라미터 누락"},
        500: {"model": ErrorResponse, "description": "내부 서버 오류 (모델 로드 실패)"}
    },
    summary="텍스트 유해성 필터링",
    description="커뮤니티 콘텐츠(게시글/댓글)의 유해성을 탐지합니다."
)
def filter_text_endpoint(request: FilterQueryRequest):
    """
    텍스트 유해성 필터링 API
    """
    
    # query 필드 검증
    if not request.query or request.query.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "BAD_REQUEST",
                "message": "필수 파라미터가 누락되었습니다.",
                "details": {"missing_fields": ["query"]}
            }
        )
    
    try:
        # 유해성 필터링 실행
        filter_result = create_filter_response(request.query)
        return FilterResponse(**filter_result)
        
    except RuntimeError as e:
        # 모델 초기화 관련 오류는 500으로 처리
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "MODEL_NOT_INITIALIZED",
                "message": str(e)
            }
        )
    except Exception as e:
        # 기타 예외 처리
        print(f"Filter API Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "유해성 필터링 중 예상치 못한 오류가 발생했습니다."
            }
        )