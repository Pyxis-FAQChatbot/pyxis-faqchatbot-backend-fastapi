import os
import uuid
import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests

# .env 파일 로드
load_dotenv()

# 🔥 RAG 챗봇 import (PolicyRAGChatbot 클래스는 존재한다고 가정)
from rag_api.rag_chatbot import PolicyRAGChatbot 

# -------------------------------------------------------------
# 1. 클린봇 AI 라우터 임포트 및 모델 초기화 (---추가됨---)
# -------------------------------------------------------------
from title_api.api import router as title_router, initialize_title_client 


# ============================================================
# 1) 모든 모델/클라이언트 초기화
# ============================================================

# --- RAG 챗봇 초기화 ---
# 파일 경로는 .env 또는 기본값으로 설정
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
METADATA_JSON_PATH = os.getenv("METADATA_JSON_PATH")
CLEANBOT_URL = os.getenv("CLEANBOT_URL", "http://localhost:9000/predict")

if not EMBEDDING_MODEL_PATH:
    raise RuntimeError("❌ EMBEDDING_MODEL_PATH (.env) 가 설정되지 않았습니다.")
if not FAISS_INDEX_PATH:
    raise RuntimeError("❌ FAISS_INDEX_PATH (.env) 가 설정되지 않았습니다.")
if not METADATA_JSON_PATH:
    raise RuntimeError("❌ METADATA_JSON_PATH (.env) 가 설정되지 않았습니다.")


# PolicyRAGChatbot 초기화 (실제 PolicyRAGChatbot 클래스를 사용)
chatbot = PolicyRAGChatbot(
    model_path=EMBEDDING_MODEL_PATH,
    index_path=FAISS_INDEX_PATH,
    metadata_path=METADATA_JSON_PATH,
    api_key=os.getenv("OPENAI_API_KEY"),
    device="cpu"
)

    
# --- GPT 제목 생성 클라이언트 초기화 (---추가됨---)
try:
    initialize_title_client()
except Exception as e:
    print(f"⚠️ 경고: GPT 제목 생성 클라이언트 초기화 실패. 제목 생성 기능이 작동하지 않을 수 있습니다.")
    print(f"오류 상세: {e}")


# ============================================================
# 2) FastAPI 초기 설정
# ============================================================

app = FastAPI(title="RAG, Cleanbot, & Title Generation API") # 타이틀 업데이트

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 3) 요청/응답 모델 정의 (API 스펙에 맞춤)
# ============================================================

# SourceData 상세 구조
class SourceItem(BaseModel):
    title: str
    source: str
    url: str
    snippet: str

# Request Model
class UserInfo(BaseModel):
    loginId: str
    nickname: str

class QueryRequest(BaseModel):
    user: UserInfo = Field(..., description="사용자 식별 정보")
    query: str = Field(..., description="사용자가 입력한 질문 원문")

# Response Model
class QueryResponse(BaseModel):
    sourceData: List[SourceItem] = Field(..., description="답변 근거 목록")
    queryTitle: str = Field(..., description="질문에 대한 요약")
    botResponse: str = Field(..., description="사용자에게 표시될 최종 답변 텍스트")
    followUpQuestions: Optional[List[str]] = Field(None, description="추천 후속 질문 목록")


# ============================================================
# 💥 CleanBot 호출 함수 (새로 추가됨)
# ============================================================

def is_toxic(text: str) -> bool:
    try:
        res = requests.post(CLEANBOT_URL, json={"text": text}, timeout=3)
        if res.status_code != 200:
            return False
        data = res.json()
        return data.get("toxic", False)
    except Exception:
        print("⚠️ CleanBot 서버 접속 실패 — 필터링 건너뜀")
        return False
# -------------------------------------------------------------
# 2. 라우터 등록 (---추가됨---)
# -------------------------------------------------------------
app.include_router(title_router)


# ============================================================
# 4) API 구현: 챗봇응답 (POST /api/v1/query)
# ============================================================

@app.post("/api/v1/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    """
    사용자 질문을 받아 RAG 챗봇을 통해 답변을 생성하고,
    API 스펙에 맞는 형식으로 반환합니다.
    """
    if not request.query:
        # 필수 파라미터(query) 누락에 대한 오류 응답 처리
        raise HTTPException(
            status_code=400,
            detail={
                "error": "BAD_REQUEST",
                "message": "필수 파라미터가 누락되었습니다.",
                "details": {"missing_fields": ["query"]}
            }
        )

    # 1) 먼저 CleanBot 검사
    if is_toxic(request.query):
        raise HTTPException(
            status_code=406,
            detail={"error": "TOXIC_CONTENT", "message": "유해성 콘텐츠가 포함되어 있습니다."}
        )
    
    try:
        # 1) RAG 호출
        # PolicyRAGChatbot.answer() 메서드는 API Response 스펙에 필요한 모든 데이터를 반환해야 합니다.
        result = chatbot.answer(request.query)
        
        # 2) 결과 파싱 및 응답 모델에 맞게 데이터 변환
        source_data_list = []
        for src in result.get('sources', []):
            source_data_list.append(SourceItem(
                title=src.get('title', 'N/A'),
                source=src.get('source', 'N/A'),
                url=src.get('url', 'N/A'),
                snippet=src.get('snippet', 'N/A')
            ))
        
        return QueryResponse(
            sourceData=source_data_list,
            queryTitle=result.get('query_title', request.query),  
            botResponse=result.get('answer', '죄송합니다. 답변을 생성하는 데 실패했습니다.'), 
            followUpQuestions=result.get('follow_up_questions') 
        )
        
    except Exception as e:
        # 챗봇 처리 중 발생한 예외 처리
        print(f"RAG Chatbot Error: {e}")
        raise HTTPException(
            status_code=500,
            detail="챗봇 답변 생성 중 오류가 발생했습니다."
        )

# ============================================================
# 6) 헬스체크 엔드포인트 (---추가됨---)
# ============================================================

@app.get("/health")
def health_check():
    """API 서버 상태 확인"""
    # title_api.api 모듈에서 TITLE_GENERATION_CLIENT 상태를 가져오기 위해 동적 임포트를 사용합니다.
    try:
        import title_api.api
        title_status = "active" if title_api.api.TITLE_GENERATION_CLIENT is not None else "failed"
    except (ImportError, AttributeError):
        title_status = "unknown"
    
    return {
        "status": "healthy",
        "services": {
            "rag_chatbot": "active",
            "toxicity_filter": "active",
            "title_generation": title_status
        }
    }


# ============================================================
# 7) 서버 실행 (로컬 테스트용)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    # uvicorn 실행 전에 필요한 모든 초기화가 완료되어야 함
    uvicorn.run(app, host="0.0.0.0", port=8000)