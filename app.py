import os
import uuid
import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 🔥 RAG 챗봇 import (PolicyRAGChatbot 클래스는 존재한다고 가정)
# 🚨 중요한 수정: 테스트용 더미 클래스 정의를 삭제하고 실제 클래스를 임포트합니다.
from rag_chatbot import PolicyRAGChatbot 

# ============================================================
# 1) RAG 챗봇 초기화
# ============================================================

# 파일 경로는 .env 또는 기본값으로 설정
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", r"C:\Users\user\Desktop\bge-m3-sft")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", r"C:\Users\user\Desktop\policy_faiss.index")
METADATA_JSON_PATH = os.getenv("METADATA_JSON_PATH", r"C:\Users\user\Desktop\metadata.json")

# PolicyRAGChatbot 초기화 (실제 PolicyRAGChatbot 클래스를 사용)
chatbot = PolicyRAGChatbot(
    model_path=EMBEDDING_MODEL_PATH,
    index_path=FAISS_INDEX_PATH,
    metadata_path=METADATA_JSON_PATH,
    api_key=os.getenv("OPENAI_API_KEY"),
    device="cpu"
)


# ============================================================
# 2) FastAPI 초기 설정
# ============================================================

app = FastAPI(title="RAG Chatbot API")

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

    try:
        # 1) RAG 호출
        # PolicyRAGChatbot.answer() 메서드는 API Response 스펙에 필요한 모든 데이터를 반환해야 합니다.
        # (sourceData, queryTitle, botResponse, followUpQuestions)
        
        # PolicyRAGChatbot의 answer 메서드가 다음과 같은 딕셔너리를 반환한다고 가정합니다:
        # { 'answer': str, 'sources': List[Dict], 'query_title': str, 'follow_up_questions': List[str] }
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