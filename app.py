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
# from rag_chatbot import PolicyRAGChatbot 

# 🚨 주의: PolicyRAGChatbot 클래스가 없으므로, 테스트를 위해 임시 더미 클래스로 대체합니다.
# 실제 환경에서는 위 주석 처리된 PolicyRAGChatbot을 사용해야 합니다.
class PolicyRAGChatbot:
    def __init__(self, **kwargs):
        # 실제 챗봇 초기화 로직 (모델 로드, 인덱스 로드 등)
        print("RAG Chatbot Initialized (Dummy)")

    def answer(self, user_query: str) -> Dict[str, Any]:
        """
        RAG 결과를 모방하여 반환합니다.
        실제 PolicyRAGChatbot은 answer 함수가
        {'answer': str, 'sources': List[Dict]} 형태를 반환한다고 가정합니다.
        """
        # 이 부분은 실제 PolicyRAGChatbot의 answer() 메소드를 사용하는 곳입니다.
        
        # 실제 RAG 챗봇을 호출하여 결과(answer, sources)를 얻습니다.
        # result = self.real_chatbot.answer(user_query)
        # bot_answer = result['answer']
        # sources = result['sources']
        
        # --- API 스펙에 맞추어 더미 데이터 생성 ---
        bot_answer = f"네, **청년창업지원금**에 대한 답변입니다. 현재 {user_query}와 관련된 예비창업패키지 모집 공고에 따르면, 만 39세 이하인 자로 사업자 등록을 하지 않은 예비 창업자를 대상으로 합니다."
        sources = [
            {
                "title": "2025년 예비창업패키지 모집 공고",
                "source": "K-스타트업",
                "url": "https://www.k-startup.go.kr/web/contents/bizpbanc-ongoing.do?pbancSn=167908",
                "snippet": "신청자격: 사업 공고일 기준으로 만 39세 이하인 자로, 사업자 등록을 하지 않은 예비 창업자..."
            }
        ]
        
        return {
            "answer": bot_answer,
            "sources": sources,
            # API 응답 스펙을 위해 추가적인 정보 (제목, 후속 질문)를 이 단계에서 준비하거나,
            # 아니면 최종 API 함수에서 준비해야 합니다.
            "query_title": "청년 창업지원금 조건 요약",
            "follow_up_questions": ["사업계획서는 어떻게 작성해야 해?", "신청 기간은 언제까지야?"]
        }
# --- PolicyRAGChatbot 더미 클래스 종료 ---

# ============================================================
# 1) RAG 챗봇 초기화
# ============================================================

# 파일 경로는 사용자의 로컬 환경에 맞게 설정
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", r"C:\Users\user\Desktop\bge-m3-sft")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", r"C:\Users\user\Desktop\policy_faiss.index")
METADATA_JSON_PATH = os.getenv("METADATA_JSON_PATH", r"C:\Users\user\Desktop\metadata.json")

# PolicyRAGChatbot은 챗봇 응답 외에 질문 요약 및 후속 질문 생성 기능도 포함해야 함
# 여기서는 api_key를 사용하지만, PolicyRAGChatbot의 answer 메서드가
# API 스펙에 필요한 모든 데이터를 반환하도록 수정해야 합니다.

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
        result = chatbot.answer(request.query)
        
        # 2) 결과 파싱 및 응답 모델에 맞게 데이터 변환
        # result['sources']가 API의 SourceData 상세 구조(title, source, url, snippet)와
        # 일치한다고 가정하고 SourceItem 리스트로 변환합니다.
        
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
            queryTitle=result.get('query_title', request.query),  # RAG 결과에서 요약을 가져옵니다.
            botResponse=result.get('answer', '죄송합니다. 답변을 생성하는 데 실패했습니다.'), # RAG 결과에서 최종 답변을 가져옵니다.
            followUpQuestions=result.get('follow_up_questions') # RAG 결과에서 후속 질문을 가져옵니다.
        )
        
    except Exception as e:
        # 챗봇 처리 중 발생한 예외 처리
        print(f"RAG Chatbot Error: {e}")
        raise HTTPException(
            status_code=500,
            detail="챗봇 답변 생성 중 오류가 발생했습니다."
        )


# -------------------------------------------------------------
# 기존 채팅방 관련 API는 제거되었습니다.
# -------------------------------------------------------------