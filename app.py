import os
import json
import faiss
import numpy as np
import openai
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List, Optional

# 🔐 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 📁 경로 설정
INDEX_PATH = "./faiss_index.index"
METADATA_PATH = "./metadata.json"

# 🤖 모델 로딩
model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, encoding='utf-8') as f:
    metadata = json.load(f)

# 🚀 FastAPI 인스턴스 생성
app = FastAPI()

# 🌐 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 데이터 모델 정의
class UserInfo(BaseModel):
    loginId: str
    nickname: str

class ChatHistory(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    user: UserInfo
    query: str
    sessionHistory: Optional[List[ChatHistory]] = []

class SourceData(BaseModel):
    title: str
    source: str
    url: str
    snippet: str

class ChatResponse(BaseModel):
    sourceData: List[SourceData]
    botResponse: str
    followUpQuestions: Optional[List[str]] = []

# 🧪 상태 확인
@app.get("/")
def root():
    return {"message": "✅ 정책 챗봇 API 서버가 실행 중입니다."}

# 📮 메인 질문 API
@app.post("/api/v1/query", response_model=ChatResponse)
async def query_endpoint(req: ChatRequest):
    query = req.query
    q_vec = model.encode(query).astype("float32").reshape(1, -1)

    # 🔍 벡터 검색
    k = 3
    distances, indices = faiss_index.search(q_vec, k)

    # 📚 관련 문서 수집
    source_data = []
    retrieved_chunks = []
    seen_titles = set()

    for idx in indices[0]:
        doc = metadata[idx]
        title = doc.get("title", "알 수 없음")
        source = doc.get("source", "정책자료")
        url = doc.get("url", "")
        snippet = doc.get("text", "")[:200]  # 앞부분만 보여줌
        if title not in seen_titles:
            source_data.append(SourceData(title=title, source=source, url=url, snippet=snippet))
            seen_titles.add(title)
        retrieved_chunks.append(doc.get("text", ""))

    # 🧠 GPT 요청 구성
    context_text = "\n\n".join(retrieved_chunks)
    gpt_messages = [
        {"role": "system", "content": "다음 문서를 참고하여 질문에 정확하게 답해주세요. 마지막에는 추천 후속 질문 2개를 제시해주세요."}
    ]
    if req.sessionHistory:
        gpt_messages.extend([sh.dict() for sh in req.sessionHistory])
    gpt_messages.append({
        "role": "user",
        "content": f"문서:\n{context_text}\n\n질문: {query}"
    })

    # 🎯 GPT 호출
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=gpt_messages
        )
        full_response = response.choices[0].message.content.strip()

        # 후속 질문 추출
        if "후속 질문" in full_response:
            bot_text, *follow = full_response.split("후속 질문")
            follow_questions = [q.strip("-• \n") for q in follow[0].split("\n") if q.strip()]
        else:
            bot_text = full_response
            follow_questions = []

    except Exception as e:
        bot_text = f"❌ GPT 오류: {str(e)}"
        follow_questions = []

    return ChatResponse(
        sourceData=source_data,
        botResponse=bot_text,
        followUpQuestions=follow_questions
    )
