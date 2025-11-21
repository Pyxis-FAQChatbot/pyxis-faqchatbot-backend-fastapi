import os
import uuid
import datetime
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

# 🔥 RAG 챗봇 import
from rag_chatbot import PolicyRAGChatbot

# ============================================================
# 1) RAG 챗봇 초기화
# ============================================================

EMBEDDING_MODEL_PATH = r"C:\Users\user\Desktop\bge-m3-sft"
FAISS_INDEX_PATH = r"C:\Users\user\Desktop\policy_faiss.index"
METADATA_JSON_PATH = r"C:\Users\user\Desktop\metadata.json"

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 3) 메모리 기반 저장소 (DB 없이 구현)
# ============================================================

class ChatRoom:
    def __init__(self, title: str):
        self.id = str(uuid.uuid4())
        self.title = title
        self.created_at = datetime.datetime.now().isoformat()
        self.messages = []  # {"id": str, "role": "assistant"|"user", "content": str, "createdAt": str}

chat_rooms: Dict[str, ChatRoom] = {}  # key = chatroom id


# ============================================================
# 4) 요청/응답 모델 정의
# ============================================================

class ChatRoomCreateRequest(BaseModel):
    title: Optional[str] = "제목"


class ChatRoomCreateResponse(BaseModel):
    botChatId: str
    title: str
    createdAt: str


class MessageRequest(BaseModel):
    userQuery: str


class MessageResponse(BaseModel):
    botMessageId: str
    botResponse: str
    sourceData: List[Dict]
    createdAt: str


class ChatItem(BaseModel):
    id: str
    role: str
    content: str
    createdAt: str
    sourceData: Optional[List[Dict]] = None


class MessageListResponse(BaseModel):
    page: int
    size: int
    totalElements: int
    totalPages: int
    items: List[ChatItem]


# ============================================================
# 5) API 구현
# ============================================================

# -----------------------------
# 5-1) 챗봇 채팅방 생성
# -----------------------------
@app.post("/api/v1/chatbot", response_model=ChatRoomCreateResponse)
def create_chatroom(request: ChatRoomCreateRequest):
    try:
        room = ChatRoom(title=request.title)
        chat_rooms[room.id] = room

        return ChatRoomCreateResponse(
            botChatId=room.id,
            title=room.title,
            createdAt=room.created_at
        )
    except:
        raise HTTPException(status_code=400, detail="챗봇 생성에 실패하였습니다.")


# -----------------------------
# 5-2) 챗봇 메시지 생성 (RAG 답변)
# -----------------------------
@app.post("/api/v1/chatbot/{chatbot_id}/message", response_model=MessageResponse)
def send_message(chatbot_id: str, request: MessageRequest):
    if chatbot_id not in chat_rooms:
        raise HTTPException(status_code=400, detail="채팅방이 존재하지 않습니다.")

    room = chat_rooms[chatbot_id]

    # 1) user 메시지 저장
    user_msg = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": request.userQuery,
        "createdAt": datetime.datetime.now().isoformat()
    }
    room.messages.append(user_msg)

    # 2) RAG 호출
    result = chatbot.answer(request.userQuery)
    bot_answer = result['answer']
    sources = result['sources']

    # 3) bot 메시지 저장
    bot_msg_id = str(uuid.uuid4())
    bot_msg = {
        "id": bot_msg_id,
        "role": "assistant",
        "content": bot_answer,
        "sourceData": sources,
        "createdAt": datetime.datetime.now().isoformat()
    }
    room.messages.append(bot_msg)

    return MessageResponse(
        botMessageId=bot_msg_id,
        botResponse=bot_answer,
        sourceData=sources,
        createdAt=bot_msg["createdAt"]
    )


# -----------------------------
# 5-3) 메시지 리스트 조회
# -----------------------------
@app.get("/api/v1/chatbot/{chatbot_id}/message", response_model=MessageListResponse)
def list_messages(chatbot_id: str, page: int = 0, size: int = 20):

    if chatbot_id not in chat_rooms:
        raise HTTPException(status_code=400, detail="채팅방이 존재하지 않습니다.")

    room = chat_rooms[chatbot_id]
    total = len(room.messages)

    start = page * size
    end = start + size

    items = [
        ChatItem(
            id=msg["id"],
            role=msg["role"],
            content=msg["content"],
            createdAt=msg["createdAt"],
            sourceData=msg.get("sourceData")
        )
        for msg in room.messages[start:end]
    ]

    return MessageListResponse(
        page=page,
        size=size,
        totalElements=total,
        totalPages=(total // size) + (1 if total % size else 0),
        items=items
    )


# -----------------------------
# 5-4) 채팅방 삭제
# -----------------------------
@app.delete("/api/v1/chatbot/{chatbot_id}")
def delete_chatbot(chatbot_id: str):
    if chatbot_id not in chat_rooms:
        raise HTTPException(status_code=400, detail="채팅방이 존재하지 않습니다.")

    del chat_rooms[chatbot_id]
    return {"message": "채팅방 삭제 완료"}
