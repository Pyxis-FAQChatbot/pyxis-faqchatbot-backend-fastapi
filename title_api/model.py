from typing import List
from pydantic import BaseModel, Field

# ============================================================
# 1) Request Model
# ============================================================
class QuestionRequest(BaseModel):
    """제목 생성을 요청하는 사용자 질문"""
    question: str = Field(..., description="사용자가 입력한 질문 원문")

# ============================================================
# 2) Response Model
# ============================================================
class TitleResponse(BaseModel):
    """제목 생성 성공 응답 모델"""
    title: str = Field(..., description="GPT 모델이 생성한 채팅방 제목")

# ============================================================
# 3) Error Response Models (클린봇 API 오류 구조 재사용)
# ============================================================

# 400 Bad Request 상세 구조
class TitleErrorDetail(BaseModel):
    missing_fields: List[str] = Field(..., description="누락된 필수 필드 목록")

class BadRequestErrorResponse(BaseModel):
    """400 Bad Request 응답 모델"""
    error: str = "BAD_REQUEST"
    message: str = "필수 파라미터가 누락되었습니다."
    details: TitleErrorDetail