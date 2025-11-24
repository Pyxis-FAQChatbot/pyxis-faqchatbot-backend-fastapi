# filter_api/model.py
# 유해성 필터링 API용 Request/Response Pydantic 모델 정의

from typing import Optional
from pydantic import BaseModel, Field


# ============================================================
# Request Models
# ============================================================

class UserInfo(BaseModel):
    """사용자 식별 정보"""
    loginId: str = Field(..., description="사용자 로그인 ID")
    nickname: str = Field(..., description="사용자 닉네임")


class FilterQueryRequest(BaseModel):
    """유해성 필터링 요청 모델"""
    user: UserInfo = Field(..., description="사용자 식별 정보")
    query: str = Field(..., description="필터링할 텍스트 (사용자 질문)")


# ============================================================
# Response Models
# ============================================================

class FilterResponse(BaseModel):
    """유해성 필터링 응답 모델"""
    originQuery: str = Field(..., description="원본 질문 텍스트")
    toxicity: float = Field(..., description="전반적 유해성 점수 (0.0~1.0)")
    insult: float = Field(..., description="모욕 점수 (0.0~1.0)")
    profanity: float = Field(..., description="욕설 점수 (0.0~1.0)")
    hate: float = Field(..., description="혐오 발언 점수 (0.0~1.0)")
    threat: float = Field(..., description="위협 점수 (0.0~1.0)")
    blocked: bool = Field(..., description="차단 여부 (toxicity >= 0.7이면 True)")


# ============================================================
# Error Response Models
# ============================================================

class ErrorDetail(BaseModel):
    """오류 상세 정보"""
    missing_fields: Optional[list] = Field(None, description="누락된 필드 목록")


class ErrorResponse(BaseModel):
    """오류 응답 모델"""
    error: str = Field(..., description="오류 코드")
    message: str = Field(..., description="오류 메시지")
    details: Optional[ErrorDetail] = Field(None, description="오류 상세 정보")


class ToxicContentErrorResponse(BaseModel):
    """유해 콘텐츠 차단 응답 모델 (406 응답용)"""
    error: str = Field(..., description="오류 코드: TOXIC_CONTENT_DETECTED")
    message: str = Field(..., description="사용자에게 표시될 메시지")
    toxicity_score: float = Field(..., description="유해성 점수")