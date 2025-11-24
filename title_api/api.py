import os
from fastapi import APIRouter, HTTPException, status
from openai import OpenAI
from title_api.model import QuestionRequest, TitleResponse, BadRequestErrorResponse

# --- 1. API 클라이언트 및 라우터 초기화 ---
# 클라이언트는 서버 시작 시 한 번만 초기화
TITLE_GENERATION_CLIENT = None

# APIRouter 인스턴스 생성
router = APIRouter(
    prefix="/generate_title",
    tags=["GPT Title Generation"]
)

def initialize_title_client():
    """OpenAI 클라이언트를 초기화합니다."""
    global TITLE_GENERATION_CLIENT
    if TITLE_GENERATION_CLIENT is None:
        try:
            # 환경 변수에서 API 키를 가져와 클라이언트 초기화
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            TITLE_GENERATION_CLIENT = OpenAI(api_key=api_key)
            print("✅ GPT 제목 생성 클라이언트 로드 완료.")
        except Exception as e:
            print(f"❌ GPT 클라이언트 로드 실패: {e}")
            # 초기화 실패 시 클라이언트 객체는 None으로 남겨둠
            TITLE_GENERATION_CLIENT = None
            raise RuntimeError(f"GPT 클라이언트 초기화 오류: {e}")


# --- 2. GPT 제목 생성 핵심 함수 ---
def generate_chat_title(user_question: str) -> str:
    """GPT-4o-mini를 사용하여 사용자 질문에 대한 채팅방 제목을 생성합니다."""
    global TITLE_GENERATION_CLIENT
    
    if TITLE_GENERATION_CLIENT is None:
        # 클라이언트가 로드되지 않은 경우 오류 반환
        return "제목 생성 오류: 클라이언트 초기화 실패"

    # Few-Shot Prompt Messages 구성 (이전과 동일)
    messages = [
        # SYSTEM: 모델의 역할, 규칙 정의
        {
            "role": "system", 
            "content": "당신은 소상공인 지원사업 FAQ 챗봇의 제목 생성 전문가입니다. 사용자의 긴 질문을 읽고, 질문의 핵심 주제를 추출하여 채팅방 제목을 만들어야 합니다.\n\n규칙:\n1. 제목은 4~8단어 이내로 작성합니다.\n2. 제목은 지원사업, 정책, 대출, 자금, 신청, 조건 등 핵심 키워드를 포함해야 합니다.\n3. 제목에 물음표(?)나 마침표(.)는 사용하지 않습니다. 오직 제목 텍스트만 출력합니다."
        },
        # Few-Shot Example 1
        {
            "role": "user", 
            "content": "코로나19로 매출이 줄었는데, 이번에 새로 나온 소상공인 손실보전금 신청 기간이 언제부터 언제까지인지 알려주세요."
        },
        {
            "role": "assistant", 
            "content": "코로나 손실보전금 신청 기간 안내"
        },
        # Few-Shot Example 2
        {
            "role": "user", 
            "content": "정부에서 지원하는 소상공인 특례보증 대출을 받으려면 어떤 서류를 준비해야 하는지 알고 싶습니다."
        },
        {
            "role": "assistant", 
            "content": "소상공인 특례보증 대출 필요 서류"
        },
        # 사용자 질문
        {
            "role": "user", 
            "content": f"이제 다음 질문에 대한 제목을 생성해 주세요. [질문]: {user_question}"
        }
    ]
    
    try:
        response = TITLE_GENERATION_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=30
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"GPT API 호출 중 오류 발생: {e}")
        # 오류 발생 시 오류 제목 반환
        return "제목 생성 오류"


# --- 3. FastAPI 엔드포인트 정의 ---
@router.post(
    "/", 
    response_model=TitleResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": BadRequestErrorResponse, "description": "필수 파라미터 누락"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "제목 생성 오류"}
    },
    summary="채팅방 제목 생성"
)
async def get_title_endpoint(request_data: QuestionRequest):
    """
    POST 요청을 받아 GPT를 통해 채팅방 제목을 생성하고 반환합니다.
    """
    user_question = request_data.question
    
    # 🚨 400 Bad Request 처리 (question 필드가 비어있을 경우)
    if not user_question or user_question.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "BAD_REQUEST",
                "message": "필수 파라미터가 누락되었습니다.",
                "details": {
                    "missing_fields": ["question"]
                }
            }
        )
    
    # 제목 생성 함수 호출
    generated_title = generate_chat_title(user_question)

    if "오류" in generated_title:
        # GPT 함수에서 오류 제목이 반환되면 HTTP 500 오류 반환
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=generated_title)

    # 성공 응답
    return {"title": generated_title}