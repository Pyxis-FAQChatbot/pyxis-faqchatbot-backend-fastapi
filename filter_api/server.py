from fastapi import FastAPI
from dotenv import load_dotenv
from filter_api.api import router, initialize_toxicity_model

# Load environment variables
load_dotenv()

app = FastAPI(title="CleanBot Toxicity API")

# 클린봇 모델 초기화
initialize_toxicity_model()

# /api/v1/filter/text 라우터 등록
app.include_router(router)

# 헬스 체크
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "cleanbot"
    }
