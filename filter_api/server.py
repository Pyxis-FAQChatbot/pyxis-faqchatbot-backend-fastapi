from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import platform
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="CleanBot Toxicity API")

# -----------------------------------------------------------
# OS 에 따라 모델 경로 자동 선택
# -----------------------------------------------------------
if platform.system() == "Windows":
    MODEL_PATH = os.getenv("TOXICITY_MODEL_PATH")  # 로컬용
else:
    MODEL_PATH = os.getenv("TOXICITY_MODEL_PATH_SERVER")  # 리눅스 서버용

if not MODEL_PATH:
    raise RuntimeError("❌ Toxicity model path is not set in .env")

# -----------------------------------------------------------
# 모델 로드
# -----------------------------------------------------------
device = "cpu"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    print(f"✅ Loaded CleanBot model from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading CleanBot model: {e}")
    raise e


# -----------------------------------------------------------
# 요청 모델 정의
# -----------------------------------------------------------
class TextRequest(BaseModel):
    text: str


# -----------------------------------------------------------
# API: 유해성 예측
# -----------------------------------------------------------
@app.post("/predict")
def predict(request: TextRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    toxic_prob = float(probs[1])  # label: {0=normal, 1=toxic}

    return {
        "toxic": toxic_prob > 0.5,
        "prob": toxic_prob
    }


# -----------------------------------------------------------
# API: 헬스체크
# -----------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "model": "cleanbot-ready"
    }
