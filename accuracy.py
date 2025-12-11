import os
import sys
import time
import json
import numpy as np
from dotenv import load_dotenv

# --------------------------
# 0. 환경 설정 및 경로 세팅
# --------------------------
load_dotenv()

# rag_api 모듈 import를 위해 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "rag_api"))

from rag_api.rag_chatbot import FineTunedEmbedder, FAISSRetriever
from openai import OpenAI

# 필수 환경변수
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
METADATA_JSON_PATH = os.getenv("METADATA_JSON_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EVAL_FILE = "eval_dataset.jsonl"
TOP_K = 5

if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY 가 설정되지 않았습니다. .env를 확인해 주세요.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

print("\n======================================================================")
print("🤖 PYXIS RAG CHATBOT — ENTERPRISE LEVEL PERFORMANCE REPORT")
print("======================================================================")
print(f"📂 임베딩 모델 경로: {EMBEDDING_MODEL_PATH}")
print(f"📂 FAISS 인덱스 경로: {FAISS_INDEX_PATH}")
print(f"📂 메타데이터 경로: {METADATA_JSON_PATH}")
print("======================================================================\n")


# --------------------------
# 1. 임베딩 & FAISS 로딩
# --------------------------
embedder = FineTunedEmbedder(model_path=EMBEDDING_MODEL_PATH)
retriever = FAISSRetriever(
    index_path=FAISS_INDEX_PATH,
    metadata_path=METADATA_JSON_PATH,
    embedder=embedder
)


# --------------------------
# 2. 평가용 컨텍스트 & 답변 생성 로직
#    (서비스용 프롬프트와 완전히 분리!)
# --------------------------
def build_eval_context(docs, max_len_chars: int = 1500) -> str:
    """평가용으로 문서 컨텍스트를 단순하게 텍스트로 구성"""
    parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.get("title", "제목 없음")
        content = doc.get("content", doc.get("text", "")) or ""
        snippet = content[:max_len_chars]
        parts.append(f"[문서 {i}] 제목: {title}\n내용: {snippet}")
    return "\n\n".join(parts)


def generate_eval_answer(question: str, docs) -> str:
    """
    평가 전용 LLM 호출.
    - 이모지, 링크, 조언 없이
    - 딱 한 문장으로만 정책 핵심을 말하게 함
    """
    context = build_eval_context(docs)
    if not context.strip():
        return "제공된 문서만으로는 답변하기 어렵습니다."

    messages = [
        {
            "role": "system",
            "content": (
                "너는 대한민국 중소기업 지원정책을 잘 아는 전문 상담사야. "
                "반드시 제공된 문서 내용만 사용해서 사실만 정확하게 말해. "
                "답변은 딱 한 문장, 존댓말, 불필요한 설명·이모지·링크·목록 없이 "
                "정책의 핵심 정보(지원 대상, 지원 내용, 기간 등)만 말해."
            ),
        },
        {
            "role": "user",
            "content": (
                "다음 문서를 참고해서 질문에 대한 정답을 한 문장으로만 써 주세요.\n\n"
                f"질문: {question}\n\n"
                f"[문서]\n{context}"
            ),
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


# --------------------------
# 3. Retrieval Hit Rate@K
# --------------------------
def compute_hit_rate() -> float:
    total, hit = 0, 0
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q = item["question"]
            gold = item["gold_doc_id"]

            res = retriever.search(q, top_k=TOP_K)
            retrieved_ids = [r["id"] for r in res]

            if gold in retrieved_ids:
                hit += 1
            total += 1
    return hit / total if total > 0 else 0.0


# --------------------------
# 4. GPT-Judge Answer Accuracy
# --------------------------
def gpt_judge_accuracy() -> float:
    total, correct = 0, 0

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q = item["question"]
            gold = item["gold_answer"]

            # 1) 검색 & 평가용 답변 생성
            retrieved = retriever.search(q, top_k=TOP_K)
            bot_answer = generate_eval_answer(q, retrieved)

            # 2) GPT에게 "정답/오답" 판정 요청
            judge_messages = [
                {
                    "role": "system",
                    "content": (
                        "너는 채점관이야. 사용자의 질문에 대해 "
                        "모델 답변이 GOLD 정답과 의미적으로 같은지 평가해. "
                        "뜻이 거의 같으면 1, 다르면 0만 숫자로 출력해."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"질문: {q}\n\n"
                        f"GOLD 정답: {gold}\n\n"
                        f"모델 답변: {bot_answer}\n\n"
                        "정답이면 1, 오답이면 0만 출력해."
                    ),
                },
            ]

            judgment = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=judge_messages,
                temperature=0.0,
                max_tokens=4,
            )

            content = judgment.choices[0].message.content.strip()
            try:
                score = int(content[0])
            except Exception:
                score = 0

            correct += 1 if score == 1 else 0
            total += 1

    return correct / total if total > 0 else 0.0


# --------------------------
# 5. Hallucination Rate
#    (문서 근거 밖의 정보 비율)
# --------------------------
def hallucination_rate() -> float:
    total, hallucinated = 0, 0

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q = item["question"]

            retrieved = retriever.search(q, top_k=TOP_K)
            bot_answer = generate_eval_answer(q, retrieved)
            context = build_eval_context(retrieved)

            judge_messages = [
                {
                    "role": "system",
                    "content": (
                        "너는 사실 검증 전문가야. 모델 답변이 아래 문서 내용에 근거했는지 평가해. "
                        "문서에 없는 내용이나 왜곡된 사실이 들어가면 1, "
                        "문서에 있는 내용으로만 답했으면 0만 숫자로 출력해."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"[문서]\n{context}\n\n"
                        f"[모델 답변]\n{bot_answer}\n\n"
                        "근거 없는 내용(환각)이 있으면 1, 없으면 0만 출력해."
                    ),
                },
            ]

            judgment = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=judge_messages,
                temperature=0.0,
                max_tokens=4,
            )

            content = judgment.choices[0].message.content.strip()
            try:
                score = int(content[0])
            except Exception:
                score = 1  # 애매하면 환각으로 처리

            hallucinated += 1 if score == 1 else 0
            total += 1

    return hallucinated / total if total > 0 else 0.0


# --------------------------
# 6. Latency (응답 속도)
# --------------------------
def response_latency():
    times = []

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q = item["question"]

            start = time.time()
            retrieved = retriever.search(q, top_k=TOP_K)
            _ = generate_eval_answer(q, retrieved)
            end = time.time()

            times.append(end - start)

    if not times:
        return 0.0, 0.0

    avg = float(np.mean(times))
    p95 = float(np.percentile(times, 95))
    return avg, p95


# --------------------------
# 7. Index Coverage
# --------------------------
def index_coverage() -> int:
    return len(retriever.metadata)


# --------------------------
# 8. 메트릭 전체 실행
# --------------------------
if __name__ == "__main__":
    hit_rate = compute_hit_rate()
    acc = gpt_judge_accuracy()
    halluc = hallucination_rate()
    avg_lat, p95_lat = response_latency()
    coverage = index_coverage()

    print("📌 ENTERPRISE PERFORMANCE REPORT\n")
    print(f"🔍 Retrieval Hit Rate@{TOP_K}: {hit_rate:.3f}")
    print(f"🧠 GPT-Judge Answer Accuracy: {acc:.3f}")
    print(f"⚠️ Hallucination Rate: {halluc:.3f}")
    print(f"⚡ Average Latency: {avg_lat:.3f} sec")
    print(f"⏱️  P95 Latency: {p95_lat:.3f} sec")
    print(f"📚 Index Coverage: {coverage} documents")

    print("\n======================================================================")
    print("📈 Pyxis 챗봇 — Enterprise-grade AI Performance Evaluation Completed")
    print("======================================================================\n")
