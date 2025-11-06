import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 🔧 설정
DATA_DIR = "./bizinfo_data"
INDEX_PATH = "./faiss_index.index"
METADATA_PATH = "./metadata.json"

# ✅ 모델 로딩
model = SentenceTransformer("all-MiniLM-L6-v2")

# 📂 모든 extracted.txt 파일 찾기
chunks = []
metadata = []

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file == "extracted.txt":
            file_path = os.path.join(root, file)
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            # 문단 단위로 나누기
            title = os.path.basename(os.path.dirname(file_path))
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for p in paragraphs:
                chunks.append(p)
                metadata.append({
                    "title": title,
                    "source": "로컬",
                    "url": "",
                    "text": p
                })

# 📌 임베딩
embeddings = model.encode(chunks).astype("float32")

# 🔍 FAISS 인덱스 생성
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)

# 💾 메타데이터 저장
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"✅ 총 {len(chunks)} 개의 문단이 인덱싱되었습니다.")
