#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정책 지원 안내 RAG 챗봇
- FAISS 인덱스 기반 검색
- OpenAI GPT-4o를 활용한 응답 생성
- 파인튜닝된 BAAI/bge-m3 임베딩 모델 사용
"""

import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional # Optional import 추가
from openai import OpenAI, APIError # APIError import 추가
from transformers import AutoModel, AutoTokenizer
import torch

# ============================================================
# 1. 경로 설정 (🚨 Path 객체 대신 순수 문자열로 수정됨 🚨)
# ============================================================

# 파일 경로
FINETUNED_MODEL_PATH = "C:\\Users\\user\\Desktop\\bge-m3-sft"
FAISS_INDEX_PATH = "C:\\Users\\user\\Desktop\\policy_faiss.index"
METADATA_PATH = "C:\\Users\\user\\Desktop\\metadata.json"

# OpenAI API 키 (환경변수에서 로드)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("⚠️  경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    print("다음 중 하나를 선택하세요:")
    print("1. 터미널에서: export OPENAI_API_KEY='your-api-key'")
    print("2. 코드에서 직접: OPENAI_API_KEY = 'your-api-key'")
    # 또는 여기에 직접 입력 (보안상 권장하지 않음)
    # OPENAI_API_KEY = "your-api-key-here"

print("="*70)
print("🤖 정책 지원 안내 RAG 챗봇")
print("="*70)
# 출력 시에는 Path 객체의 .exists() 대신 os.path.exists를 사용하도록 수정 필요
print(f"📂 모델 경로: {FINETUNED_MODEL_PATH}")
print(f"📂 FAISS 인덱스: {FAISS_INDEX_PATH}")
print(f"📂 메타데이터: {METADATA_PATH}")
print("="*70 + "\n")


# ============================================================
# 2. 파인튜닝된 임베딩 모델 로더 클래스
# ============================================================

class FineTunedEmbedder:
    """파인튜닝된 BAAI/bge-m3 임베딩 모델"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Args:
            model_path: 파인튜닝된 모델 경로
            device: 'cuda' 또는 'cpu' (None이면 자동 감지)
        """
        print("📦 임베딩 모델 로딩 중...")
        
        # 디바이스 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"  - 디바이스: {self.device}")
        
        # 토크나이저 및 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True) # local_file_only 오타 수정
        self.model.to(self.device)
        self.model.eval()
        
        print(f"  ✅ 모델 로드 완료: {model_path}\n")
    
    def encode(self, texts: List[str], batch_size: int = 32, max_length: int = 512) -> np.ndarray:
        """
        텍스트를 임베딩 벡터로 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기
            max_length: 최대 토큰 길이
            
        Returns:
            numpy array (n_texts, embedding_dim)
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 토큰화
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                # GPU로 이동
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # 임베딩 생성
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # CLS 토큰 추출 및 정규화
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
                
                # CPU로 이동 및 numpy 변환
                embeddings.append(cls_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)


# ============================================================
# 3. FAISS 검색기 클래스
# ============================================================

class FAISSRetriever:
    """FAISS 인덱스 기반 문서 검색기"""
    
    def __init__(self, index_path: str, metadata_path: str, embedder: FineTunedEmbedder):
        """
        Args:
            index_path: FAISS 인덱스 파일 경로
            metadata_path: 메타데이터 JSON 파일 경로
            embedder: 임베딩 모델 인스턴스
        """
        print("📚 FAISS 인덱스 로딩 중...")
        
        # FAISS 인덱스 로드
        self.index = faiss.read_index(str(index_path))
        print(f"  - 인덱스 크기: {self.index.ntotal:,}개 문서")
        
        # 메타데이터 로드
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"  - 메타데이터: {len(self.metadata):,}개 항목")
        
        # 임베딩 모델
        self.embedder = embedder
        
        # GPU 사용 가능 시 FAISS 인덱스를 GPU로 이동
        if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
            print("  - FAISS GPU 모드 활성화")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        print("  ✅ FAISS 검색기 준비 완료\n")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리에 대한 top-k 문서 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 개수
            
        Returns:
            검색 결과 리스트 (각 항목은 metadata + score)
        """
        # 쿼리 임베딩
        query_embedding = self.embedder.encode([query])

        query_embedding = query_embedding.astype('float32')

        print(f"  🔍 쿼리 벡터 차원: {query_embedding.shape[1]}")
        print(f"  🔍 쿼리 벡터 Dtype: {query_embedding.dtype}")

        # FAISS 검색 (L2 거리)
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 결과 구성
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity']=float(dist)
                result['score']=float(dist)
                results.append(result)
        
        return results


# ============================================================
# 4. RAG 프롬프트 생성기
# ============================================================

def create_rag_prompt(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    검색된 문서와 쿼리를 결합하여 LLM 프롬프트 생성
    """
    # 검색된 문서 컨텍스트 구성
    context_parts = []
    MAX_CONTENT_LENGTH = 3000

    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[문서 {i}]")
        context_parts.append(f"제목: {doc.get('title', '제목 없음')}")
        
        full_content = doc.get('content', doc.get('text', '내용없음'))
        truncated_content = full_content[:MAX_CONTENT_LENGTH] + ("..." if len(full_content) > MAX_CONTENT_LENGTH else "")
        context_parts.append(f"내용: {truncated_content}")
        
        # 출처 정보를 LLM 프롬프트에서 제외합니다.
        # if 'source' in doc:
        #     context_parts.append(f"출처: {doc['source']}")
        context_parts.append("") # 빈 줄
    
    context = "\n".join(context_parts)
    
    # 🚨🚨 최종 프롬프트 수정: JSON 출력 및 동적 질문 생성 지시 🚨🚨
    system_instruction = f"""
    너는 정책이나 법률 용어를 어려워하는 친구에게 쉽게 설명해주는 친절한 정책 전문 조언자야.
    
    너의 최종 목표는 사용자의 질문에 답변하고, 관련된 후속 질문 4개를 생성하며, 답변에 필요한 정보를 JSON 형식으로 반환하는 거야.
    
    --- LLM 행동 및 응답 규칙 ---
    1. **응답 형식:** 응답은 반드시 **다음 JSON 스키마**를 따라야 해.
       {{
         "botResponse": "생성된 최종 답변 텍스트",
         "queryTitle": "사용자 질문에 대한 요약 제목",
         "followUpQuestions": ["후속 질문 1", "후속 질문 2", "후속 질문 3", "후속 질문 4"]
       }}
    2. **글쓰기 스타일:** 'botResponse' 필드에 들어갈 답변은 반드시 항목 제목(`지원 대상:`, `신청 방법:`)을 사용하지 않고, 모든 정보를 하나의 자연스러운 글로 녹여내야 해. (친근한 편지나 메시지 형태)
    3. **페르소나와 가독성:** 'botResponse'는 '친구처럼', '선배처럼' 친근하게 작성하고, 문단 구분을 명확히 하고, 문장이 너무 길어지지 않도록 주의하여 가독성을 최우선으로 해.
    4. **본문 URL 금지:** 'botResponse' 본문(실행력 키우기 이전)에는 **절대로** URL, 웹사이트 주소, 문서 제목, 출처 등의 정보를 **포함하지 마세요.**
    5. **실행력 키우기:** 'botResponse'의 마지막에는 소상공인 친구가 바로 움직일 수 있도록, 가장 빠르고 구체적인 다음 행동 단계를 딱 하나만 콕 집어 제시해 줘. 이 단계에는 **가장 적합한 공고문 URL을 하이퍼링크 형식([공고문 바로가기](URL))으로 깔끔하게 첨부**해야 해.
    6. **후속 질문:** 'followUpQuestions' 필드에 **답변 내용에 기반한, 사용자가 궁금해 할 만한 4개의 새로운 질문**을 생성해.
    7. **문서 기반:** 제공된 문서에 내용이 없으면 "제공된 정보만으로는 답변하기 어려워. 더 찾아보자!"라고 응답하고, 이 경우 'followUpQuestions'에는 임의의 4개 질문을 넣어.
    
    --- 끝 ---
    """
    
    # 🚨🚨 사용자 콘텐츠 (User Content) 정의 🚨🚨
    user_content = f"""
    --- 답변 참고 자료 ---
    {context}
    
    사용자 질문: {query}
    
    답변:
    """
    
    # SYSTEM 역할과 USER 역할의 메시지를 결합하여 반환 (API 호출 시 분리됨)
    return f"{system_instruction.strip()}\n{user_content.strip()}"


# ============================================================
# 5. OpenAI GPT-4o 응답 생성기 (JSON 파싱 로직 포함)
# ============================================================

class GPT4oGenerator:
    """OpenAI GPT-4o 기반 응답 생성기"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        # ... (중략)
        self.client = OpenAI(api_key=api_key)
        self.model = model
        print(f"  - 모델: {model}")
        print("  ✅ 준비 완료\n")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1500, # JSON 출력을 위해 토큰 여유를 둡니다.
        stream: bool = False
    ) -> Optional[Dict[str, Any]]: # 반환 타입을 딕셔너리로 변경
        """
        프롬프트를 기반으로 응답 생성 (JSON 형식 강제)
        """
        if stream:
            # 스트리밍 모드는 JSON 출력과 호환되지 않아 여기서는 비활성화합니다.
            print("❌ 스트리밍 모드는 JSON 응답 형식과 동시에 사용하기 어렵습니다.")
            return None 

        try:
            # 1. 프롬프트 분리: 시스템 지침과 사용자 콘텐츠로 나눕니다.
            if "--- 답변 참고 자료 ---" in prompt:
                system_content = prompt.split("--- 답변 참고 자료 ---")[0].strip()
                user_content = prompt.split("--- 답변 참고 자료 ---")[1].strip()
                
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            # 2. API 호출 및 JSON 형식 강제
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} # 👈 JSON 형식 강제
            )
            
            json_text = response.choices[0].message.content
            
            # 3. JSON 파싱
            try:
                result_dict = json.loads(json_text)
                return result_dict
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 오류: {e}")
                print(f"받은 텍스트: {json_text[:200]}...")
                return None
                
        except APIError as e:
            print(f"❌ OpenAI API 오류: {e.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"❌ 응답 생성 오류: {e}")
            return None


# ============================================================
# 6. RAG 챗봇 클래스 (전체 파이프라인)
# ============================================================

class PolicyRAGChatbot:
    """정책 지원 안내 RAG 챗봇"""
    
    def __init__(
        self,
        model_path: str,
        index_path: str,
        metadata_path: str,
        api_key: str,
        device: str = None
    ):
        """
        Args:
            model_path: 파인튜닝된 임베딩 모델 경로
            index_path: FAISS 인덱스 파일 경로
            metadata_path: 메타데이터 JSON 파일 경로
            api_key: OpenAI API 키
            device: 디바이스 ('cuda' 또는 'cpu')
        """
        # 1. 임베딩 모델 로드
        self.embedder = FineTunedEmbedder(str(model_path), device)
        
        # 2. FAISS 검색기 로드
        self.retriever = FAISSRetriever(index_path, metadata_path, self.embedder)
        
        # 3. GPT-4o 생성기 초기화
        self.generator = GPT4oGenerator(api_key)
        
        print("="*70)
        print("✅ RAG 챗봇 준비 완료!")
        print("="*70 + "\n")
    
    def answer(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.7,
        stream: bool = False, # JSON 출력 때문에 False로 고정 (주석으로 처리)
        show_sources: bool = True
    ) -> Dict[str, Any]:
        """
        사용자 질문에 대한 답변 생성 (FastAPI API 스펙 준수)
        """
        print(f"💭 질문: {query}\n")
        
        # 1. 관련 문서 검색
        print(f"🔍 관련 문서 검색 중... (top-{top_k})")
        retrieved_docs = self.retriever.search(query, top_k)
        print(f"  ✅ {len(retrieved_docs)}개 문서 검색 완료\n")
        
        # 출처 표시 (백엔드 콘솔 출력용)
        if show_sources:
            print("📚 참고 문서:")
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"  [{i}] {doc.get('title', '제목 없음')} (유사도: {doc['similarity']:.3f})")
            print()
        
        # 2. 프롬프트 생성
        prompt = create_rag_prompt(query, retrieved_docs)
        
        # 3. GPT-4o로 응답 생성 (JSON 응답)
        print("🤖 답변 생성 중...\n")
        print("-" * 70)
        
        # 🚨 JSON 출력을 위해 stream 인수는 무시하고 False로 호출합니다.
        json_result = self.generator.generate(
            prompt,
            temperature=temperature,
            stream=False 
        )
        print("-" * 70 + "\n")
        
        # 에러 처리
        if not json_result:
             # LLM 응답 실패 시 임시 질문 4개와 함께 반환
            return {
                "answer": '죄송합니다. 답변 생성에 실패했습니다. (API 오류)',
                "sources": [],
                "query_title": query,
                "follow_up_questions": [ 
                    "지원금 신청 기간이 궁금해요.",
                    "신청 자격 요건은 무엇인가요?",
                    "다른 지역의 유사 사업이 있나요?",
                    "제출해야 할 서류를 알려주세요."
                ]
            }
        
        # 4. 결과 반환 (🚨 LLM이 반환한 JSON 결과 사용 🚨)
        # ----------------------------------------------------------------------
        
        # 4-1. 출처 데이터(sources) 형식 변환
        final_sources = []
        for doc in retrieved_docs:
            source_path = doc.get('source', '')
            
            # source 변환: 파일 경로에서 파일 이름(확장자 제외)만 추출
            source_name = Path(source_path).stem if source_path else 'N/A'
            
            # snippet 생성: content의 처음 150자만 추출
            content = doc.get('content', doc.get('text', '내용없음'))
            snippet = content[:150].strip() + "..."
            
            final_sources.append({
                "title": doc.get('title', '제목 없음'),
                "source": source_name, 
                "url": doc.get('url', 'N/A'),
                "snippet": snippet
            })

        # 4-2. 최종 API 응답 구성
        # LLM이 생성한 JSON의 필드를 사용하고, sources만 별도로 구성하여 합칩니다.
        final_response = {
            "answer": json_result.get("botResponse", "답변 텍스트가 누락되었습니다."), 
            "sources": final_sources, 
            "query_title": json_result.get("queryTitle", query),
            "follow_up_questions": json_result.get("followUpQuestions", [])
        }
        
        # 4-3. 후속 질문이 4개가 아닌 경우를 대비하여 (LLM 오류 시) 안전 장치
        if len(final_response["follow_up_questions"]) < 4:
             print("⚠️  LLM이 4개의 질문을 모두 생성하지 못했습니다. 질문 개수를 보장할 수 없습니다.")


        return final_response


# ============================================================
# 7. 메인 실행 함수 (main 함수도 JSON 형식으로 출력하도록 수정)
# ============================================================

def main():
    # ... (중략: 경로 및 API 키 확인 로직)

    if not os.path.exists(METADATA_PATH):
        print(f"❌ 메타데이터를 찾을 수 없습니다: {METADATA_PATH}")
        return
    
    try:
        # RAG 챗봇 초기화
        chatbot = PolicyRAGChatbot(
            model_path=FINETUNED_MODEL_PATH,
            index_path=FAISS_INDEX_PATH,
            metadata_path=METADATA_PATH,
            api_key=OPENAI_API_KEY,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 대화형 루프
        print("💬 챗봇이 준비되었습니다. 질문을 입력하세요. (종료: 'quit' 또는 'exit')\n")
        
        while True:
            try:
                # 사용자 입력
                user_input = input("👤 당신: ").strip()
                
                if not user_input:
                    continue
                
                # 종료 명령
                if user_input.lower() in ['quit', 'exit', '종료', '나가기']:
                    print("\n👋 챗봇을 종료합니다. 감사합니다!")
                    break
                
                print()  # 빈 줄
                
                # 답변 생성
                result = chatbot.answer(
                    query=user_input,
                    top_k=5,
                    temperature=0.7,
                    stream=False, # JSON 출력 때문에 False로 고정
                    show_sources=True
                )
                
                # 결과 출력 (JSON 형식으로 출력)
                if result:
                    print("📝 최종 답변:")
                    print(result['answer'])
                    print("\n❓ 후속 질문:")
                    for q in result.get('follow_up_questions', []):
                         print(f" - {q}")
                    
                print("\n" + "="*70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 챗봇을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}\n")
                continue
    
    except Exception as e:
        print(f"❌ 초기화 오류: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# 8. 단일 질문 테스트 함수 (수정 반영)
# ============================================================

def test_single_query(query: str):
    """단일 질문 테스트 (대화형 루프 없이)"""
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return
    
    # RAG 챗봇 초기화
    chatbot = PolicyRAGChatbot(
        model_path=FINETUNED_MODEL_PATH,
        index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH,
        api_key=OPENAI_API_KEY,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 답변 생성
    result = chatbot.answer(
        query=query,
        top_k=5,
        temperature=0.7,
        stream=False,  # 스트리밍 없이
        show_sources=True
    )
    
    # 결과 출력
    if result:
        print("📝 최종 답변:")
        print(result['answer'])
        print("\n❓ 후속 질문:")
        for q in result.get('follow_up_questions', []):
             print(f" - {q}")
        print("\n" + "="*70 + "\n")


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    # 대화형 모드로 실행
    main()
    
    # 또는 단일 질문 테스트 (아래 주석 해제)
    # test_single_query("중소기업을 위한 R&D 지원 사업이 있나요?")