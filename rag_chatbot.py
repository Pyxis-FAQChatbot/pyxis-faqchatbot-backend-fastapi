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
from typing import List, Dict, Any
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer
import torch

# ============================================================
# 1. 경로 설정 (🚨 Path 객체 대신 순수 문자열로 수정됨 🚨)
# ============================================================

# 바탕화면 경로 자동 감지 (사용하지 않으므로 주석 처리하거나 제거)
# DESKTOP_PATH = Path.home() / "Desktop"

# 파일 경로 (Path 객체에서 문자열로 변경)
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
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
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
    # 검색된 문서 컨텍스트 구성 (이전 코드와 동일)
    context_parts = []
    MAX_CONTENT_LENGTH = 3000

    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[문서 {i}]")
        context_parts.append(f"제목: {doc.get('title', '제목 없음')}")
        
        full_content = doc.get('content', doc.get('text', '내용없음'))
        truncated_content = full_content[:MAX_CONTENT_LENGTH] + ("..." if len(full_content) > MAX_CONTENT_LENGTH else "")
        context_parts.append(f"내용: {truncated_content}")
        
        if 'source' in doc:
            context_parts.append(f"출처: {doc['source']}")
        context_parts.append("") # 빈 줄
    
    context = "\n".join(context_parts)
    
    # 🚨🚨 최종 프롬프트 수정: 페르소나 및 스타일 적용 🚨🚨
    system_instruction = f"""
    너는 정책이나 법률 용어를 어려워하는 친구에게 쉽게 설명해주는 친절한 정책 전문 조언자야.
    
    --- LLM 행동 및 응답 규칙 ---
    1. **글쓰기 스타일:** **답변은 반드시 항목 제목(`지원 대상:`, `신청 방법:`, `문의처:`)을 사용하지 않고, 모든 정보를 하나의 자연스러운 글로 녹여내야 해.** (친근한 편지나 메시지 형태)
    2. **페르소나와 말투:** 답변 전체에서 '친구처럼', '선배처럼' 친근하고 쉽게 말해줘.
    3. **용어 설명:** '세무', '회계', '법률', '융자', '체납' 등 어려운 전문 용어는 (괄호 안에 쉬운 말로 풀어서) 반드시 설명해 줘야 해.
    4. **URL 검색 규칙:** 답변에 웹사이트나 플랫폼 이름이 포함되면, Google 검색을 사용해서 공식 URL을 찾아 하이퍼링크 형식([플랫폼 이름](URL))으로 깔끔하게 첨부해야 해.
    5. **실행력 키우기 (Next Step):** 답변의 마지막에는 소상공인 친구가 바로 움직일 수 있도록, 가장 빠르고 구체적인 다음 행동 단계를 딱 하나만 콕 집어 제시해 줘. '바로 해보자!', '이것부터 시작하자!' 같은 독려하는 말투로 마무리해야 해.
    6. **문서 기반:** 제공된 문서에 내용이 없으면 "제공된 정보만으로는 답변하기 어려워. 더 찾아보자!"라고 말해줘.
    
    --- 끝 ---
    """
    
    # 🚨🚨 사용자 콘텐츠 (User Content) 정의 🚨🚨
    # LLM이 답변 생성에 사용해야 할 자료와 최종 질문을 포함합니다.
    user_content = f"""
    --- 답변 참고 자료 ---
    {context}
    
    사용자 질문: {query}
    
    답변:
    """
    
    # SYSTEM 역할과 USER 역할의 메시지를 결합하여 반환 (API 호출 시 분리됨)
    # LLM이 'SYSTEM' 영역은 출력하지 않고 'USER' 영역에 대한 답변만 하도록 유도합니다.
    return f"{system_instruction.strip()}\n{user_content.strip()}"


# ============================================================
# 5. OpenAI GPT-4o 응답 생성기
# ============================================================

class GPT4oGenerator:
    """OpenAI GPT-4o 기반 응답 생성기"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Args:
            api_key: OpenAI API 키
            model: 사용할 모델명 (기본: gpt-4o)
        """
        print("🤖 OpenAI 클라이언트 초기화 중...")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        print(f"  - 모델: {model}")
        print("  ✅ 준비 완료\n")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> str:
        """
        프롬프트를 기반으로 응답 생성 (시스템 역할 분리 적용)
        """
        try:
            # 1. 프롬프트 분리: 시스템 지침과 사용자 콘텐츠로 나눕니다.
            if "--- 답변 참고 자료 ---" in prompt:
                # 규칙 및 페르소나 (시스템 메시지)
                system_content = prompt.split("--- 답변 참고 자료 ---")[0].strip()
                # 참고 자료 및 사용자 질문 (사용자 메시지)
                user_content = prompt.split("--- 답변 참고 자료 ---")[1].strip()
                
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            else:
                # 안전 장치: 분리가 안 될 경우 기존대로 user content에 통합
                messages = [{"role": "user", "content": prompt}]

            # 2. API 호출
            if stream:
                # 스트리밍 모드
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages, # 👈 분리된 메시지 사용
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                )
                
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_response += content
                print() # 줄바꿈
                return full_response
            else:
                # 일반 모드
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages, # 👈 분리된 메시지 사용
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
                
        except Exception as e:
            print(f"❌ OpenAI API 오류: {e}")
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
        stream: bool = False,
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
        
        # 출처 표시
        if show_sources:
            print("📚 참고 문서:")
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"  [{i}] {doc.get('title', '제목 없음')} (유사도: {doc['similarity']:.3f})")
            print()
        
        # 2. 프롬프트 생성
        prompt = create_rag_prompt(query, retrieved_docs)
        
        # 3. GPT-4o로 응답 생성
        print("🤖 답변 생성 중...\n")
        print("-" * 70)
        answer = self.generator.generate(
            prompt,
            temperature=temperature,
            stream=stream
        )
        print("-" * 70 + "\n")
        
        # 4. 결과 반환 (🚨 최종 수정된 부분 🚨)
        # ----------------------------------------------------------------------
        # 4. 결과 반환 (API 스펙에 맞게 데이터 변환 및 형식 맞추기)
        # ----------------------------------------------------------------------
        
        # 4-1. 출처 데이터(sources) 형식 변환
        final_sources = []
        for doc in retrieved_docs:
            source_path = doc.get('source', '')
            
            # source 변환: 파일 경로에서 파일 이름(확장자 제외)만 추출
            source_name = Path(source_path).stem
            
            # snippet 생성: content의 처음 150자만 추출
            content = doc.get('content', doc.get('text', '내용없음'))
            snippet = content[:150].strip() + "..."
            
            final_sources.append({
                "title": doc.get('title', '제목 없음'),
                "source": source_name, 
                "url": doc.get('url', 'N/A'), # metadata에 url이 없다면 N/A
                "snippet": snippet
            })

        # 4-2. LLM이 생성하지 않는 query_title 및 follow_up_questions에 임시 값 할당
        # (실제 구현 시 LLM에게 JSON으로 요청하여 추출해야 함)
        query_title = f"질문 요약: {query[:20]}..."
        follow_up_questions = ["신청 자격 요건은 무엇인가요?", "다른 관련 사업도 찾아볼 수 있나요?"]

        return {
            "answer": answer, # 👈 LLM이 생성한 최종 답변 텍스트
            "sources": final_sources, # 👈 API 형식에 맞게 변환된 출처 리스트
            "query_title": query_title, # 👈 임시로 생성된 질문 요약
            "follow_up_questions": follow_up_questions # 👈 임시로 생성된 후속 질문
        }


# ============================================================
# 7. 메인 실행 함수
# ============================================================

def main():
    """메인 실행 함수"""
    
    # API 키 확인
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return
    
    
    if not os.path.exists(FAISS_INDEX_PATH): # Path.exists() 대신 os.path.exists() 사용
        print(f"❌ FAISS 인덱스를 찾을 수 없습니다: {FAISS_INDEX_PATH}")
        return
    
    if not os.path.exists(METADATA_PATH): # Path.exists() 대신 os.path.exists() 사용
        print(f"❌ 메타데이터를 찾을 수 없습니다: {METADATA_PATH}")
        return
    
    try:
        # RAG 챗봇 초기화
        chatbot = PolicyRAGChatbot(
            model_path=str(FINETUNED_MODEL_PATH),
            index_path=str(FAISS_INDEX_PATH),
            metadata_path=str(METADATA_PATH),
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
                    stream=True,  # 스트리밍 출력
                    show_sources=True
                )
                
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
# 8. 단일 질문 테스트 함수
# ============================================================

def test_single_query(query: str):
    """단일 질문 테스트 (대화형 루프 없이)"""
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return
    
    # RAG 챗봇 초기화
    chatbot = PolicyRAGChatbot(
        model_path=str(FINETUNED_MODEL_PATH),
        index_path=str(FAISS_INDEX_PATH),
        metadata_path=str(METADATA_PATH),
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
    print("📝 최종 답변:")
    print(result['answer'])


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    # 대화형 모드로 실행
    main()
    
    # 또는 단일 질문 테스트 (아래 주석 해제)
    # test_single_query("중소기업을 위한 R&D 지원 사업이 있나요?")