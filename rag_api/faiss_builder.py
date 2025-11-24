#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
비즈인포 정책 데이터 FAISS 벡터 인덱스 구축 스크립트
- 각 정책 폴더의 extracted.txt를 읽어 임베딩 생성
- BAAI/bge-m3 모델 사용 (파인튜닝 버전)
- FAISS 인덱스 및 메타데이터 JSON 저장
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
from tqdm import tqdm

class PolicyVectorBuilder:
    def __init__(self, base_path: str, model_name: str = "BAAI/bge-m3"):
        """
        Args:
            base_path: bizinfo_data 폴더 경로
            model_name: 사용할 임베딩 모델
        """
        self.base_path = Path(base_path).expanduser()
        self.model_name = model_name
        
        # GPU 사용 설정
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"🟢 GPU 사용: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("⚠️ GPU 사용 불가 → CPU로 실행됩니다.")

        print(f"🔧 디바이스: {self.device}")
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"📂 데이터 경로: {self.base_path}")
        
        # 모델 로드
        print(f"📥 모델 로딩 중: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        
        # GPU 최적화 설정
        if hasattr(torch.cuda, 'amp'):
            print("⚡ Mixed Precision (FP16) 활성화")
            self.use_amp = True
        else:
            self.use_amp = False
        
        print("✅ 모델 로딩 완료")
        
        self.policies = []
        self.metadata = []
        self.embeddings_cache = {}  # 임베딩 캐시
    
    def load_existing_metadata(self, metadata_path: Path):
        """기존 메타데이터 로드하여 캐시 구축"""
        if metadata_path.exists():
            print(f"📥 기존 메타데이터 로딩: {metadata_path}")
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
                
                # 기존 데이터를 캐시에 저장
                for idx, meta in enumerate(existing_metadata):
                    cache_key = f"{meta['id']}_{meta['title']}"
                    self.embeddings_cache[cache_key] = {
                        'index': idx,
                        'metadata': meta
                    }
                
                print(f"✅ {len(existing_metadata)}개 기존 정책 캐시 로드 완료")
                return existing_metadata
            except Exception as e:
                print(f"⚠️  메타데이터 로딩 실패: {e}")
                return []
        return []
    
    def extract_folder_info(self, folder_name: str) -> Tuple[str, str]:
        """
        폴더명에서 ID와 제목 추출
        예: [PBLN_000000000123456]탄소저감을 위한... 
        -> id: PBLN_000000000123456, title: 탄소저감을 위한...
        """
        pattern = r'\[([^\]]+)\](.+)'
        match = re.match(pattern, folder_name)
        
        if match:
            policy_id = match.group(1)
            title = match.group(2).strip()
            return policy_id, title
        else:
            return None, None
    
    def scan_policies(self, skip_cached=True):
        """모든 정책 폴더를 스캔하여 extracted.txt 파일 찾기"""
        print("\n🔍 정책 폴더 스캔 중...")
        
        if not self.base_path.exists():
            raise FileNotFoundError(f"경로를 찾을 수 없습니다: {self.base_path}")
        
        policy_folders = [d for d in self.base_path.iterdir() if d.is_dir()]
        print(f"📁 총 {len(policy_folders)}개 폴더 발견")
        
        found_count = 0
        skipped_count = 0
        cached_count = 0
        
        for folder in tqdm(policy_folders, desc="폴더 스캔"):
            policy_id, title = self.extract_folder_info(folder.name)
            
            if not policy_id or not title:
                skipped_count += 1
                continue
            
            # 캐시 체크
            cache_key = f"{policy_id}_{title}"
            if skip_cached and cache_key in self.embeddings_cache:
                cached_count += 1
                continue
            
            extracted_file = folder / "extracted.txt"
            
            if not extracted_file.exists():
                skipped_count += 1
                continue
            
            # 파일 읽기
            try:
                with open(extracted_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    skipped_count += 1
                    continue
                
                # 메타데이터 구조 수정
                self.policies.append({
                    'title': title,
                    'id': policy_id,
                    'content': content,  # 본문 전체
                    'source': str(extracted_file)  # filepath → source로 변경
                })
                found_count += 1
                
            except Exception as e:
                print(f"⚠️  {folder.name} 읽기 실패: {e}")
                skipped_count += 1
                continue
        
        print(f"\n✅ 새로 발견: {found_count}개")
        print(f"♻️  캐시됨: {cached_count}개")
        print(f"⏭️  스킵: {skipped_count}개")
        print(f"📊 총 처리 대상: {len(self.policies)}개 정책")
    
    @torch.no_grad()
    def generate_embedding(self, text: str) -> np.ndarray:
        """텍스트를 임베딩 벡터로 변환 (CLS 토큰 사용, GPU 최적화)"""
        # 토큰화 (긴 텍스트는 앞부분만 사용)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192  # bge-m3는 최대 8192 토큰 지원
        ).to(self.device)
        
        # GPU Mixed Precision 사용 (더 빠른 처리)
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
        
        # CLS 토큰 사용 (bge-m3 권장 방식)
        embeddings = outputs.last_hidden_state[:, 0]
        
        # 정규화
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()[0]
    
    def build_faiss_index(self, output_dir: str = None, use_gpu_index: bool = True):
        """FAISS 인덱스 구축 (기존 임베딩 재사용, GPU 가속)"""
        if output_dir is None:
            output_dir = self.base_path.parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 기존 인덱스와 메타데이터 로드
        index_path = output_dir / "policy_faiss.index"
        metadata_path = output_dir / "metadata.json"
        
        existing_embeddings = []
        existing_metadata = self.load_existing_metadata(metadata_path)
        
        if index_path.exists() and existing_metadata:
            print(f"📥 기존 FAISS 인덱스 로딩: {index_path}")
            try:
                existing_index = faiss.read_index(str(index_path))
                # GPU 인덱스인 경우 CPU로 이동
                if hasattr(existing_index, 'getDevice'):
                    existing_index = faiss.index_gpu_to_cpu(existing_index)
                # 기존 임베딩을 numpy 배열로 복원
                existing_embeddings = np.array([
                    existing_index.reconstruct(i) 
                    for i in range(existing_index.ntotal)
                ]).astype('float32')
                print(f"✅ {len(existing_embeddings)}개 기존 임베딩 로드 완료")
            except Exception as e:
                print(f"⚠️  기존 인덱스 로딩 실패: {e}")
                existing_embeddings = []
                existing_metadata = []
        
        # 새로운 정책이 있는 경우에만 임베딩 생성
        new_embeddings = []
        if self.policies:
            print(f"\n🧮 새로운 {len(self.policies)}개 정책 임베딩 생성 중 (GPU 가속)...")
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
            
            for policy in tqdm(self.policies, desc="임베딩 생성"):
                # content 본문을 임베딩
                embedding = self.generate_embedding(policy['content'])
                new_embeddings.append(embedding)
                
                # 메타데이터 저장 (4개 필드만)
                existing_metadata.append({
                    'title': policy['title'],
                    'id': policy['id'],
                    'content': policy['content'],  # 본문 전체 포함
                    'source': policy['source']  # extracted.txt 경로
                })
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
        
        # 기존 + 새로운 임베딩 결합
        if len(existing_embeddings) > 0:
            if len(new_embeddings) > 0:
                new_embeddings_array = np.array(new_embeddings).astype('float32')
                embeddings_array = np.vstack([existing_embeddings, new_embeddings_array])
                print(f"\n📊 기존 {len(existing_embeddings)}개 + 신규 {len(new_embeddings)}개 = 총 {len(embeddings_array)}개")
                print(f"📊 최종 배열 Dtype: {embeddings_array.dtype}")
            else:
                embeddings_array = existing_embeddings
                print(f"\n📊 기존 임베딩만 사용: {len(embeddings_array)}개")
        else:
            if len(new_embeddings) > 0:
                embeddings_array = np.array(new_embeddings).astype('float32')
                print(f"\n📊 새로운 임베딩만 생성: {len(embeddings_array)}개")
            else:
                raise ValueError("처리할 정책이 없습니다.")
        
        self.metadata = existing_metadata
        dimension = embeddings_array.shape[1]

        print(f"📏 임베딩 차원: {dimension}")
        
        print(f"📏 임베딩 차원: {dimension}")
        print(f"📊 총 벡터 개수: {len(embeddings_array)}")
        
        # FAISS 인덱스 생성 (Inner Product = Cosine Similarity)
        print("\n🔨 FAISS 인덱스 구축 중...")
        cpu_index = faiss.IndexFlatIP(dimension)
        cpu_index.add(embeddings_array)
        
        # GPU 인덱스로 변환 (선택사항)
        save_index = cpu_index
        print(f"✅ CPU 인덱스에 {cpu_index.ntotal}개 벡터 추가됨")
        
        # 저장
        print(f"\n💾 저장 중...")
        faiss.write_index(save_index, str(index_path))
        print(f"✅ FAISS 인덱스 저장: {index_path}")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"✅ 메타데이터 저장: {metadata_path}")
        
        # 메타데이터 구조 확인 출력
        if self.metadata:
            print(f"\n📋 메타데이터 샘플 (첫 번째 항목):")
            print("-" * 60)
            sample = self.metadata[0]
            print(f"  title: {sample['title'][:50]}...")
            print(f"  id: {sample['id']}")
            print(f"  content: {sample['content'][:100]}...")
            print(f"  source: {sample['source']}")
            print("-" * 60)
        
        # 통계 출력
        print("\n" + "="*60)
        print("📊 구축 완료 통계")
        print("="*60)
        print(f"총 정책 수: {len(self.metadata)}")
        print(f"임베딩 차원: {dimension}")
        print(f"신규 추가: {len(new_embeddings)}개")
        print(f"GPU 가속: {'✅ 사용' if use_gpu_index else '❌ 미사용'}")
        print(f"인덱스 파일: {index_path}")
        print(f"메타데이터 파일: {metadata_path}")
        print("\n메타데이터 필드:")
        print("  1. title (제목)")
        print("  2. id (정책 ID)")
        print("  3. content (본문 전체)")
        print("  4. source (extracted.txt 경로)")
        print("="*60)
        
        return index_path, metadata_path
    
    def test_search(self, query: str, k: int = 5, sort_by: str = 'similarity', use_gpu: bool = True):
        """
        테스트 검색 수행 (GPU 가속)
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 개수
            sort_by: 정렬 기준 ('similarity', 'id', 'title')
            use_gpu: GPU 사용 여부
        """
        if not self.metadata:
            print("⚠️  메타데이터가 없습니다. build_faiss_index()를 먼저 실행하세요.")
            return
        
        print(f"\n🔍 테스트 검색: '{query}'")
        
        # 쿼리 임베딩
        query_embedding = self.generate_embedding(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # 인덱스 로드
        output_dir = self.base_path.parent
        index_path = output_dir / "policy_faiss.index"
        cpu_index = faiss.read_index(str(index_path))
        
        # GPU 인덱스로 변환 (선택사항)
        index = cpu_index
        
        # 검색 (더 많은 결과를 가져온 후 정렬)
        search_k = max(k * 2, 20)  # 정렬을 위해 더 많이 가져옴
        distances, indices = index.search(query_embedding, min(search_k, len(self.metadata)))
        
        # 결과를 리스트로 변환
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append({
                    'similarity': float(distance),
                    'id': meta['id'],
                    'title': meta['title'],
                    'content': meta['content'][:200] + "...",  # 미리보기
                    'source': meta['source']
                })
        
        # 정렬
        if sort_by == 'similarity':
            results.sort(key=lambda x: x['similarity'], reverse=True)
        elif sort_by == 'id':
            results.sort(key=lambda x: x['id'])
        elif sort_by == 'title':
            results.sort(key=lambda x: x['title'])
        
        # 상위 k개만 선택
        results = results[:k]
        
        print(f"\n📋 상위 {k}개 검색 결과 (정렬: {sort_by}, GPU: {'✅' if use_gpu else '❌'}):")
        print("-" * 80)
        
        for rank, result in enumerate(results, 1):
            print(f"\n{rank}. [{result['id']}]")
            print(f"   제목: {result['title']}")
            print(f"   유사도: {result['similarity']:.4f}")
            print(f"   본문 미리보기: {result['content']}")
            print(f"   출처: {result['source']}")
        
        print("-" * 80)
        
        return results


def main():
    """메인 실행 함수"""
    # ⚠️ 여기를 본인의 바탕화면 경로에 맞게 수정하세요
    # Windows 예시: "C:/Users/YourName/Desktop/bizinfo_data"
    # Mac/Linux 예시: "~/Desktop/bizinfo_data"
    
    BASE_PATH = "C:\\Users\\user\\Desktop\\bizinfo_data"  # 👈 이 부분을 수정하세요!
    MODEL_PATH = "C:\\Users\\user\\Desktop\\bge-m3-sft"  # 👈 파인튜닝 모델 경로
    
    try:
        # 빌더 초기화
        builder = PolicyVectorBuilder(
            base_path=BASE_PATH,
            model_name=MODEL_PATH
        )
        
        # 정책 스캔
        builder.scan_policies()
        
        # FAISS 인덱스 구축
        index_path, metadata_path = builder.build_faiss_index()
        
        # 테스트 검색 (선택사항)
        print("\n" + "="*60)
        print("🧪 테스트 검색 실행")
        print("="*60)
        
        test_queries = [
            "중소기업 기술개발 지원",
            "스타트업 창업 자금",
            "수출 지원 사업"
        ]
        
        # 유사도 순 검색
        for query in test_queries:
            builder.test_search(query, k=3, sort_by='similarity')
            print()
        
        # ID 순 정렬 예시
        print("\n" + "="*60)
        print("📌 ID 순 정렬 예시")
        print("="*60)
        builder.test_search("기술개발", k=5, sort_by='id')
        
        # 제목 순 정렬 예시
        print("\n" + "="*60)
        print("📌 제목 순 정렬 예시")
        print("="*60)
        builder.test_search("지원사업", k=5, sort_by='title')
        
        print("\n🎉 모든 작업이 완료되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()