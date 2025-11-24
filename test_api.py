"""
API 테스트 스크립트
유해성 필터링 API (클린봇 AI)와 RAG 챗봇 API, GPT 제목 생성 API를 테스트합니다.
"""

import requests
import json

# API 서버 URL
BASE_URL = "http://localhost:8000"

# ===================================================================
# 1. 유해성 필터링 API 테스트
# ===================================================================

def test_filter_api(query: str):
    """유해성 필터링 API (클린봇 AI) 테스트"""
    print(f"\n{'='*60}")
    print(f"[클린봇 AI 테스트] 질문: {query}")
    print(f"{'='*60}")
    
    url = f"{BASE_URL}/api/v1/filter/text" 
    payload = {
        "user": {"loginId": "test_user", "nickname": "테스터"},
        "query": query
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        
        if response.status_code == 200:
            data = response.json()
            if data.get('blocked'):
                print(f"\n🚫 [차단됨] 유해성 탐지: toxicity={data.get('toxicity'):.4f}")
            else:
                print(f"\n✅ [통과] toxicity={data.get('toxicity'):.4f}")
        elif response.status_code == 400:
             print(f"\n❌ [400 오류] 필수 필드 누락 검사 성공.")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

# ===================================================================
# 2. RAG 챗봇 API 테스트
# ===================================================================

def test_rag_api(query: str):
    """RAG 챗봇 API 테스트 (유해성 필터링 제외됨)"""
    print(f"\n{'='*60}")
    print(f"[RAG 챗봇 테스트] 질문: {query}")
    print(f"{'='*60}")
    
    url = f"{BASE_URL}/api/v1/query"
    payload = {
        "user": {"loginId": "test_user", "nickname": "테스터"},
        "query": query
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        
        if response.status_code == 200:
            print(f"\n✅ [답변 성공]")
        elif response.status_code == 400:
            print(f"\n❌ [400 오류] 필수 필드 누락 검사 성공.")
        elif response.status_code == 500:
            print(f"\n⚠️ [500 오류] RAG 처리 중 내부 오류 발생.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

# ===================================================================
# 3. GPT 제목 생성 API 테스트
# ===================================================================

def test_title_api(query: str):
    """GPT 제목 생성 API 테스트"""
    print(f"\n{'='*60}")
    print(f"[제목 생성 테스트] 질문: {query}")
    print(f"{'='*60}")
    
    # title_router의 prefix가 /generate_title이므로 엔드포인트는 /입니다.
    url = f"{BASE_URL}/generate_title/" 
    payload = {"question": query}
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        
        if response.status_code == 200:
            print(f"\n✅ [제목 생성 성공] 제목: {response.json().get('title')}")
        elif response.status_code == 400:
            print(f"\n❌ [400 오류] 필수 필드 누락 검사 성공.")
        elif response.status_code == 500:
            print(f"\n⚠️ [500 오류] 제목 생성 오류 반환 또는 내부 오류.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


# ===================================================================
# 4. 헬스체크 API 테스트
# ===================================================================

def test_health():
    """헬스체크 API 테스트"""
    print(f"\n{'='*60}")
    print(f"[헬스체크 API 테스트]")
    print(f"{'='*60}")
    
    url = f"{BASE_URL}/health"
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    print("="*60)
    print(" RAG Chatbot with Toxicity Filter & Title Generation - API 테스트")
    print("="*60)
    
    # 1. 헬스체크
    test_health()
    
    # 2. 클린봇 AI 테스트 (POST /api/v1/filter/text)
    print("\n\n" + "="*60)
    print(" 2. 클린봇 AI (유해성 필터링) 테스트")
    print("="*60)
    test_filter_api("야 이 미친놈아 뭐하는 짓이야") # 유해 (차단 기대)
    test_filter_api("중소기업 지원 정책을 알려주세요") # 정상 (통과 기대)
    test_filter_api("") # 빈 쿼리 (400 기대)
    
    # 3. RAG 챗봇 테스트 (POST /api/v1/query)
    print("\n\n" + "="*60)
    print(" 3. RAG 챗봇 테스트")
    print("="*60)
    test_rag_api("창업 자금 지원 방법을 알고 싶습니다") # 정상 (200 기대)
    test_rag_api("씨발 정책이 왜 이따위야") # 욕설 포함 (필터링 제외. 200 또는 500 기대)
    test_rag_api("") # 빈 쿼리 (400 기대)

    # 4. GPT 제목 생성 테스트 (POST /generate_title/)
    print("\n\n" + "="*60)
    print(" 4. GPT 제목 생성 테스트")
    print("="*60)
    test_title_api("코로나19로 매출이 줄었는데, 손실보전금 신청 기간을 알려주세요.") # 정상 (200 기대)
    test_title_api("") # 빈 쿼리 (400 기대)
    
    print("\n\n" + "="*60)
    print(" 모든 테스트 시나리오 완료")
    print("="*60)