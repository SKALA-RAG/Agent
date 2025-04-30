import logging
import os
from typing import Dict, Any, List, Union
from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import json

# 환경 변수 로드
load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

model = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.1, 
    openai_api_key=OPENAI_API_KEY
)

# 보고서 생성 프롬프트 - 각 Agent의 결과가 주입될 위치 명시
report_template = """
당신은 전문 투자 분석가입니다. 다음 정보를 바탕으로 스타트업 투자 검토 보고서를 작성해주세요.
각 섹션은 주어진 정보를 충실히 반영하고, 최종 투자 판단은 모든 정보를 종합하여 논리적으로 내려주세요.
보고서는 반드시 한국어로 작성되어야 합니다.

**스타트업 투자 검토 보고서**

**1. 기업 개요**
{exploration_summary}
* *상세 정보는 아래 각 항목 참고*
* 회사명 / 설립일 / 소재지
* 대표자 및 주요 경영진: 이력, 주요 경력, 핵심 인력 소개
* 주요 연혁: 설립, 투자유치, 특허/인증, 주요 거래처 등

**2. 사업 및 비즈니스 모델**
(기업 개요({exploration_summary}) 내 '주요 사업 분야' 등 관련 내용을 바탕으로 상세히 서술)
* 핵심 사업 내용: 무엇을, 누구에게, 어떻게 제공하는가?
* 비즈니스 모델: 수익 구조, 고객 유치/유지 전략
* 시장 문제점 및 Pain Point: 어떤 문제를 해결하는가?

**3. 시장성 평가**
(시장성 분석 결과({market_summary})를 바탕으로 다음 항목을 상세히 서술)
* 시장 규모 및 성장성: TAM/SAM/SOM, 연평균 성장률, 주요 통계
* 시장 트렌드 및 기회: 산업 동향, 성장 요인, 진입장벽
* 고객 및 수요 분석: 주요 타깃 고객, 고객 세분화

**4. 경쟁사 및 차별성 분석**
(경쟁사 분석 결과({competitor_list}) 와 '주요 차별화 요소' ({competitor_summary}) 등 관련 내용을 바탕으로 다음 항목을 상세히 서술)

* 주요 경쟁사 리스트: 경쟁사별 장단점 요약
* 경쟁 우위 요소: 기술, 가격, 네트워크, 브랜드 등
* 차별화 전략: 우리만의 강점, 진입장벽

**5. 기술력 및 지식재산권**
(기업 개요({exploration_summary}) 내 '주요 AI 기술' 및 실적 정보({perform_summary}) 내 특허 관련 내용을 바탕으로 서술)
* 핵심 기술 요약: 기술 설명, 적용 분야, 혁신성
* 논문/특허/인증: 보유 현황, 출원/등록 내역, 기술적 차별성
* 기술 로드맵: 향후 개발 계획

**6. 팀 구성 및 조직 역량**
(기업 개요({exploration_summary}) 내 '핵심 인력' 및 창업자 정보 ({founder_summary})내용을 바탕으로 서술)
* 핵심 인력 및 역할: CTO, CMO 등 주요 인력 소개
* 조직 구조: 팀원 수, 조직도, 외부 자문/파트너
* 팀의 강점: 실무 경험, 업계 네트워크, 실행력 등

**7. 재무 현황 및 계획**
(기업 실적 정보({perform_summary}) 내 '투자 유치', '매출' 등 관련 내용을 바탕으로 서술)
* 주요 재무 지표: 매출, 영업이익, 순이익, 현금 흐름 등
* 투자 유치 내역: 과거 투자자, 투자금, 지분 구조
* 향후 자금 계획: 투자금 사용 계획, 추가 자금 조달 계획

**8. 사업 확장 및 성장 전략**
(기업 개요({exploration_summary}) 내 '주요 연혁', 성장 계획 등 관련 내용을 바탕으로 서술)
* 단기/중장기 성장 계획: 제품/서비스 확장, 글로벌 진출 등
* 주요 마일스톤: 향후 1~3년 내 달성 목표
* 위험 요인 및 대응 전략: 시장, 기술, 재무 등

**9. 투자 판단 및 종합 의견**
(투자 판단 결과 {investment_analysis}를 바탕으로 서술)
* 투자 포인트: 투자 매력, 기대 효과, Exit 전략(회수 방안)
* 리스크 요인 및 개선 과제
* 최종 평가 및 투자 의견: 투자/보류/기타

**[부록]** (참고 자료 목록은 별도로 관리)
* 데이터 출처 명시 (가능한 경우 각 항목 분석 시 명시)

---
보고서 작성을 시작하세요.
"""

report_prompt = PromptTemplate.from_template(report_template)
output_parser = StrOutputParser()
report_generation_chain = report_prompt | model | output_parser

async def create_final_report(results_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    여러 에이전트의 결과를 종합하여 최종 투자 보고서를 생성
    
    Args:
        results_data: 여러 에이전트의 결과가 담긴 리스트
        [0]: 스타트업 탐색 에이전트의 결과
        [1]: 실적 및 창업자 정보 에이전트의 결과
        [2]: 경쟁사 비교 에이전트의 결과
        [3]: 시장성 평가 에이전트의 결과
        
    Returns:
        Dict[str, Any]: 최종 보고서 및 관련 정보
    """
    try:
        # 입력 데이터 정리
        exploration_result = results_data[0]
        perform_info = results_data[1]
        competitor_info = results_data[2]
        market_info = results_data[3]
        investment_analysis_result = results_data[4]

        exploration_summary = exploration_result.get("기업 정보 요약", "정보 없음")

        # perform_info에서 두 정보를 별도로 추출
        perform_summary = perform_info.get("기업 실적 요약", "정보 없음")
        founder_summary = perform_info.get("창업자 정보 요약", "정보 없음") 

        # competitor_info에서 두 정보를 별도로 추출
        competitor_list = competitor_info.get("주요 경쟁사 목록", "정보 없음")
        competitor_analysis = competitor_info.get("경쟁사 비교 분석", "정보 없음")

        market_summary = market_info.get("시장성 종합 분석", "정보 없음")

        investment_analysis = investment_analysis_result.get("투자 판단 보고서", "정보 없음")

        logging.info("최종 보고서 작성 시작")
        # 2. 최종 보고서 생성
        final_report = await report_generation_chain.ainvoke({
            "exploration_summary": exploration_summary,
            "perform_summary": perform_summary,
            "founder_summary": founder_summary,
            "competitor_summary": competitor_analysis,
            "competitor_list": competitor_list,
            "market_summary": market_summary,
            "investment_analysis": investment_analysis
        })
        logging.info("최종 보고서 작성 완료")
                
        return final_report

    except Exception as e:
        logging.error(f"보고서 생성 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report Generation Error: {str(e)}")
