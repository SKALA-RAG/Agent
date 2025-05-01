import logging
import os
import re
from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

# 환경 변수 로드
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    openai_api_key=OPENAI_API_KEY
)

tavily = TavilySearchResults(max_results=20)

# 프롬프트 템플릿
market_analysis_prompt = PromptTemplate.from_template(
    """
    당신은 VC(벤처 캐피탈)의 투자 심사역으로, 실제 투자 의사결정에 필요한 시장성 분석을 수행합니다.
    아래의 웹 검색 결과를 바탕으로, '{company_name}'의 주요 사업 모델에 대한 시장성을 평가하세요.
    
    분석 항목:
    1. 시장 규모 및 성장성 (반드시 구체적인 수치 포함)
       - TAM(Total Addressable Market): 시장 전체 규모 (달러/원 단위, CAGR)
       - SAM(Serviceable Available Market): 실제 도달 가능한 시장 규모 (달러/원 단위)
       - SOM(Serviceable Obtainable Market): 현실적으로 점유 가능한 시장 비율 및 규모 (%)
       - 매출액 기준 잠재 시장 규모와 해당 기업의 잠재 점유율 추정

    2. 시장 매력도 및 성장 동인
       - 향후 3-5년 시장 CAGR 전망 및 근거 (구체적인 % 수치)
       - 시장 성장을 견인하는 주요 요인 3-5가지 (기술 발전, 규제 환경, 소비자 행동 변화 등)
       - 해당 시장의 수익성 지표 (업계 평균 영업이익률, 매출총이익률 등)
       - 시장 성숙도 단계 (도입기/성장기/성숙기/쇠퇴기) 및 판단 근거

    3. 시장 경쟁 강도 분석
       - 주요 경쟁자 시장점유율 분포 및 집중도
       - 시장 진입장벽 (높음/중간/낮음) 및 판단 근거
       - 해당 기업의 차별화 포인트 및 지속가능한 경쟁우위 요소
       - 신규 진입자 위협 수준 및 대체재 위협 수준

    4. 고객 및 채널 분석
       - B2B/B2C/B2B2C 판매 비중 및 주요 고객군 특성
       - 주 수익원 및 수익 모델의 다각화 정도
       - 고객 획득 비용(CAC) 및 고객 생애 가치(LTV) 추정치 또는 업계 평균
       - 판매/유통 채널의 특성 및 확장성
    
    5. 종합 결론
        - **위 1~4번 분석 결과를 바탕으로** 스타트업의 시장성에 대한 구체적인 결론을 도출해주세요.
        - 잠재 수요 분석이 스타트업의 성장 가능성을 어떻게 뒷받침하는지 설명하세요.
        - 최종적으로, 분석된 기회와 위험을 고려하여 스타트업의 **시장성**에 대한 합리적인 판단을 제시하세요.
    
    
    데이터 기반 분석이 중요합니다. 가능한 한 구체적인 수치, 비율, 금액을 포함하세요.
    확실하지 않은 추정치는 "업계 평균 기준", "유사 기업 비교", "전문가 의견 기준" 등 근거를 명시하세요.
    결론은 매우 구체적이고 실행 가능한 내용으로 작성하세요. 모호한 일반론은 피하고, 투자 결정에 직접적으로 도움이 되는 인사이트를 제공하세요.
    
    [검색 자료]
    {text}
    """
)

output_parser = StrOutputParser()

market_analysis_chain = market_analysis_prompt | model | output_parser

async def assess_market_potential(data: str):
    company_name, industry = extract_company_info(data)
    
    try:
        # 통합된 시장성 분석 쿼리 (규모, 성장성, 트렌드 포함)
        market_query = f"{company_name} {industry} 시장 규모 TAM SAM SOM 성장률 트렌드 전망"
        market_search_result = tavily.invoke(market_query)
        market_analysis = market_analysis_chain.invoke({
            "company_name": company_name,
            "text": market_search_result
        })
        logging.info(f"Market Analysis Result: {market_analysis}")
        
        # 추가 정보 검색 (고객 세그먼트, 진입 장벽)
        additional_query = f"{company_name} {industry} 고객 세그먼트 시장 진입장벽 경쟁 규제"
        additional_search_result = tavily.invoke(additional_query)
        
        # 검색 결과 합치기 및 분석 확장
        combined_search_results = market_search_result + additional_search_result
        comprehensive_analysis = market_analysis_chain.invoke({
            "company_name": company_name,
            "text": combined_search_results
        })
        
        logging.info(f"Comprehensive Market Analysis Result: {comprehensive_analysis}")
        
        # 결과 반환
        final_result = {
            "시장성 종합 분석": comprehensive_analysis
        }
        
        return final_result
    
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI or Search Error: {str(e)}")

def extract_company_info(llm_response: str):
    company_pattern = r"회사명\s*[:：]\s*(.*)"
    industry_pattern = r"산업분야\s*[:：]\s*(.*)"
    
    company_match = re.search(company_pattern, llm_response)
    industry_match = re.search(industry_pattern, llm_response)
    
    company = company_match.group(1).strip() if company_match else None
    industry = industry_match.group(1).strip() if industry_match else None
    
    logging.info(f"company: {company}, industry: {industry}")
    return company, industry

# from dotenv import load_dotenv
# from typing import Annotated, Sequence, TypedDict, Literal, Dict, Any, List
# from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
# from langgraph.graph.message import add_messages
# from langgraph.graph import END, StateGraph, START
# from langchain.tools import Tool
# from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from pydantic import BaseModel, Field
# from langchain_openai import ChatOpenAI
# from langchain_community.tools import TavilySearchResults
# import os
# import re

# # 환경 변수 로드
# load_dotenv()

# # 검색 도구 생성
# def create_tools():
#     # 웹 검색 도구 생성
#     web_search_tool = TavilySearchResults(
#         max_results=5,
#         api_key=os.environ.get("TAVILY_API_KEY")
#     )

#     tools = [
#         Tool(
#             name="web_search",
#             func=web_search_tool.invoke,
#             description="Search the web for recent and real-time information about ICT and BioHealth startups, their funding, market trends, and competitive landscape. " \
#             "Use this when you need up-to-date information to supplement the 2025 industry outlook report or when looking for specific company details."
#         )
#     ]
#     return tools

# # 시장성 평가 Agent 상태 정의
# class MarketAssessmentState(TypedDict):
#     startup_info: Dict
#     market_data: Dict
#     analysis_results: Dict
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     error: str
#     search_query: str
#     search_result: str

# # 스타트업 정보 텍스트를 startup_info 딕셔너리로 변환하는 함수
# def text_to_startup_info(text):
#     """   
#     Returns:
#         dict: 구조화된 startup_info 딕셔너리
#     """
#     # 기본 startup_info 구조 초기화
#     startup_info = {
#         "name": "",
#         "founding_year": "",
#         "founders": [],
#         "business_area": "",
#         "homepage": "",
#         "contact": "",
#         "key_personnel": [],
#         "major_milestones": [],
#     }
    
#     # 1. 회사명 추출
#     company_match = re.search(r'1\.\s*\*{0,2}회사명\*{0,2}:\s*(.*?)(?:\n|$)', text, re.DOTALL)
#     if company_match:
#         startup_info["name"] = company_match.group(1).strip()
    
#     # 2. 설립연도 추출
#     year_match = re.search(r'2\.\s*\*{0,2}설립연도\*{0,2}:\s*(.*?)(?:\n|$)', text, re.DOTALL)
#     if year_match:
#         year_text = year_match.group(1).strip()
#         # 숫자만 추출
#         year_num = re.search(r'\d{4}', year_text)
#         if year_num:
#             startup_info["founding_year"] = year_num.group(0)
#         else:
#             startup_info["founding_year"] = ""
    
#     # 3. 대표자 추출
#     ceo_match = re.search(r'3\.\s*\*{0,2}대표자\*{0,2}:\s*(.*?)(?:\n|$)', text, re.DOTALL)
#     if ceo_match:
#         ceo_text = ceo_match.group(1).strip()
#         # 괄호가 있는 경우 처리 (예: 이름 (직책))
#         name_parts = re.split(r'\s+\(', ceo_text)
#         ceo_name = name_parts[0].strip()
#         startup_info["founders"] = [ceo_name]
    
#     # 4. 주요 사업 분야 추출
#     business_match = re.search(r'4\.\s*\*{0,2}주요\s*사업\s*분야\*{0,2}:\s*(.*?)(?:\n|$)', text, re.DOTALL)
#     if business_match:
#         business_text = business_match.group(1).strip()
#         startup_info["business_area"] = business_text
#         startup_info["product"] = business_text  # product 필드도 함께 설정
        
#         # 산업 분야 추론
#         if "AI" in business_text or "인공지능" in business_text:
#             if "고객" in business_text or "서비스" in business_text:
#                 startup_info["industry"] = "AI Customer Service"
#             else:
#                 startup_info["industry"] = "Artificial Intelligence"
        
#         # 타겟 시장 추론
#         if "고객" in business_text and "센터" in business_text:
#             startup_info["target_market"] = "기업(B2B), 고객 서비스 센터"
    
#     # 5. 홈페이지 추출
#     homepage_match = re.search(r'5\.\s*\*{0,2}홈페이지\*{0,2}:\s*(.*?)(?:\n|$)', text, re.DOTALL)
#     if homepage_match:
#         homepage = homepage_match.group(1).strip()
#         # URL 형식이 아니면 형식 추가
#         if not homepage.startswith(('http://', 'https://')):
#             if not homepage.startswith('www.'):
#                 homepage = 'https://www.' + homepage
#             else:
#                 homepage = 'https://' + homepage
#         startup_info["homepage"] = homepage
    
#     # 6. 연락처 추출
#     contact_match = re.search(r'6\.\s*\*{0,2}연락처\*{0,2}:\s*(.*?)(?:\n|$)', text, re.DOTALL)
#     if contact_match:
#         contact = contact_match.group(1).strip()
#         startup_info["contact"] = contact
    
#     # 7. 핵심 인력 추출
#     personnel_pattern = r'7\.\s*\*{0,2}핵심\s*인력.*?\*{0,2}:(?:\s*\n?)(.+?)(?=8\.|$)'
#     personnel_match = re.search(personnel_pattern, text, re.DOTALL)
    
#     if personnel_match:
#         personnel_text = personnel_match.group(1).strip()
#         personnel_list = []
        
#         # 각 줄 또는 항목별로 분리
#         for line in re.split(r'\n\s*\*\s*|\n\s*-\s*|\n\s*•\s*', personnel_text):
#             line = line.strip()
#             if line and not line.startswith(('8.', '9.')):  # 다음 섹션 제외
#                 personnel_list.append(line)
        
#         startup_info["key_personnel"] = personnel_list
    
#     # 8. 주요 연혁 추출
#     milestones_pattern = r'8\.\s*\*{0,2}주요\s*연혁.*?\*{0,2}:(?:\s*\n?)(.+?)(?=9\.|$)'
#     milestones_match = re.search(milestones_pattern, text, re.DOTALL)
    
#     if milestones_match:
#         milestones_text = milestones_match.group(1).strip()
#         milestones_list = []
        
#         # 각 줄 또는 항목별로 분리
#         for line in re.split(r'\n\s*\*\s*|\n\s*-\s*|\n\s*•\s*', milestones_text):
#             line = line.strip()
#             if line and not line.startswith('9.'):  # 다음 섹션 제외
#                 milestones_list.append(line)
        
#         startup_info["major_milestones"] = milestones_list
    
#     return startup_info


# # 검색 결과 관련성 확인을 위한 데이터 모델
# class RelevanceGrade(BaseModel):
#     binary_score: str = Field(
#         description="Response 'yes' if the document is relevant to the question or 'no' if it is not."
#     )
#     reasoning: str = Field(
#         description="Brief explanation of why the document is or is not relevant."
#     )

# # AI 에이전트 노드: 사용자 입력을 처리하고 적절한 도구를 호출
# def agent(state: MarketAssessmentState) -> MarketAssessmentState:
#     # 현재 상태에서 메시지 추출
#     messages = state.get("messages", [])
    
#     # 스타트업 정보 추출 (첫 메시지에서)
#     if "startup_info" not in state or not state["startup_info"]:
#         try:
#             first_message = messages[0]
#             # 첫 메시지에서 스타트업 정보를 추출하거나 파싱
#             # 여기서는 첫 메시지가 이미 스타트업 정보 딕셔너리로 가정
#             if isinstance(first_message.content, dict):
#                 startup_info = first_message.content
#             else:
#                 # 텍스트에서 스타트업 정보 파싱 로직 구현
#                 startup_info = {"name": "Unknown Startup", "industry": "AI"}
#         except Exception as e:
#             return {
#                 "messages": messages + [AIMessage(content=f"스타트업 정보를 처리하는 중 오류가 발생했습니다: {str(e)}")],
#                 "error": f"스타트업 정보 처리 오류: {str(e)}"
#             }
#     else:
#         startup_info = state["startup_info"]
    
#     # LLM 모델 초기화
#     model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
#     # 도구 바인딩
#     model_with_tools = model.bind_tools(tools)
    
#     # 에이전트 프롬프트 템플릿
#     agent_prompt = """당신은 AI 스타트업 투자 가능성을 평가하는 전문가입니다.
#     스타트업의 주요 사업 모델에 대한 시장성을 평가하기 위해 필요한 정보를 수집해야 합니다.
    
#     평가할 스타트업 정보:
#     {startup_info}
    
#     당신의 임무는 다음과 같습니다:
#     1. 해당 스타트업이 속한 시장의 TAM(Total Addressable Market), SAM(Serviceable Available Market), SOM(Serviceable Obtainable Market)을 분석
#     2. 시장 성장률 및 트렌드 파악
#     3. 판매영역별, 고객별 잠재수요 분석
#     4. 시장 진입 장벽 분석
    
#     필요한 정보를 검색하기 위해 제공된 도구를 활용하세요.
#     정보를 찾으면 한국어로 응답하세요.
#     """
    
#     # 프롬프트 포맷팅
#     formatted_prompt = agent_prompt.format(startup_info=startup_info)
    
#     # 새 메시지 생성
#     new_messages = messages + [HumanMessage(content=formatted_prompt)]
    
#     # 에이전트 응답 생성
#     response = model_with_tools.invoke(new_messages)
    
#     # 상태 업데이트 및 반환
#     return {
#         "startup_info": startup_info,
#         "messages": messages + [response],
#         "market_data": state.get("market_data", {})
#     }

# # 검색 결과 관련성 평가 노드
# def grade_documents(state: MarketAssessmentState) -> Literal["generate", "rewrite"]:
#     """검색 결과의 관련성을 평가하는 라우터 노드"""
#     # LLM 모델 초기화
#     model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
#     # 구조화된 출력을 위한 LLM 설정
#     llm_with_tool = model.with_structured_output(RelevanceGrade)
    
#     # 프롬프트 템플릿 정의
#     prompt_template = """당신은 검색된 문서의 관련성을 평가하는 전문가입니다. 스타트업 시장 분석 질문에 대한 검색 결과의 관련성을 평가해주세요.

#     스타트업 정보:
#     {startup_info}

#     검색된 문서 또는 검색 결과:
#     {context}

#     찾고 있는 시장 분석 측면:
#     {question}

#     이 문서가 해당 스타트업의 시장 분석에 관련된 키워드나 의미론적 내용을 포함하고 있다면, 관련성이 있다고 평가하세요.

#     문서가 질문과 관련이 있는지 여부를 'yes' 또는 'no'로 응답하고, 간략하게 이유를 설명해주세요."""
    
#     # 현재 상태에서 메시지 추출
#     messages = state["messages"]
#     last_message = messages[-1]
    
#     # 검색된 문서 추출
#     search_result = state.get("search_result", "")
#     if not search_result and hasattr(last_message, "content"):
#         search_result = last_message.content
    
#     # 스타트업 정보
#     startup_info = state.get("startup_info", {})
    
#     # 검색 쿼리/질문
#     search_query = state.get("search_query", "시장 규모 및 성장성 분석")
    
#     # 관련성 평가 실행
#     human_message = HumanMessage(
#         content=prompt_template.format(
#             startup_info=startup_info,
#             context=search_result,
#             question=search_query
#         )
#     )
    
#     # 응답 생성
#     result = llm_with_tool.invoke([human_message])
    
#     # 관련성 여부에 따른 결정
#     if result.binary_score.lower() == "yes":
#         return "generate"
#     else:
#         return "rewrite"


# def rewrite_query(state: MarketAssessmentState) -> MarketAssessmentState:
#     """검색 쿼리를 재작성하는 노드"""
    
#     # 현재 상태에서 필요한 정보 추출
#     startup_info = state.get("startup_info", {})
#     search_query = state.get("search_query", "")
    
#     # LLM 모델 초기화
#     model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
#     # 쿼리 재작성 프롬프트
#     rewrite_prompt = """당신은 시장 분석 전문가입니다. 주어진 스타트업에 대한 시장 정보를 검색하기 위해 쿼리를 개선해주세요.

#     스타트업 정보:
#     {startup_info}

#     원래 검색 쿼리:
#     {search_query}

#     이 쿼리로는 관련 정보를 충분히 찾지 못했습니다. 더 구체적이고 관련성 높은 검색 결과를 얻기 위해 쿼리를 재작성해주세요.
#     다음 사항을 고려하세요:
#     1. 산업별 키워드 포함
#     2. 시장 규모, 성장률, 트렌드 등 구체적인 측면 명시
#     3. 최신 정보를 얻기 위한 시간 범위 지정 (필요시)

#     개선된 검색 쿼리만 작성해주세요:"""
        
#     # 프롬프트 포맷팅
#     formatted_prompt = rewrite_prompt.format(
#         startup_info=startup_info,
#         search_query=search_query
#     )
    
#     # 재작성된 쿼리 생성
#     response = model.invoke([HumanMessage(content=formatted_prompt)])
#     rewritten_query = response.content.strip()
    
#     # 상태 업데이트 및 반환
#     return {
#         "startup_info": startup_info,
#         "search_query": rewritten_query,
#         "messages": state.get("messages", []) + [AIMessage(content=f"검색 쿼리를 재작성했습니다: {rewritten_query}")]
#     }

# def search_market_data(state: MarketAssessmentState) -> MarketAssessmentState:
#     """시장 데이터 검색 노드"""
#     # 필요한 정보 추출
#     startup_info = state.get("startup_info", {})
#     search_query = state.get("search_query", "")
#     messages = state.get("messages", [])
    
#     # 검색 쿼리가 없는 경우 생성
#     if not search_query:
#         # 스타트업 정보에서 검색 쿼리 자동 생성
#         company_name = startup_info.get("name", "")
#         industry = startup_info.get("industry", "")
        
#         search_query = f"{company_name} {industry} market size growth trends analysis"
    
#     try:
#         # 웹 검색 도구 초기화
#         tavily_tool = TavilySearchResults(
#             max_results=5,
#             api_key=os.environ.get("TAVILY_API_KEY")
#         )
        
#         # 검색 실행
#         search_results = tavily_tool.invoke({
#             "query": search_query,
#             "max_results": 5
#         })
        
#         # 결과 포맷팅
#         formatted_results = "## 시장 자료 검색 결과\n\n"
#         for i, result in enumerate(search_results, 1):
#             formatted_results += f"### 결과 {i}\n"
#             formatted_results += f"**제목**: {result.get('title', '제목 없음')}\n"
#             formatted_results += f"**URL**: {result.get('url', 'URL 없음')}\n"
#             formatted_results += f"**내용**: {result.get('content', '내용 없음')}\n\n"
        
#         # 검색 결과 메시지 추가
#         result_message = AIMessage(content=formatted_results)
        
#         # 상태 업데이트 및 반환
#         return {
#             "startup_info": startup_info,
#             "search_query": search_query,
#             "search_result": formatted_results,
#             "messages": messages + [result_message]
#         }
    
#     except Exception as e:
#         error_message = f"시장 데이터 검색 중 오류 발생: {str(e)}"
#         return {
#             "startup_info": startup_info,
#             "search_query": search_query,
#             "error": error_message,
#             "messages": messages + [AIMessage(content=error_message)]
#         }
    
# # 시장 분석 결과 생성
# def generate_market_analysis(state: MarketAssessmentState) -> MarketAssessmentState:
#     # 필요한 정보 추출
#     startup_info = state.get("startup_info", {})
#     search_result = state.get("search_result", "")
#     messages = state.get("messages", [])
    
#     if not search_result and messages:
#         # 마지막 메시지에서 검색 결과 추출 시도
#         search_result = messages[-1].content if hasattr(messages[-1], "content") else ""
    
#     # LLM 모델 초기화
#     model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
#     # 시장 분석 프롬프트
#     analysis_prompt = """당신은 스타트업 시장성 평가 전문가입니다. 수집된 정보를 바탕으로 해당 스타트업의 시장성에 대한 종합 분석을 제공해주세요.

#     스타트업 정보:
#     {startup_info}

#     수집된 시장 정보:
#     {search_result}

#     다음 항목에 대해 체계적으로 분석해주세요:

#     1. 시장 규모 분석 (TAM, SAM, SOM)
#     - TAM(Total Addressable Market): 해당 제품/서비스가 도달 가능한 최대 시장 규모
#     - SAM(Serviceable Available Market): 회사의 비즈니스 모델로 실제 타겟팅할 수 있는 시장 규모
#     - SOM(Serviceable Obtainable Market): 합리적으로 획득 가능한 시장 점유율

#     2. 시장 성장률 및 트렌드
#     - 현재 시장 성장률
#     - 향후 3-5년간 예상 성장률
#     - 주요 시장 트렌드 및 변화 요인

#     3. 잠재수요 분석
#     - 주요 고객 세그먼트별 수요 분석
#     - 지역별 수요 분석
#     - 현재 및 잠재 고객의 요구사항과 구매 동인

#     4. 시장 진입 장벽 분석
#     - 규제 환경
#     - 기술적 장벽
#     - 경쟁 상황

#     5. 종합 결론
#     * **위 1~4번 분석 결과를 바탕으로** 스타트업의 시장성에 대한 구체적인 결론을 도출해주세요.
#     * 잠재 수요 분석이 스타트업의 성장 가능성을 어떻게 뒷받침하는지 설명하세요.
#     * 최종적으로, 분석된 기회와 위험을 고려하여 스타트업의 **시장성**에 대한 합리적인 판단을 제시하세요.

    
#     분석은 가능한 한 정량적 데이터와 근거를 포함해야 합니다. 데이터 출처도 함께 명시해주세요."""
        
#     # 프롬프트 포맷팅
#     formatted_prompt = analysis_prompt.format(
#         startup_info=startup_info,
#         search_result=search_result
#     )
    
#     # 시장 분석 생성
#     response = model.invoke([HumanMessage(content=formatted_prompt)])
#     market_analysis = response.content
    
#     # 분석 결과 정리
#     analysis_results = {
#         "market_analysis": market_analysis
#     }
    
#     # 상태 업데이트 및 반환
#     return {
#         "startup_info": startup_info,
#         "search_result": search_result,
#         "analysis_results": analysis_results,
#         "messages": messages + [AIMessage(content=market_analysis)]
#     }

# # 시장성 평가 Agent 그래프
# def create_market_assessment_graph():
#     # 그래프 초기화
#     workflow = StateGraph(MarketAssessmentState)
    
#     # 노드 추가
#     workflow.add_node("agent", agent)
#     workflow.add_node("search_market_data", search_market_data)
#     workflow.add_node("rewrite_query", rewrite_query)
#     workflow.add_node("generate_market_analysis", generate_market_analysis)
    
#     # 엣지 추가
#     workflow.add_edge(START, "agent")
#     workflow.add_edge("agent", "search_market_data")
    
#     # 조건부 엣지 추가 (검색 결과 관련성 평가)
#     workflow.add_conditional_edges(
#         "search_market_data",
#         grade_documents,
#         {
#             "generate": "generate_market_analysis",
#             "rewrite": "rewrite_query"
#         }
#     )
    
#     workflow.add_edge("rewrite_query", "search_market_data")
#     workflow.add_edge("generate_market_analysis", END)
    
#     # 그래프 컴파일
#     return workflow.compile()

# # 시장성 평가 실행
# def run_market_assessment(startup_info):
#     """시장성 평가 실행 함수"""
#     # 도구 설정
#     global tools
#     tools = create_tools()
    
#     # 그래프 생성
#     market_assessment_graph = create_market_assessment_graph()
    
#     # 초기 상태 설정
#     initial_state = {
#         "startup_info": startup_info,
#         "messages": [HumanMessage(content=f"다음 스타트업의 시장성을 평가해주세요: {startup_info}")],
#         "market_data": {},
#         "analysis_results": {},
#         "error": "",
#         "search_query": "",
#         "search_result": ""
#     }
    
#     # 그래프 실행
#     result = market_assessment_graph.invoke(initial_state)
    
#     # 결과 반환
#     return result