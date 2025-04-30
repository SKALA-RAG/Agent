import logging
import os
import re
from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    openai_api_key=OPENAI_API_KEY
)

tavily = TavilySearchResults(max_results=5)

# 프롬프트 템플릿
competitor_analysis_prompt = PromptTemplate.from_template(
    """다음 웹 검색 자료를 읽고, {company_name}과(와) 경쟁하는 주요 기업들에 대한 비교 분석을 해줘.
    
    비교 항목:
    1. 사업 모델 비교
    2. 기술력 비교
    3. 시장 점유율 비교
    4. 투자 유치 현황 비교
    5. 주요 차별점
    
    개조식으로 작성하지 말고, 문장 형태로 작성해줘.
    
    검색 결과:
    {text}"""
)

competitor_list_prompt = PromptTemplate.from_template(
    """다음 웹 검색 자료를 읽고, {company_name}의 주요 경쟁사들을 최대 5개까지 리스트업해줘.
    각 경쟁사에 대해 간략한 설명과 함께 제시해줘.
    
    검색 결과:
    {text}"""
)

output_parser = StrOutputParser()

competitor_analysis_chain = competitor_analysis_prompt | model | output_parser
competitor_list_chain = competitor_list_prompt | model | output_parser

async def compare_competitors(data: str):
    company_name, _ = extract_company_info(data)
    
    try:
        # 1. 경쟁사 리스트업
        competitor_query = f"{company_name} 경쟁사 리스트 스타트업"
        competitor_search_result = tavily.invoke(competitor_query)
        competitor_list = competitor_list_chain.invoke({
            "company_name": company_name,
            "text": competitor_search_result
        })
        logging.info(f"Competitor List Result: {competitor_list}")
        
        # 2. 경쟁사 비교 분석
        competitor_analysis_query = f"{company_name} 경쟁사 비교 분석 차별점 시장점유율"
        comparison_search_result = tavily.invoke(competitor_analysis_query)
        competitor_analysis = competitor_analysis_chain.invoke({
            "company_name": company_name,
            "text": comparison_search_result
        })
        logging.info(f"Competitor Analysis Result: {competitor_analysis}")
        
        # 3. 결과 합치기
        final_result = {
            "주요 경쟁사 목록": competitor_list,
            "경쟁사 비교 분석": competitor_analysis
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