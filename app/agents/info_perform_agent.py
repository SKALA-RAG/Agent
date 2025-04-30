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

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

model = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.1, 
    openai_api_key=OPENAI_API_KEY
)
tavily = TavilySearchResults(max_results=20)

# 프롬프트 템플릿
company_prompt = PromptTemplate.from_template(
    "다음 웹 검색 자료를 읽고, 해당 회사의 실적 (투자유치, 매출, 수상 등)을 요약해줘. 개조식으로 작성하지 말고, 문장 형태로 작성해줘.\n\n{text}"
)
founder_prompt = PromptTemplate.from_template(
    "다음 웹 검색 자료를 읽고, 창업자의 주요 경력과 학력을 요약해줘. 개조식으로 작성하지 말고, 문장 형태로 작성해줘.\n\n{text}"
)

output_parser = StrOutputParser()

company_chain = company_prompt | model | output_parser
founder_chain = founder_prompt | model | output_parser

async def get_info_perform(data: str):
    company_name, ceo_name = extract_company_and_ceo(data)

    try:
        # 1. Tavily로 기업 실적 검색
        company_query = f"{company_name} 투자유치 매출 수상 실적"
        company_search_result = tavily.invoke(company_query)
        company_summary = company_chain.invoke({"text": company_search_result})

        logging.info(f"Company Search Result: {company_summary}")

        # 2. Tavily로 창업자 정보 검색
        if ceo_name is None or "찾을 수 없음" in ceo_name:
            logging.info("CEO name not found in the response. Using default CEO name.")
            ceo_name = ""
        # founder_query = f"{company_name} 창업자 {ceo_name}의 학력 경력 창업 이력"
        founder_query = f"{company_name} 창업자 {ceo_name} 경력 이력"
        founder_search_result = tavily.invoke(founder_query)
        founder_summary = founder_chain.invoke({"text": founder_search_result})

        logging.info(f"Founder Search Result: {founder_summary}")

        # 3. 결과 합치기
        final_result = {
            "기업 실적 요약": company_summary,
            "창업자 정보 요약": founder_summary
        }

        return final_result

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI or Search Error: {str(e)}")
    
def extract_company_and_ceo(llm_response: str):
    company_pattern = r"회사명\s*[:：]\s*(.*)"
    ceo_pattern = r"대표자\s*[:：]\s*(.*)"

    company_match = re.search(company_pattern, llm_response)
    ceo_match = re.search(ceo_pattern, llm_response)

    company = company_match.group(1).strip() if company_match else None
    ceo = ceo_match.group(1).strip() if ceo_match else None

    logging.info(f"company: {company}, ceo: {ceo}")
    return company, ceo
