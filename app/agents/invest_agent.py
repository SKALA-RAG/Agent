import logging
import os
from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda

load_dotenv()

# 1. 임베딩 모델을 불러오고
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. 저장된 벡터 DB를 불러오기
vector_db = Chroma(
    persist_directory="./invest_db",  # 저장된 디렉토리
    embedding_function=embeddings     # 같은 임베딩 모델로!
)

def get_industry_baseline(query: str, k=1):
    results = vector_db.similarity_search(query, k=k)
    return "\n".join([r.page_content for r in results])

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.1, 
    openai_api_key=OPENAI_API_KEY
)

# 프롬프트 템플릿
prompt = PromptTemplate.from_template(
    """
    다음은 AI 스타트업의 업계 평균 벤치마크 정보입니다:

    {baseline}

    위 정보를 바탕으로, 아래 회사의 실적과 비교해 평가하세요.
    평가 항목은 다음과 같습니다:

    | 항목 | 비중(%) | 평가 포인트 |
    | --- | --- | --- |
    | 창업자 (Owner) | 30% | 전문성, 커뮤니케이션, 실행력 |
    | 시장성 (Opportunity Size) | 25% | 시장 크기, 성장 가능성  |
    | 제품/기술력 | 15% | 독창성, 구현 가능성 |
    | 경쟁 우위 | 10% | 진입장벽, 특허, 네트워크 효과 |
    | 실적 | 10% | 매출, 계약, 유저수 등 |
    | 투자조건 (Deal Terms) | 10% | Valuation, 지분율 등  |

    회사 정보:
    {text}

    위의 평가 기준에 따라 전체적으로 투자 관점에서 이 회사를 평가해줘. 표 형식은 사용하지 말고 문장으로 정리하고, 각 평가 항목의 점수는 평균을 100%로 해서 업계 평균보다 높다고 판단되면 100% 이상, 낮다고 판단되면 100% 미만으로 설정해줘. 각 항목을 언급하면서 평가 내용을 제시해줘.
    """
)

output_parser = StrOutputParser()

chain = (
    RunnableLambda(lambda x: {
        "baseline": get_industry_baseline("2025년 AI 스타트업 업계 평균 Revenue Multiple"),
        "text": x["text"]
    }) 
    | prompt
    | model
    | output_parser
)

def format_input_for_invest_judgement(blocks: list[dict]) -> str:
    lines = []
    for block in blocks:
        for key, value in block.items():
            lines.append(f"## {key.strip()}\n{value.strip()}\n")
    return "\n".join(lines).strip()


async def get_invest_judgement(data: str):

    try:
        data = format_input_for_invest_judgement(data)

        logging.info(f"Received data: {data}")

        invest_judge = chain.invoke({"text": data})

        logging.info(f"Judge Result: {invest_judge}")

        final_result = {
            "투자 판단 보고서": invest_judge
        }

        return final_result

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI or Search Error: {str(e)}")

