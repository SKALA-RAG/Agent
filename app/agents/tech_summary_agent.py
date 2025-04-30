import os
import requests
import logging
import xml.etree.ElementTree as ET
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv


# 환경 변수 로드
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(
    level=logging.INFO,  # INFO 이상의 로그만 출력
    format="%(asctime)s - %(levelname)s - %(message)s",  # 시간 + 레벨 + 메시지
)


# 회사 특허 전체 가져오기
def fetch_patents(applicant_name):
    url = "http://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getAdvancedSearch"
    params = {
        "ServiceKey": os.getenv("KIPRIS_API_KEY"),
        "applicant": applicant_name,
        "pageNo": 1,
        "numOfRows": 100,
        "sortSpec": "PD",
        "descSort": False,
    }
    response = requests.get(url, params=params)
    root = ET.fromstring(response.text)

    patents = []
    for item in root.findall(".//item"):
        patent = {
            "출원인": item.findtext("applicantName", default=""),
            "출원일자": item.findtext("applicationDate", default=""),
            "출원번호": item.findtext("applicationNumber", default=""),
            "초록": item.findtext("astrtCont", default=""),
            "발명의명칭": item.findtext("inventionTitle", default=""),
            "IPC번호": item.findtext("ipcNumber", default=""),
            "공개일자": item.findtext("openDate", default=""),
            "공개번호": item.findtext("openNumber", default=""),
            "등록상태": item.findtext("registerStatus", default=""),
        }
        patents.append(patent)
    return patents


# 통합 키워드 추출
def extract_keywords_from_patents(patents, top_n=5):
    combined_text = ""
    for p in patents:
        combined_text += f"제목: {p['발명의명칭']}\n초록: {p['초록']}\n\n"

    prompt = f"""
다음은 한 회사의 전체 특허 목록입니다. 이 특허들의 기술 내용을 바탕으로 핵심 기술 키워드 {top_n}개를 뽑아 주세요.
키워드는 한 줄에 하나씩, 명확한 기술 용어로 작성해 주세요.

{combined_text}
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        keyword_text = response.choices[0].message.content.strip()
        keywords = [
            kw.strip("-•* ").strip() for kw in keyword_text.split("\n") if kw.strip()
        ]
        return keywords
    except Exception as e:
        logging.info(f"키워드 추출 오류: {e}")
        return []


# 통합 키워드로 논문 검색
def search_docs_by_combined_keywords(keywords, db_path, top_k=5):
    vectorstore = Chroma(
        persist_directory=db_path, embedding_function=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )

    combined_query = " ".join(keywords)
    try:
        docs = retriever.invoke(combined_query)
        return docs
    except Exception as e:
        logging.info(f"논문 검색 실패: {e}")
        return []


# 특허 리스트 변환 함수
def convert_kipris_patents_to_llm_ready(patents):
    converted = []
    for p in patents:
        title = p.get("발명의명칭", "").strip()
        summary = p.get("초록", "").strip()
        if title and summary:
            converted.append({"title": title, "summary": summary})
    return converted


def tech_summary(company_name, db_path):
    patents = fetch_patents(company_name)
    # 변환 적용
    converted_patents = convert_kipris_patents_to_llm_ready(patents)

    if not patents:
        logging.info("특허 없음. 회사명을 다시 확인하세요.")
        return

    logging.info(f"총 {len(patents)}건의 특허 수집됨")

    # 키워드 추출
    keywords = extract_keywords_from_patents(patents, top_n=5)

    # 초록 통합
    abstract_text = "\n".join(
        [f"- {p['발명의명칭']}:\n{p['초록']}" for p in patents if p["초록"]]
    )

    # 논문 검색
    docs = search_docs_by_combined_keywords(keywords, db_path=db_path, top_k=5)
    logging.info(f"관련 논문 검색 결과: {len(docs)}건")

    summaries = []
    for idx, doc in enumerate(docs, 1):
        content = doc.page_content.strip().replace("\n", " ")
        summaries.append(f"{idx}. {content[:500]}...")  # 요약 길이 제한

    # LLM 프롬프트 구성
    joined_summaries = "\n".join(summaries)
    # 특허 텍스트 생성
    joined_patents = "\n".join(
        [f"- 특허명: {p['title']}\n  내용: {p['summary']}" for p in converted_patents]
    )

    prompt = f"""
    당신은 기술 스타트업에 투자할지 판단해야 하는 전문가가입니다.

    아래는 스타트업 '{company_name}'의 기술적 배경을 보여주는 핵심 자료입니다:
    - 총 {len(patents)}건의 특허
    - 특허에서 추출된 주요 키워드
    - 전체 특허 초록을 통합한 내용
    - 관련 논문 요약

    이 자료를 바탕으로 다음 조건에 맞는 기술 요약 보고서를 작성해 주세요:

    목적:
    - 특허 기반 기술의 구조, 작동 방식, 응용 방안, 차별성을 모두 포함해 객관적으로 서술
    - **특허 내용을 빠짐없이 요약에 녹여내고**, 관련 논문과 함께 기술의 독창성과 실현 가능성을 뒷받침
    - 향후 확장 가능성, 응용 분야, 상용화 가능성까지 포함

    제공 정보:
    특허 키워드:
    {', '.join(keywords)}

    전체 특허 초록 통합 요약:
    {abstract_text}

    관련 논문 요약:
    {joined_summaries}

    출력 요건:
    - 전체 기술을 대표할 수 있도록 최대한 상세하고 객관적으로 작성
    - bullet 없이 하나의 논리적 단락으로 구성 (20문장 이상)
    - **보고서 하단에 아래 특허 목록을 특허명과 내용을 통해 보여줄 것 단, 내용은 특허 설명과 함께 장단점을 포함하여 한,두문장을 사용해서 요약** (요약 결과 이후 구분선으로 삽입)

    요약 결과 아래에 다음과 같은 특허 목록을 포함하세요:

    {joined_patents}
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=3000,
        )
        summary = response.choices[0].message.content.strip()

        logging.info(f"요약 결과: {summary}")
        return {
            "기술 요약": summary,
            "추출 키워드": keywords,
            "특허 개수": len(patents),
        }

    except Exception as e:
        logging.info(f"LLM 요약 실패: {e}")
        return None
