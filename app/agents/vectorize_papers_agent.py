import os
import requests
import logging
import xml.etree.ElementTree as ET
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tech_summary_agent import tech_summary

from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

logging.basicConfig(
    level=logging.INFO,  # INFO 이상의 로그만 출력
    format="%(asctime)s - %(levelname)s - %(message)s",  # 시간 + 레벨 + 메시지
)


# arXiv API를 통해 논문 메타데이터 가져오기
def fetch_arxiv_papers(query, max_results=300):
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.text


# arXiv API 응답 파싱
def parse_arxiv_response(xml_data):
    root = ET.fromstring(xml_data)
    ns = {"arxiv": "http://www.w3.org/2005/Atom"}
    entries = []

    for entry in root.findall("arxiv:entry", ns):
        title = entry.find("arxiv:title", ns).text.strip()
        summary = entry.find("arxiv:summary", ns).text.strip()
        published = entry.find("arxiv:published", ns).text.strip()
        authors = []
        for author in entry.findall("arxiv:author/arxiv:name", ns):
            authors.append(author.text.strip())

        entries.append(
            {
                "title": title,
                "summary": summary,
                "authors": ", ".join(authors),
                "published": published[:10],  # YYYY-MM-DD 형태로 추출
            }
        )

    return entries


# 논문 정보를 PDF로 변환
def create_papers_pdf(papers, filename="ai_papers_summary_2.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()

    # 사용자 정의 스타일 추가 - 'Title' 대신 'PaperTitle' 사용
    styles.add(
        ParagraphStyle(
            name="PaperTitle",  # 이름 변경: 'Title' → 'PaperTitle'
            parent=styles["Heading1"],
            fontSize=14,
        )
    )

    styles.add(
        ParagraphStyle(
            name="Authors",
            parent=styles["Normal"],
            fontSize=10,
            textColor="blue",
        )
    )

    styles.add(
        ParagraphStyle(
            name="Date",
            parent=styles["Normal"],
            fontSize=9,
            textColor="gray",
        )
    )

    # PDF 내용 구성
    content = []
    content.append(Paragraph("arXiv summary", styles["Heading1"]))
    content.append(Spacer(1, 0.2 * inch))

    for i, paper in enumerate(papers, 1):
        # 제목 - 여기도 'Title' 대신 'PaperTitle' 사용
        content.append(Paragraph(f"{i}. {paper['title']}", styles["PaperTitle"]))
        # 저자 및 발행일
        content.append(Paragraph(f"Author: {paper['authors']}", styles["Authors"]))
        content.append(Paragraph(f"Date: {paper['published']}", styles["Date"]))
        content.append(Spacer(1, 0.1 * inch))
        # 요약
        content.append(Paragraph("<b>Summary:</b>", styles["Normal"]))
        content.append(Paragraph(paper["summary"], styles["Normal"]))
        content.append(Spacer(1, 0.3 * inch))

    # PDF 생성
    doc.build(content)
    return filename


# 1. PDF에서 텍스트 추출
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


# 2. 텍스트 분할 후 벡터스토어 생성
def build_vectorstore_from_text(text, persist_dir="chroma2_db"):
    documents = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        docs, OpenAIEmbeddings(), persist_directory=persist_dir
    )

    return vectorstore


# 3. 전체 실행 함수
def summarize_company_from_pdf(pdf_path, persist_dir):
    text = extract_text_from_pdf(pdf_path)

    logging.info(f"Vector DB 구축 중 (저장 경로: {persist_dir})")
    vectorstore = build_vectorstore_from_text(text, persist_dir=persist_dir)

    return vectorstore


# 실행
if __name__ == "__main__":

    company_name = "뉴라이브"

    query = 'cat:cs.AI OR cat:stat.ML OR all:"artificial intelligence" OR all:"deep learning"'
    xml_data = fetch_arxiv_papers(query, max_results=300)
    papers = parse_arxiv_response(xml_data)

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_path = os.path.join(output_dir, "ai_papers_summary.pdf")

    logging.info(f"총 {len(papers)}개의 논문을 찾았습니다.")
    pdf_file = create_papers_pdf(papers, filename=pdf_path)

    # 벡터 DB 저장 경로 설정 및 생성
    persist_dir = os.path.join(output_dir, "vector_db")
    summarize_company_from_pdf(pdf_file, persist_dir=persist_dir)

    tech_summary(company_name, db_path=persist_dir)
