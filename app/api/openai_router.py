from typing import Union
from app.agents.startup_explorer_agent import StartupExplorerAgent
from app.agents.invest_agent import get_invest_judgement
from app.agents.competitor_compare_agent import compare_competitors
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.agents.open_ai import get_streaming_message_from_openai
from app.agents.info_perform_agent import get_info_perform
from app.agents.generate_report_agent import convert_report_to_pdf
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
import logging 
from io import BytesIO # BytesIO 추가

latest_report = None

router = APIRouter()

class AskRequest(BaseModel):
    data: str

class ReportRequest(BaseModel):
    report_text: str

@router.post(
    "/ask",
    summary="Generate messages using OpenAI (streaming)",
)
async def get_generated_messages_with_header(
    request: AskRequest
):
    """
    ## Open AI에 질문하기
    - data: 질문 내용
    - return: OpenAI의 응답을 스트리밍 방식으로 반환
    """
    
    return StreamingResponse(
        get_streaming_message_from_openai(request.data), 
        media_type="text/event-stream"
    )

@router.post(
    "/info_perform",
    summary="Generate messages using OpenAI (streaming)",
)
async def get_generated_messages_with_header(
    request: AskRequest
):
    """
    ## Tavily로 기업 실적 및 창업자 정보 검색
    - data: 기업 정보 템플릿
    - return: 기업 실적 및 창업자 정보 요약
    """
    
    return await get_info_perform(request.data)

@router.post(
    "/competitor_compare",
    summary="Generate competitor analysis using OpenAI and Tavily",
)
async def get_competitor_analysis(
    request: AskRequest
):
    """
    ## Tavily로 경쟁사 비교 분석
    - data: 기업 정보 템플릿
    - return: 경쟁사 목록 및 비교 분석 요약
    """
    return await compare_competitors(request.data)

@router.post(
    "/invest",
    summary="invest",
)
async def get_invest_analysis(
    request: AskRequest
):
    return await get_invest_analysis(request.data)
  
@router.get(
    "/explore_startup",
    summary="Generate messages using OpenAI (streaming)",
)
async def get_startup_info():
    """
    # Tavily로 스타트업 자료 검색하여 투자 검토 회사 선정
    """
    
    explorer = StartupExplorerAgent()
    startup_data = await explorer.supervisor()
    
    global latest_report
    latest_report = startup_data[5]  # 튜플의 6번째 요소(인덱스 5)가 최종 보고서

    return startup_data

@router.post(
    "/download_report",
    summary="Download the generated report as a PDF file",
    response_class=Response
)
async def download_report_pdf(
    request: ReportRequest
):
    """
    ## 투자 보고서를 PDF로 다운로드
    - report_text: 보고서 내용(텍스트)
    - return: PDF 파일
    """
    try:
        report_text = request.report_text
        
        # 입력된 report_text가 비어있거나 너무 짧은 경우 최신 보고서 사용
        if not report_text or len(report_text.strip()) < 50:
            global latest_report
            if latest_report is not None:
                logging.info("입력된 보고서 내용이 없거나 너무 짧아 최근 생성된 보고서를 사용합니다.")
                report_text = latest_report
            else:
                return JSONResponse(
                    content={"error": "보고서 내용이 없습니다. 먼저 /explore_startup을 실행하거나 report_text를 제공하세요."}, 
                    status_code=400
                )
        
        pdf_bytes = convert_report_to_pdf(report_text)
        filename = "startup_investment_report.pdf"

        headers = {
            'Content-Disposition': f'attachment; filename="{filename}"'
        }

        return Response(content=pdf_bytes, media_type='application/pdf', headers=headers)

    except Exception as e:
        logging.error(f"PDF 변환 중 오류 발생: {e}", exc_info=True)
        return JSONResponse(
            content={"error": f"PDF 변환 중 오류가 발생했습니다: {str(e)}"}, 
            status_code=500
        )