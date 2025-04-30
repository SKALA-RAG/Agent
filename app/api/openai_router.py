from typing import Union
from app.agents.startup_explorer_agent import StartupExplorerAgent
from fastapi import APIRouter
from app.agents.competitor_compare_agent import compare_competitors
from fastapi import APIRouter

from app.agents.open_ai import get_streaming_message_from_openai
from app.agents.info_perform_agent import get_info_perform
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


router = APIRouter()

class AskRequest(BaseModel):
    data: str

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
    
    return startup_data
