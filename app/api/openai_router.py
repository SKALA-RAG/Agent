from typing import Union
from fastapi import APIRouter, Header, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.open_ai import get_streaming_message_from_openai
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