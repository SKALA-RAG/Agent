#skala 12조 AI Agent RAG 실습

## AI Startup Investment Evaluation Framework

빠르게 변화하는 AI 산업에서 글로벌 스타트업을 탐색하고 객관적 기준에 따라 투자 여부를 판단하는 시스템을 구축하고자 함 

### 🧭 Project Goal

- **목표**: 여러 관점에서 AI 스타트업을 통합 분석하여 투자 적합성을 자동으로 판단
- **접근법**: LangGraph 기반의 멀티 에이전트 아키텍처 + Tavily 검색 및 문서 기반 RAG

### Overview

- PDF 자료 기반 정보 추출 (특허 및 AI 관련 논문 자료, 기사 등)
- 투자 기준별 점수 산정 (Scorecard Method 활용)
- 투자 판단 및 종합 투자 요약 출력
  
### Tech Stack

| Category   | Details                      |
|------------|------------------------------|
| Framework  | LangGraph, LangChain, Python |
| LLM        | GPT-4o-mini via OpenAI API   |
| Retrieval  | Chroma DB                    |
| Search     | Tavily Tool                  |


### 📊 평가 방식 및 Scorecard 설계

- **Scorecard 시스템**:
  - 글로벌 평균 수치를 기반으로 등급화
  - 각 평가 항목별 점수 산정 → 가중 평균 계산
  - 기준표를 기반으로 '투자 / 보류' 판단

> **Note**: 보류 판단 시, 탐색 단계부터 재분석하여 정보를 갱신



### 🧠 Agent Responsibilities

| Agent 명칭                     | 기능 설명 |
|--------------------------------|-----------|
| **Startup Explorer Agent**     | 웹에서 스타트업 관련 기본 정보, 핵심 제품, 활동 현황 등을 수집 |
| **Market Insight Agent**       | 시장 성장성, 유사 기업 동향, 투자 유치 현황 등 외부 검색 기반 시장성 분석 (Tavily 활용) |
| **Competitor Analysis Agent**  | 관련 5개 경쟁사 추출 및 비교 분석, 차별화 포인트 도출 |
| **Supervisor Agent**           | 개별 Agent의 분석 결과를 통합하여 투자 판단용 정리 정보 생성 |
| **Investment Decision Agent**  | 글로벌 유사 기업의 수치 데이터를 기반으로 점수화 (매출, 유저 수, 계약 등 / RAG + Tavily) |
| **Report Generator Agent**     | 모든 평가 결과를 바탕으로 최종 투자 검토 보고서 자동 생성 |
| **tech_summary_agent**         | 특허 기반으로 벡터DB에서 논문 내용 추출 및 두 가지(논문 + 특허) 내용 활용을 통한 기술 요약 생성 |
| **vectorize_papers_agent**     | AI 관련 논문을 바탕으로 vectorDB 구축 |

### Agent Overview with RAG Usage 

| 에이전트               | 역할                              | RAG 여부 |
|------------------------|-----------------------------------|----------|
| 🔍스타트업 탐색 에이전트     | 유명한 AI 스타트업 정보 수집           | X        |
| 🗜️기술 요약 에이전트        | 스타트업의 기술력 핵심 요약           | O        |
| 📊시장성 평가 에이전트      | 시장 성장성, 수요 분석               | X        |
| 🥊경쟁사 비교 에이전트      | 경쟁 구도, 차별성 분석               | X        |
| 🧮투자 판단 에이전트        | 종합 판단 (리스트, ROI 등)          | O        |
| 📝보고서 생성 에이전트      | 결과 요약 보고서 생성               | X        |



### Directory Structure
```
Agent/
├── app/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── competitor_compare_agent.py
│   │   ├── info_perform_agent.py
│   │   ├── market_agent.py
│   │   ├── invest_agent.py
│   │   ├── generate_report_agent.py
│   │   ├── open_ai.py
│   │   ├── invest_db.py
│   │   ├── vectorize_papers_agent.py
│   │   ├── tech_summary_agent.py
│   │   └── startup_explorer_agent.py
│   ├── graph/
│   │   ├── investment_graph.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── openai_router.py
│   ├── static/
        ├── home.html
│   └── main.py
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

### langgraph
시도중..........

### Contributors
- 김세은: 창업자 및 실적 분석 에이전트 구축, 투자 판단 에이전트 구축 (RAG 구축 및 Tavily 활용)
- 부승호: 경쟁자 비교 에이전트 구축 (tavily 활용), lang graph 시도중 
- 여다건: 시장성 분석 에이전트 구축, 보고서 작성 에이전트 구축 (tavily 활용)
- 조민서: 스타트업 탐색 및 선정 에이전트 구축(tavily 활용), lang graph 시도중
- 황유정: 기술 요약 에이전트 구축 (3가지 문서 기반 RAG 구축 및 Tavily 활용)



