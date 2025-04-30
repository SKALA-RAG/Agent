#skala 12조 AI Agent RAG 실습

## AI Startup Investment Evaluation Framework

빠르게 변화하는 AI 산업에서 글로벌 스타트업을 탐색하고 객관적 기준에 따라 투자 여부를 판단하는 시스템을 구축하고자 함 

### 🧭 Project Goal

- **목표**: 여러 관점에서 AI 스타트업을 통합 분석하여 투자 적합성을 자동으로 판단
- **접근법**: LangGraph 기반의 멀티 에이전트 아키텍처 + Tavily 검색 및 문서 기반 RAG

### Overview

- PDF 자료 기반 정보 추출 (IR 자료, 기사 등)
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
| **Tech Summary Agent**         | IR 문서, 기술 백서, 보도자료 등에서 기술 역량을 요약 (RAG + 3개 문서 기준) |
| **Market Insight Agent**       | 시장 성장성, 유사 기업 동향, 투자 유치 현황 등 외부 검색 기반 시장성 분석 (Tavily 활용) |
| **Competitor Analysis Agent**  | 관련 5개 경쟁사 추출 및 비교 분석, 차별화 포인트 도출 |
| **Supervisor Agent**           | 개별 Agent의 분석 결과를 통합하여 투자 판단용 정리 정보 생성 |
| **Investment Decision Agent**  | 글로벌 유사 기업의 수치 데이터를 기반으로 점수화 (매출, 유저 수, 계약 등 / RAG + Tavily) |
| **Report Generator Agent**     | 모든 평가 결과를 바탕으로 최종 투자 검토 보고서 자동 생성 |



### Architecture


### Directory Structure
├── data/                  # 스타트업 PDF 문서
├── agents/                # 평가 기준별 Agent 모듈
├── prompts/               # 프롬프트 템플릿
├── outputs/               # 평가 결과 저장
├── app.py                 # 실행 스크립트
└── README.md

### Contributors
- 김세은:
- 부승호:
- 여다건:
- 조민서:
- 황유정:



