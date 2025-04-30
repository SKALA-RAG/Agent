from app.agents.competitor_compare_agent import compare_competitors
from app.agents.info_perform_agent import get_info_perform
from app.agents.market_agent import assess_market_potential
from app.agents.vectorize_papers_agent import get_tech_summary
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
import os
import random

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
CHAT_MODEL = "gpt-4o-mini"


class StartupExplorerAgent:
    """
    AI 스타트업을 탐색하고 투자 가능성이 있는 스타트업을 선정하여 정보를 제공하는 에이전트

    기능:
    1. 국내외 유명 AI 스타트업 검색 및 조사
    2. 투자 가치가 있는 스타트업 선정
    3. 정형화된 형식으로 스타트업 정보 제공
    """

    def __init__(self):
        """
        에이전트 초기화
        """
        self.startup_data = ""
        self.selected_startups = []
        self.search_tool = TavilySearchResults(max_results=20, api_key=TAVILY_API_KEY)
        self.llm = ChatOpenAI(model=CHAT_MODEL)
        self.search_results = []
        self.startup_name = ""
        self.found_startups = []

    def search_startups(self) -> List[Dict[str, Any]]:
        """
        스타트업 검색을 수행

        Returns:
            검색된 스타트업
        """
        # 검색 쿼리 조정
        search_query = self._generate_rich_startup_query()

        print(f"검색 쿼리: {search_query}")

        # 실제 검색 수행
        self.search_results = self._perform_web_search(search_query)

        return self.search_results

    def _generate_rich_startup_query(self) -> str:
        openings = [
            "혁신적인 인공지능 스타트업을 조사 중입니다.",
            "기술력과 성장성을 갖춘 AI 스타트업을 찾고 있습니다.",
            "다음 투자를 위한 유망한 AI 기반 스타트업 정보를 수집하고 있습니다.",
            "AI 기술을 주축으로 빠르게 성장 중인 스타트업을 찾습니다.",
        ]

        conditions = [
            "- 최근 3년(2021~2024년) 이내에 설립된 스타트업",
            "- 생성형 AI(Generative AI), 머신러닝(ML), 자연어 처리(NLP) 기술 활용",
            "- 독자적 알고리즘 또는 딥러닝 모델을 개발 및 보유",
            "- 기술력을 기반으로 한 SaaS 또는 플랫폼형 서비스 제공",
            "- 유의미한 기술 검증(PoC), 상용화 또는 파일럿 테스트 진행 경험",
            "- AI 관련 특허 또는 연구 논문이 존재",
            "- 소수 정예로 구성된 고급 기술 인력 보유",
            "- 최근 1년 내 외부 투자 유치 경험 있음 (시드~시리즈 C)",
            "- 정부 또는 글로벌 프로그램(R&D 지원, 스타트업 육성 등) 참여 이력 있음",
            "- AI를 기존 산업(예: 농업, 제조, 물류 등)에 접목한 융합형 모델 운영",
            "- 글로벌 진출을 위한 다국어 서비스 또는 해외 진출 계획 명시",
        ]

        extras = [
            "산업군은 제한 없으나, 헬스케어, 로보틱스, 에듀테크, 핀테크, 제조 자동화 분야면 더욱 좋습니다.",
            "국내외를 불문하고, 글로벌 진출 가능성이 있는 스타트업이면 더욱 좋습니다.",
            "기술 데모, PoC 사례, 기업 블로그 또는 미디어에 언급된 기사를 포함해 주세요.",
            "투자자(VC), 액셀러레이터, 인큐베이터의 평가 사례가 있는 자료가 있으면 포함해주세요.",
        ]

        selected_conditions = random.sample(conditions, 3)

        query = (
            f"{random.choice(openings)}\n"
            + "\n".join(selected_conditions)
            + "\n"
            + f"{random.choice(extras)}"
        )

        return query

    def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """
        LangChain TavilySearchResults 도구를 이용한 검색 결과 반환
        """
        try:
            self.search_results = self.search_tool.invoke(query)

            return self.search_results

        except Exception as e:
            print(f"Tavily 검색 도구 실행 중 오류 발생: {str(e)}")
            return []

    def select_startup_from_search_results(self) -> str:
        tools = [self.search_tool]
        # 1. 프롬프트 템플릿 설정
        prompt_template = PromptTemplate.from_template(
            """
            당신은 AI 분야의 유망한 스타트업을 탐색하는 투자 분석가입니다.
            현재 투자 여부를 판단할 기업을 선정하는 역할을 맡고 있습니다.
            동일한 패턴의 선택을 반복하지 않기 위해, 기존과 **다른 유형의 스타트업**을 추천하는 것이 중요합니다.
            다음 검색 결과를 바탕으로, 이미 선택한 기업 리스트({list})에 없는 기업 중:
            - **기술이나 비즈니스 모델에서 새로운 관점**을 제시하거나,
            - **특정 AI 응용 분야에서 독창적인 접근**을 한 기업을 선정해주세요.
            - 지역, 산업, 기술 분야에서 **다양성을 고려**하세요.
            
            *단, 너무 일반적이거나 정보가 부족한 회사는 제외해주세요.*
            
            다음의 검색 결과를 활용해서 투자 평가를 진행할 스타트업 *1곳*을 선정하여 *회사명만* 출력해주세요.
            꼭 회사의 이름을 출력해야합니다.
          
            [검색 결과]
            {context}
            """
        )

        # 3. LLM Chain 생성
        chain = prompt_template | self.llm | StrOutputParser()

        # 4. 질문에 대한 답변 생성
        response = chain.invoke(
            {"context": self.search_results, "list": self.found_startups}
        )

        return response.strip()

    def collect_detailed_info(self) -> str:
        """
        회사명을 기반으로 Tavily 검색 후, LLM으로 요약된 회사 정보 반환

        Returns:
            str: 요약된 회사 정보
        """
        try:
            # 1. Tavily 검색
            query = f"{self.startup_name} 회사 및 대표자 정보, 설립연도, 주요 AI 기술, 투자 현황, 최근 뉴스"
            search_result = self.search_tool.invoke(query)
            tools = [self.search_tool]

            # 2. 프롬프트 정의
            prompt = ChatPromptTemplate.from_messages(
                messages=[
                    (
                        "system",
                        """
                아래는 '{company}'에 대한 검색 결과입니다. 이 정보를 바탕으로 회사 정보를 정리해 주세요.
                필요한 경우 {tools}를 사용해서 추가 검색을 진행하여 최대한 정확하게 답변하세요.

                - 회사명
                - 설립연도
                - 대표자
                - 주요 AI 기술 또는 제품 및 회사 홈페이지
                - 투자 현황 또는 주요 투자자
                - 핵심 인력
                - 주목할 만한 뉴스 또는 최근 성과

                [검색 결과]
                {context}

                답변은 아래의 형식으로 반환해주세요.
                아래와 같이 개조식으로 작성하되, 볼드체 등 글 꾸밈 요소는 **절대** 포함하지 마세요!
                만약 해당 정보를 정확하게 찾을 수 없는 경우 "찾을 수 없음"이라고 표시해주세요.
                1. 회사명
                2. 설립일
                3. 대표자
                4. 주요 사업 분야
                5. 홈페이지
                6. 연락처
                7. 핵심 인력 (팀 구성)
                8. 주요 연혁 (설립, 투자유치, 특허 등 중요 이력)
                """,
                    ),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )

            # 3. LLMChain 실행
            agent = create_tool_calling_agent(self.llm, tools, prompt)

            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            inputs = {
                "company": self.startup_name,
                "context": search_result,
                "tools": tools,
            }
            summary = agent_executor.invoke(inputs)

            print(f"기업정보: {summary['output'].strip()}")

            return summary["output"].strip()

        except Exception as e:
            return f"[오류 발생] {str(e)}"

    async def run_exploration_pipeline(self) -> List[str]:
        """
        스타트업 탐색 전체 파이프라인 실행

        Returns:
            포맷팅된 스타트업 정보
        """
        print(f"=== 스타트업 탐색 시작 ===")

        # 1. 스타트업 검색
        print("1. 스타트업 검색 중...")
        self.search_startups()
        print(f"자료 검색 완료")

        # 2. 검색된 정보를 바탕으로 회사 선정
        print("2. 회사 선정 중...")
        self.startup_name = self.select_startup_from_search_results()
        self.found_startups.append(self.startup_name)
        print(f"회사 이름: {self.startup_name}")

        # 3. 회사 상세 정보 수집
        print("3. 회사 상세 정보 수집 중...")
        self.startup_data = self.collect_detailed_info()
        print(f"{self.startup_name}의 상세정보 검색 완료")

        print("=== 스타트업 탐색 완료 ===")

        return self.startup_data

    async def supervisor(self):
        await self.run_exploration_pipeline()

        exploration_result = {"기업 정보 요약": self.startup_data}

        # 창업자 정보 실적 반환
        perform_info = await get_info_perform(self.startup_data)

        # 경쟁사 비교 분석 반환
        competiter_info = await compare_competitors(self.startup_data)

        # 시장 비교 분석 반환
        market_info = await assess_market_potential(self.startup_data)

        # 기술 요약 반환
        tech_info = await get_tech_summary(self.startup_data)

        return exploration_result, perform_info, competiter_info, market_info, tech_info
