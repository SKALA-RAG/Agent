from langchain_teddynote.tools.tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

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
        self.search_tool = TavilySearch()
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
        search_query = (
          "아래 조건에 따라 스타트업 1곳을 선정하기 위한 자료를 검색해주세요"
          "다양한 산업 분야와 지역을 고려하여 AI 기술(생성 AI, 머신러닝, 자연어 처리 등)을 활용하는 스타트업을 검색해주세요."

          "조건:"
          "- 최근 3년 이내 설립"
          "- 다양한 산업 분야와 지역 포함"
          "- AI 기술(생성 AI, 머신러닝, 자연어 처리 등)을 핵심 기술로 활용"
          "- 독창성, 성장 가능성, 기술력이 뛰어난 곳"
        )
        
        print(f"검색 쿼리: {search_query}")
        
        # 실제 검색 수행
        self.search_results = self._perform_web_search(search_query)
        
        return self.search_results
    
    def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """
        LangChain TavilySearchResults 도구를 이용한 검색 결과 반환
        """
        try:
            # LangChain Tool의 run() 호출
            self.search_results = self.search_tool.search(query = query)

            return self.search_results

        except Exception as e:
            print(f"Tavily 검색 도구 실행 중 오류 발생: {str(e)}")
            return []
          
    def select_startup_from_search_results(self) -> str:
        # 1. 프롬프트 템플릿 설정
        prompt_template = PromptTemplate.from_template(
            """
            당신은 AI 분야의 유망한 스타트업을 탐색하는 투자 분석가입니다.
            현재 투자 여부를 판단할 기업을 선정하는 역할을 맡고 있습니다.
            다음의 검색 결과를 활용해서 투자 평가를 진행할 스타트업 *1곳*을 선정하여 *회사명만* 출력해주세요.
            이미 선정했던 기업인 {list}에 있는 기업은 제외해주세요.

            [검색 결과]
            {context}
            """
        )

        # 3. LLM Chain 생성
        chain = prompt_template | self.llm | StrOutputParser()

        # 4. 질문에 대한 답변 생성
        response = chain.invoke({
            "context": self.search_results,
            "list": self.found_startups
        })

        return response.strip()
    
    def collect_detailed_info(self) -> str:
          """
          회사명을 기반으로 Tavily 검색 후, LLM으로 요약된 회사 정보 반환

          Returns:
              str: 요약된 회사 정보
          """
          try:
            # 1. Tavily 검색
            query = f"{self.startup_name} 회사 정보, 설립연도, 주요 AI 기술, 투자 현황, 최근 뉴스"
            search_result = self.search_tool.search(query=query, format_output=True)

            # 2. 프롬프트 정의
            prompt = PromptTemplate.from_template(
                """
                아래는 '{company}'에 대한 검색 결과입니다. 이 정보를 바탕으로 회사 정보를 정리해 주세요.

                - 회사명
                - 설립연도
                - 위치(가능하다면)
                - 주요 AI 기술 또는 제품
                - 투자 현황 또는 주요 투자자
                - 주목할 만한 뉴스 또는 최근 성과

                [검색 결과]
                {context}

                답변은 아래의 형식으로 반환해주세요.
                아래와 같이 개조식으로 작성하되, 볼드체 등 글 꾸밈 요소는 **절대** 포함하지 마세요!
                1. 회사명
                2. 설립일
                3. 대표자
                4. 주요 사업 분야
                5. 홈페이지
                6. 연락처
                7. 핵심 인력 (팀 구성)
                8. 주요 연혁 (설립, 투자유치, 특허 등 중요 이력)
                """
            )

            # 3. LLMChain 실행
            chain = prompt | self.llm | StrOutputParser()
            summary = chain.invoke({
                "company": self.startup_name,
                "context": search_result
            })

            print(f"기업정보: {summary.strip()}")


            return summary.strip()

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