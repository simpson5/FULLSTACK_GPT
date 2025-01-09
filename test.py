# LangChain의 주요 컴포넌트들을 임포트
from langchain.chat_models import ChatOpenAI  # ChatGPT API를 사용하기 위한 클래스
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate  # Few-shot 학습을 위한 프롬프트 템플릿
from langchain.callbacks import StreamingStdOutCallbackHandler  # 스트리밍 출력을 처리하기 위한 콜백
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder  # 채팅 프롬프트 관련 클래스들
from langchain.memory import ConversationSummaryBufferMemory  # 대화 내용을 요약하여 저장하는 메모리
from langchain.schema.runnable import RunnablePassthrough  # 체인 구성을 위한 유틸리티

# ChatGPT 모델 초기화
chat = ChatOpenAI(
    temperature=0.1,  # 낮은 temperature로 일관된 응답 생성
    streaming=True,   # 응답을 스트리밍 방식으로 받기
    callbacks=[
        StreamingStdOutCallbackHandler(),  # 응답을 실시간으로 출력
    ],
)

# 대화 내용을 요약하여 저장하는 메모리 초기화
summary_memory = ConversationSummaryBufferMemory(
    llm=chat,  # 요약에 사용할 언어 모델
    memory_key="history",  # 메모리를 참조할 때 사용할 키
    return_messages=True   # 메시지 객체 형태로 반환
)

# Few-shot 학습을 위한 예시 데이터
examples = [
    {
        "movie": "Express topgun with 3 icons and explain",
        "answer": """
                🛩️👨‍✈️🔥
                첫 번째 아이콘은 탑건의 비행기를 나타내며, 두 번째 아이콘은 탑건의 파일럿을 나타내며, 세 번째 아이콘은 비행기 전투를 나타냅니다.
        """,
    },
    {
        "movie": "Express godzilla with 3 icons and explain",
        "answer": """
                🦖🔥👑
                첫 번째 아이콘은 고지라의 모습을 나타내며, 두 번째 아이콘은 고지라의 불꽃을 나타내며, 세 번째 아이콘은 고지라의 왕관을 나타냅니다.
        """,
    },
]

# 각 예시를 위한 채팅 프롬프트 템플릿 생성
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Express {movie} with 3 icons and explain"),
        ("ai", "{answer}"),
    ]
)

# Few-shot 학습을 위한 프롬프트 템플릿 생성
example_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 최종 프롬프트 템플릿 구성
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie expert. Here are some examples of how you talk. and answer in Korean."),  # 시스템 역할 정의
        example_prompt,  # Few-shot 예시들
        MessagesPlaceholder(variable_name="history"),  # 이전 대화 내용을 위한 플레이스홀더
        ("human", "{question}"),  # 사용자 입력
    ]
)

# 메모리에서 대화 내용을 로드하는 함수
def load_memory(_):
    return summary_memory.load_memory_variables({})["history"]

# 체인 구성: 메모리 로드 -> 프롬프트 생성 -> 챗봇 응답
chain = RunnablePassthrough.assign(history=load_memory) | final_prompt | chat

# 체인을 실행하고 결과를 메모리에 저장하는 함수
def invoke_chain(question):
    # 체인 실행
    result = chain.invoke({"question": question})
    # 대화 내용을 메모리에 저장
    summary_memory.save_context(
        {"input": question},
        {"output": result.content},
    )
    print(result)