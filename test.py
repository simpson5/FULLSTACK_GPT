from langchain.chat_models import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory, # 대화 내용을 버퍼에 저장
    ConversationBufferWindowMemory, # 대화 내용을 윈도우 크기만큼 버퍼에 저장
    ConversationSummaryMemory, # 대화 내용을 요약해서 버퍼에 저장
    ConversationSummaryBufferMemory, # 대화 내용을 요약해서 버퍼에 저장
    ConversationKGMemory # 대화 내용을 지식 그래프에 저장
)
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage

# ChatOpenAI 모델 초기화
chat = ChatOpenAI(
    temperature=0.1  # 낮은 temperature 값으로 일관된 응답 생성
)

# 1. 기본적인 ConversationBufferMemory 사용 예시
memory = ConversationBufferMemory(
    memory_key="history",  # ConversationChain의 기본 프롬프트와 일치하도록 변경
    return_messages=True   # 메시지 객체 형태로 반환
)

# 2. ConversationBufferWindowMemory 사용 예시 - 최근 k개의 대화만 저장
window_memory = ConversationBufferWindowMemory(
    k=2,  # 최근 2개의 대화만 유지
    memory_key="history",
    return_messages=True
)

# 3. ConversationSummaryMemory 사용 예시 - 대화 내용을 요약하여 저장
summary_memory = ConversationSummaryMemory(
    llm=chat,
    memory_key="history",
    return_messages=True
)

# 4. ConversationChain으로 메모리 사용하기
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True  # 체인의 동작 과정을 출력
)

# 대화 예시
response = conversation.predict(input="안녕하세요!")
print(response)

# 메모리에 저장된 대화 내용 확인
print("\n저장된 대화 내용:")
print(memory.load_memory_variables({})["history"])

# 새로운 대화 추가
response = conversation.predict(input="오늘 날씨는 어떤가요?")
print(response)

# 업데이트된 대화 내용 확인
print("\n업데이트된 대화 내용:")
print(memory.load_memory_variables({})["history"])

"""
메모리 타입별 특징:

1. ConversationBufferMemory
   - 모든 대화 내용을 그대로 저장
   - 장점: 완전한 대화 기록 유지
   - 단점: 긴 대화에서 메모리 사용량 증가

2. ConversationBufferWindowMemory
   - 지정된 개수(k)의 최근 대화만 저장
   - 장점: 메모리 사용량 제한, 최근 문맥에 집중
   - 단점: 오래된 대화 컨텍스트 손실

3. ConversationSummaryMemory
   - 대화 내용을 요약하여 저장
   - 장점: 메모리 효율적 사용, 긴 대화에서 유용
   - 단점: 요약 과정에서 세부 정보 손실 가능

4. ConversationSummaryBufferMemory
   - 최근 대화는 그대로 저장, 오래된 대화는 요약
   - 장점: 최근 대화의 상세함과 과거 대화의 컨텍스트 모두 유지
   - 단점: 요약 과정에서 계산 비용 발생

5. ConversationKGMemory
   - 대화에서 추출한 정보로 지식 그래프 구성
   - 장점: 구조화된 정보 저장, 관계 기반 검색 가능
   - 단점: 설정이 복잡하고 처리 비용이 높음
"""