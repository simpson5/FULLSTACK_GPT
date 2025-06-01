from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory

# LLM 초기화
llm = ChatOpenAI(
    temperature=0.1,
)

# 문서 로더 및 스플리터 설정
splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\\n",
    chunk_size=600,
    chunk_overlap=100,
)
loader = UnstructuredFileLoader("./files/chapter_one.txt")
docs = loader.load_and_split(text_splitter=splitter)

# 임베딩 설정
cache_dir = LocalFileStore("./.cache/")
embeddings = OpenAIEmbeddings()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

# Vector Store 생성
vectorstore = FAISS.from_documents(docs, cached_embeddings)
retriever = vectorstore.as_retriever()

# 메모리 설정
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
)

# 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant. Answer questions based on the following context and chat history. 
        If you don't know the answer, just say you don't know. Don't make up answers.
        
        Context: {context}
        
        Chat History: {chat_history}
        """
    ),
    ("human", "{question}"),
])

# Chain 구성
chain = {
    "context": retriever,
    "chat_history": lambda x: memory.load_memory_variables({})["chat_history"],
    "question": RunnablePassthrough(),
} | prompt | llm

# 질문 함수 정의
def ask_question(question):
    response = chain.invoke(question)
    # 메모리에 대화 저장
    memory.save_context(
        {"question": question},
        {"answer": response.content}
    )
    return response.content

# 테스트 질문
questions = [
    "Aaronson 은 유죄인가요?",
    "그가 테이블에 어떤 메시지를 썼나요?",
    "Julia 는 누구인가요?"
]

# 질문 실행
print("=== RAG 테스트 시작 ===\\n")
for question in questions:
    print(f"질문: {question}")
    answer = ask_question(question)
    print(f"답변: {answer}\\n") 