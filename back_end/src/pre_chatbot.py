import os
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import EnsembleRetriever

# 환경 변수 로드
load_dotenv()

# API 키 및 설정
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
embedding_upstage = UpstageEmbeddings(model="embedding-query")
index_name = "samsung"

# 1. Pinecone에서 벡터 저장소 불러오기
def load_vectorstore(index_name="samsung"):
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embedding_upstage
    )
    return vectorstore

# 2. 검색기 설정
def setup_retrievers(vectorstore):
    # Dense Retriever (Pinecone)
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={"k": 10}
    )
    
    # Ensemble Retriever (Dense만 사용)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever],
        weights=[1.0]  # Dense Retriever에 가중치 100%
    )
    return ensemble_retriever

# 3. 챗봇 응답 생성
def generate_response(ensemble_retriever, query):
    result_docs = ensemble_retriever.invoke(query)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 취업 컨설턴트로서 전문성과 공감 능력을 갖춘 챗봇입니다. 
                사용자들에게 원하는 직무와 기업에 대해 맞춤형 조언을 제공하며, 
                주어진 데이터를 기반으로 기업 추천과 상세한 정보를 전달하고, 근거를 함께 제시해야 합니다.

                당신의 역할:
                1. 사용자가 원하는 직무나 직종을 제시하면, 관련 기업을 추천하고 해당 기업에 대한 상세 정보를 제공하세요.
                   (예: 재무 상태, 최근 소식, 채용 공고 등)
                2. 사용자가 특정 기업을 언급하면, 해당 기업에 대해 가능한 모든 정보를 상세히 설명하세요.
                3. 답변은 항상 명확하고, 정확하며, 사용자가 바로 행동에 옮길 수 있는 내용을 포함하세요.
                4. 데이터에 기반한 답변이 불가능할 경우, 정중히 안내하고 다른 방법을 제안하세요.

                항상 전문적이면서도 친근한 태도로 답변하며, 사용자가 취업 목표를 달성할 수 있도록 진심으로 돕는 모습을 보여주세요.
                ---
                CONTEXT:
                {context}
                """
            ),
            ("human", "{input}"),
        ]
    )
    llm = ChatUpstage()
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": result_docs, "input": query})
    return response

# 4. 메인 함수
def main():
    # Pinecone에서 벡터 저장소 불러오기
    vectorstore = load_vectorstore(index_name)
    print("✅ Pinecone 벡터 저장소 로드 완료.")
    
    # 검색기 설정
    ensemble_retriever = setup_retrievers(vectorstore)
    print("✅ 검색기 설정 완료.")
    
    # 사용자 쿼리 입력
    query = input("🔍 질문을 입력하세요: ")
    response = generate_response(ensemble_retriever, query)
    print("🤖 챗봇 응답:")
    print(response)

if __name__ == "__main__":
    main()