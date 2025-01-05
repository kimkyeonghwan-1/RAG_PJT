import os
from dotenv import load_dotenv
import numpy as np
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# 환경 변수 로드
load_dotenv()

# API 키 및 설정
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
embedding_upstage = UpstageEmbeddings(model="embedding-query")
index_name = "samsung"

def create_index(api_key, index_name):
    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=4096,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(index_name)
    return index

# 1. Pinecone에서 벡터 저장소 불러오기
def load_vectorstore(index_name="samsung"):
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embedding_upstage,
        text_key="chunk"
    )
    return vectorstore

# 3. 챗봇 응답 생성
def generate_response(query, index):
    query = query
    llm = ChatUpstage()
    prompt1 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 해당 질문에서 다른 답변을 하지 않고, 필요한 년도만 정확하게 'YYYY' 형식으로 출력하는 역할을 합니다.
                ---
                예시답변 : 2024
                ---
                주의: 절대 다른 답변을 하지 말고, 년도만 'YYYY' 형식으로 출력하세요. 

                현재 연도는 2024년 입니다.
                """
            ),
            ("human", "{input}"),
        ]
    )
    chain1 = prompt1 | llm | StrOutputParser()
    response1 = chain1.invoke({"input": query})
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    
    query_vector = model.encode(query).tolist()
    query_vector = np.pad(query_vector, (0, 4096 - len(query_vector)), 'constant').tolist()
    result_docs = index.query(
        vector=query_vector,
        filter={"year": {"$eq": response1[:4]}},
        top_k=3,  # 상위 3개의 결과를 가져옵니다.
        include_metadata=True  # 메타데이터를 포함합니다.
    )

    # result_docs = ensemble_retriever.invoke(query)
    prompt2 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                ---
                CONTEXT:
                {context}
                당신은 삼성전자 다트(DART) 자료를 바탕으로 취업 관련 정보를 제공하는 전문 챗봇입니다. 
                모든 질문에 대해 삼성전자 다트 자료에서 검색한 정보를 기반으로 답변을 제공합니다.
                검색 된 자료는 모두 사용자가 원하는 년도의 자료이므로 해당년도 자료가 없다하지 말고, 답변합니다.

                제공된 자료 내에서 적합한 정보를 검색하여 응답합니다.
                답변의 정확성과 관련성을 유지하며, 자료에서 언급되지 않은 사항은 추측하지 않고 명확히 설명합니다.

                유저 메시지 예시:

                "내가 현재 삼성전자 DS부문 면접을 준비 중인데, 면접 때 알아야 할 해당 부서의 재무정보나 최신 기술 등을 알려줘."

                모범 응답 예시:

                "다트에 공시된 삼성전자의 가장 최근 자료는 2024년도 자료입니다. 해당 년도의 재무 정보는 다음과 같습니다. 
                매출액은 XX조 원이며, 영업이익은 XX조 원으로 기록되었습니다. 또한 기술로는 로봇 및 IoT 부문을 최근 사업 주제로 선정하고 있습니다. 
                해당 내용을 바탕으로 기업의 최신 동향을 묻는 질문에 '삼성전자는 최근 로봇 및 IoT 부문에 많은 관심을 두고 있습니다. 
                이를 통해 기술 혁신과 신사업 확장을 이루고자 하는 전략을 엿볼 수 있습니다.'와 같은 방식으로 대답하면 좋을 것 같습니다."

                주의 : 현재 년도는 2024년입니다
                """
            ),
            ("human", "{input}"),
        ]
    )
    llm = ChatUpstage()
    chain2 = prompt2 | llm | StrOutputParser()
    response2 = chain2.invoke({"context": result_docs, "input": query})
    return response2

# 4. 메인 함수
def main():
    # Pinecone에서 벡터 저장소 불러오기
    vectorstore = load_vectorstore(index_name)
    print("✅ Pinecone 벡터 저장소 로드 완료.")
    
    # 사용자 쿼리 입력
    query = input("🔍 질문을 입력하세요: ")
    index = create_index(pinecone_api_key, index_name)
    response = generate_response(query, index)
    print("🤖 챗봇 응답:")
    print(response)

if __name__ == "__main__":
    main()

    