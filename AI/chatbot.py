import os
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, UpstageDocumentParseLoader, ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

# 환경 변수 로드
load_dotenv()

# Pinecone 및 Upstage 관련 API 키 로드
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
embedding_upstage = UpstageEmbeddings(model="embedding-query")

# 1. 데이터 불러오기
def load_data():
    # df_dart.csv 파일에서 데이터 로드
    df = pd.read_csv('df_dart.csv', encoding='utf-8-sig')
    
    # 'text' 컬럼을 문서 내용으로 사용하여 Document 객체로 변환
    docs = [
        Document(
            page_content=row['text'],
            metadata={
                "corp_name": row['corp_name'],
                "report_nm": row['report_nm'],
                "rcept_dt": row['rcept_dt']
            }
        )
        for _, row in df.iterrows()
    ]
    
    return docs

# 2. 문서 분할
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 텍스트 분할 크기
        chunk_overlap=200,  # 분할된 텍스트의 중첩 크기
        length_function=len,  # 텍스트 길이 계산
        separators=["\n\n", "\n", " ", ""]  # 분할 기준
    )
    
    # 문서 분할
    splits = text_splitter.split_documents(docs)
    return splits

# 3. Pinecone 인덱스 설정
def setup_pinecone(index_name="samsung"):
    # Pinecone 초기화
    pc = Pinecone(api_key=pinecone_api_key)
    
    # 인덱스가 없으면 새로 생성
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=4096,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-2")
        )
    
    return pc

# 4. 벡터 저장소 생성
def create_vectorstore(splits, embedding_upstage, index_name="samsung"):
    # embedding 파라미터를 추가하여 PineconeVectorStore 생성
    vectorstore = PineconeVectorStore.from_documents(
        splits, embedding=embedding_upstage, index_name=index_name
    )
    return vectorstore

# 5. 검색기 설정
def setup_retrievers(splits):
    # Dense Retriever 설정
    retriever = PineconeVectorStore.from_documents(splits, embedding=embedding_upstage, index_name="samsung").as_retriever(search_type='mmr', search_kwargs={"k": 10})
    
    # Sparse Retriever (BM25) 설정
    bm25_retriever = BM25Retriever.from_documents(documents=splits)
    
    # Ensemble Retriever 설정 (Dense + Sparse)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever],
        weights=[0.7, 0.3]  # 각 Retriever의 가중치
    )
    
    return ensemble_retriever

# 6. 챗봇 응답 생성
def generate_response(ensemble_retriever, query):
    result_docs = ensemble_retriever.invoke(query)
    
    # 챗봇 프롬프트 설정
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
    
    # LLM 모델 호출
    llm = ChatUpstage()
    chain = prompt | llm | StrOutputParser()
    
    # 응답 생성
    response = chain.invoke({"context": result_docs, "input": query})
    return response

# 7. 메인 함수
def main():
    # 데이터 로드
    docs = load_data()
    
    # 문서 분할
    splits = split_documents(docs)
    
    # Pinecone 설정
    pc = setup_pinecone()
    
    # 벡터 저장소 생성
    vectorstore = create_vectorstore(splits, embedding_upstage)
    
    # 검색기 설정
    ensemble_retriever = setup_retrievers(splits)
    
    # 쿼리 및 응답 생성
    query = "2024년 삼성전자의 주요 기술 동향"
    response = generate_response(ensemble_retriever, query)
    
    # 응답 출력
    print(response)

# 실행
if __name__ == "__main__":
    main()