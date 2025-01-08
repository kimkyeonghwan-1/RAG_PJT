import os
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

# 환경 변수 로드
load_dotenv()

# API 키 로드
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
embedding_upstage = UpstageEmbeddings(model="embedding-query")

# 1. 데이터 로드
def load_data():
    df = pd.read_csv('df_dart.csv', encoding='utf-8-sig')
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
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    return splits

# 3. Pinecone 설정
def setup_pinecone(index_name="samsung"):
    pc = Pinecone(api_key=pinecone_api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=4096,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-2")
        )
    return pc

# 4. 벡터 저장소 생성 및 저장
def save_to_pinecone(splits, embedding_upstage, index_name="samsung"):
    vectorstore = PineconeVectorStore.from_documents(
        splits,
        embedding=embedding_upstage,
        index_name=index_name
    )
    print(f"✅ 데이터가 Pinecone 인덱스 '{index_name}'에 성공적으로 저장되었습니다.")

# 메인 함수
def main():
    # 데이터 로드
    docs = load_data()
    print("✅ 데이터 로드 완료.")
    
    # 문서 분할
    splits = split_documents(docs)
    print("✅ 문서 분할 완료.")
    
    # Pinecone 설정
    setup_pinecone()
    print("✅ Pinecone 설정 완료.")
    
    # Pinecone에 저장
    save_to_pinecone(splits, embedding_upstage)
    print("✅ 데이터 저장 완료.")

if __name__ == "__main__":
    main()