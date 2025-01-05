import os
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from uuid import uuid4
import time

# 🌟 1. 환경 변수 로드
load_dotenv()
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
embedding_upstage = UpstageEmbeddings(model="embedding-query")
index_name = "samsung"


# 🌟 2. Pinecone 초기화 및 설정
def setup_pinecone(api_key):
    pc = Pinecone(api_key=api_key)
    
    # 기존 인덱스 삭제
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
    
    # 새로운 인덱스 생성
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    return pc.Index(index_name)


# 🌟 3. 데이터 로드 및 Document 객체 생성
def load_documents(file_path):
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    return [
        Document(
            page_content=row['text'],
            metadata={
                "corp_name": row['corp_name'],
                "report_nm": row['report_nm'],
                "rcept_dt": row['rcept_dt'],
                "year": row['report_nm'][7:11]
            }
        )
        for _, row in df.iterrows()
    ]


# 🌟 4. 텍스트 청크 분할
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    texts, metas = [], []
    for doc in docs:
        chunks = text_splitter.split_text(doc.page_content)
        metadata = doc.metadata
        
        for i, chunk in enumerate(chunks):
            texts.append(f"{metadata['year']}-{metadata['report_nm'][12:14]} : {chunk}")
            metas.append({
                'chunk_id': i,
                'chunk': f"{metadata['year']}-{metadata['report_nm'][12:14]} : {chunk}",
                **metadata,
            })
    return texts, metas


# 🌟 5. 데이터 Pinecone에 업서트
def upload_to_pinecone(index, texts, metas, embedding_model, batch_size=100):
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metas = metas[i:i + batch_size]
        
        ids = [str(uuid4()) for _ in batch_texts]
        embeddings = embedding_model.embed_documents(batch_texts)
        
        index.upsert(vectors=zip(ids, embeddings, batch_metas))
        time.sleep(1)  # 과도한 요청 방지


# 🌟 6. Pinecone 벡터 저장소 설정
def save_to_pinecone(index, embedding_model):
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        text_key="chunk"
    )
    print(f"✅ 데이터가 Pinecone 인덱스 '{index_name}'에 성공적으로 저장되었습니다.")


# 🌟 7. 메인 함수
def main():
    print("✅ Pinecone 인덱스 설정 중...")
    index = setup_pinecone(pinecone_api_key)
    
    print("✅ 데이터 로드 중...")
    docs = load_documents('df_dart.csv')
    
    print("✅ 문서 분할 중...")
    texts, metas = split_documents(docs)
    
    print("✅ Pinecone에 데이터 업로드 중...")
    upload_to_pinecone(index, texts, metas, embedding_upstage)
    
    print("✅ Pinecone 벡터 저장소 설정 중...")
    save_to_pinecone(index, embedding_upstage)
    
    print("🎯 모든 작업이 성공적으로 완료되었습니다!")


# 🌟 8. 프로그램 실행
if __name__ == "__main__":
    main()
