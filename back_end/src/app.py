from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema.output_parser import StrOutputParser
from chatbot import create_index, load_vectorstore, generate_response
from langchain_upstage import UpstageEmbeddings
# from vectorstore import identify_manufacturer, get_filtered_retriever, vectorstore, prompt, llm

load_dotenv()  # Load .env file if present

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
embedding_upstage = UpstageEmbeddings(model="embedding-query")
index_name = "samsung"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# class ChatMessage(BaseModel):
#     role: str
#     content: str
#
#
# class AssistantRequest(BaseModel):
#     message: str
#     thread_id: Optional[str] = None
#
#
# class ChatRequest(BaseModel):
#     messages: List[ChatMessage]  # Entire conversation for naive mode


class MessageRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # Pinecone에서 벡터 저장소 불러오기
    vectorstore = load_vectorstore(index_name)
    print("✅ Pinecone 벡터 저장소 로드 완료.")

    index = create_index(pinecone_api_key, index_name)
    response = generate_response(req.message, index)
    print("🤖 챗봇 응답:")

    return {"reply": response}


# @app.post("/assistant")
# async def assistant_endpoint(req: AssistantRequest):
#     assistant = await openai.beta.assistants.retrieve("asst_tc4AhtsAjNJnRtpJmy1gjJOE")
#
#     if req.thread_id:
#         # We have an existing thread, append user message
#         await openai.beta.threads.messages.create(
#             thread_id=req.thread_id, role="user", content=req.message
#         )
#         thread_id = req.thread_id
#     else:
#         # Create a new thread with user message
#         thread = await openai.beta.threads.create(
#             messages=[{"role": "user", "content": req.message}]
#         )
#         thread_id = thread.id
#
#     # Run and wait until complete
#     await openai.beta.threads.runs.create_and_poll(
#         thread_id=thread_id, assistant_id=assistant.id
#     )
#
#     # Now retrieve messages for this thread
#     # messages.list returns an async iterator, so let's gather them into a list
#     all_messages = [
#         m async for m in openai.beta.threads.messages.list(thread_id=thread_id)
#     ]
#     print(all_messages)
#
#     # The assistant's reply should be the last message with role=assistant
#     assistant_reply = all_messages[0].content[0].text.value
#
#     return {"reply": assistant_reply, "thread_id": thread_id}


@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
