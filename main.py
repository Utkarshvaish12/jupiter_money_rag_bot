from fastapi import FastAPI
from pydantic import BaseModel
from src.search import RAGSearch

app = FastAPI(
    title="RAG API",
    version="1.0",
    description="RAG-based Reason Prediction, Nudges & Chat APIs"
)

rag = RAGSearch()

class QueryRequest(BaseModel):
    query: str

class ChatRequest(BaseModel):
    message: str

@app.post("/predict_reason")
def predict_reason(req: QueryRequest):
    answer = rag.search_and_answer(req.query, top_k=5)
    return {
        "query": req.query,
        "predicted_reason": answer
    }

@app.post("/nudge_user")
def nudge_user(req: QueryRequest):

    prompt = f"""
    A user faces this issue: "{req.query}"

    Based on the issue, generate a short actionable nudge message.
    Keep the message under 4 lines.
    Be simple, polite, and helpful.
    """

    response = rag.llm.invoke(prompt).content

    return {
        "query": req.query,
        "nudge": response
    }

@app.post("/chat")
def chat(req: ChatRequest):
    reply = rag.search_and_answer(req.message, top_k=5)
    return {
        "user_message": req.message,
        "bot_reply": reply
    }


@app.get("/")
def root():
    return {"status": "RAG API running!"}
