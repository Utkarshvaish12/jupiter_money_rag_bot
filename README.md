📘 RAG-Driven Jupiter Money Reasoning Engine (FastAPI)

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline with:

/predict_reason — Predict why a user dropped from the funnel

/nudge_user — Generate nudges using RAG + LLM

/chat — General assistant chat endpoint

Drop Reasoning Engine (Deterministic + RAG + LLM)

Vector database (FAISS) for rule/doc retrieval

FastAPI for serving model APIs

The system uses LangChain, Groq/Mistral, FAISS, and Python 3.10+.


🚀 Project Setup
1. Create virtual environment
python -m venv .venv

2. Activate environment

Windows:

.venv\Scripts\Activate

3. Install dependencies
pip install -r requirements.txt

📘 Pre-processing (Embeddings)

Generate embeddings for the knowledge base:

python -m src.embedding

▶️ Run the FastAPI Server
uvicorn main:app --reload --host 0.0.0.0 --port 8080

🔌 API Endpoints
1. /predict_reason

Predicts primary & secondary drop-off reasons + confidence.

2. /nudge_user

Generates personalised nudges to bring user back.

3. /chat

RAG-based assistant for resolving user queries.