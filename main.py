import os
import re
import json
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Load environment variables from .env file if available
load_dotenv()

# --- Configuration (all from environment variables — set these in Render's dashboard or .env) ---
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "education_documents")

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "google")  # "google" or "anthropic"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-2.0-flash")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")

LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "500"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
TOP_K = int(os.environ.get("TOP_K", "5"))
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

# Comma-separated list; includes localhost for your local Next.js dev server
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [origin.strip() for origin in _raw_origins.split(",") if origin.strip()]

SYSTEM_PROMPT = """You are an educational AI assistant.
Answer the user's question using only the retrieved document context.
Do not invent facts that are not supported by the retrieved documents.
If the context does not contain enough information, say that the available educational material does not provide enough information to answer confidently.
Give a clear educational explanation.
For technical questions, preserve formulas, terminology, code, commands, and numerical values from the source.
For every important statement based on the documents, provide the source document and page number.
Never fabricate a citation."""

BLOCKED_KEYWORDS = [
    "sex", "sexual", "porn", "nude", "naked", "boobs", "penis", "vagina", "condom",
    "fuck", "fucking", "shit", "bitch", "bastard", "asshole", "slut", "whore",
    "idiot", "stupid ass", "dumbass", "motherfucker",
    "bsdk", "bsdki", "mc", "bc", "chutiya", "chutiye", "randi", "gandu", "gaandu",
    "harami", "haraami", "saala", "kutta", "kamina", "lund", "lauda", "chodu",
    "madarchod", "behenchod",
]

FLAGGED_LOG_FILE = os.environ.get("FLAGGED_LOG_FILE", "flagged_questions.log")

def contains_blocked_content(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(r"\b" + re.escape(word) + r"\b", text_lower) for word in BLOCKED_KEYWORDS)

def log_flagged_question(question: str):
    with open(FLAGGED_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
        }) + "\n")

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("WARNING: QDRANT_URL or QDRANT_API_KEY is not set. Qdrant operations will fail.")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (this takes a minute on startup)...")
    state["embedding_model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Connecting to Qdrant Cloud...")
    if QDRANT_URL and QDRANT_API_KEY:
        state["qdrant"] = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    else:
        state["qdrant"] = None

    print(f"Setting up LLM client ({LLM_PROVIDER})...")
    if LLM_PROVIDER == "google":
        from google import genai
        state["genai"] = genai
        state["google_client"] = genai.Client(api_key=GOOGLE_API_KEY)
    elif LLM_PROVIDER == "anthropic":
        import anthropic
        state["anthropic_client"] = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    else:
        raise RuntimeError(f"LLM_PROVIDER must be 'google' or 'anthropic', got {LLM_PROVIDER!r}")

    print("Backend ready.")
    yield
    state.clear()

app = FastAPI(title="Education RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

class Source(BaseModel):
    document: str
    page: int

class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    flagged: bool = False

def embed_query(text: str):
    vec = state["embedding_model"].encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vec.tolist()

def build_context(results) -> str:
    return "\n\n".join(
        f"[{r.payload.get('document_name', 'Unknown')} - Page {r.payload.get('page_number', 'N/A')}]\n{r.payload.get('text', '')}"
        for r in results
    )

def generate_answer(question: str, context_text: str) -> str:
    user_content = f"Retrieved context:\n\n{context_text}\n\nQuestion: {question}"
    if LLM_PROVIDER == "google":
        response = state["google_client"].models.generate_content(
            model=GOOGLE_MODEL,
            contents=user_content,
            config=state["genai"].types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            ),
        )
        return (response.text or "").strip()
    else:
        response = state["anthropic_client"].messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_provider": LLM_PROVIDER,
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME
    }

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    if contains_blocked_content(question):
        log_flagged_question(question)
        return AskResponse(answer="This question has been reported to the admin.", sources=[], flagged=True)

    if not state.get("qdrant"):
        raise HTTPException(status_code=500, detail="Qdrant client not initialized. Check server logs.")

    top_k = req.top_k or TOP_K
    query_vec = embed_query(question)

    results = state["qdrant"].query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    ).points

    if not results:
        return AskResponse(
            answer="The available educational material does not provide enough information to answer confidently.",
            sources=[],
        )

    context_text = build_context(results)
    answer = generate_answer(question, context_text)
    sources = [
        Source(
            document=r.payload.get("document_name", "Unknown"),
            page=int(r.payload.get("page_number", 0))
        )
        for r in results
    ]

    return AskResponse(answer=answer, sources=sources)
