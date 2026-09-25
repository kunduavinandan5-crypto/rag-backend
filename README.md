# Education RAG Backend

FastAPI backend for Educational Retrieval-Augmented Generation (RAG) using Qdrant Vector DB, BAAI/bge-m3 embeddings, and Google Gemini / Anthropic Claude LLMs.

## 🚀 Quickstart

### 1. Setup Virtual Environment & Install Dependencies

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

### 3. Run Locally

```bash
uvicorn main:app --reload --port 8000
```

- API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Ask Endpoint: `POST http://localhost:8000/ask`

## 🌐 Deploy to Render

1. Create a **Web Service** on [Render](https://render.com).
2. Set **Build Command**: `pip install -r requirements.txt`
3. Set **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. In **Environment Variables**, add the keys defined in `.env.example`.
