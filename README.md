# Enterprise RAG - SEC EDGAR 10-K

Local RAG over 2020 SEC 10-K filings.

- **Backend:** FastAPI (`/ask`, `/ask/stream`, `/health`)
- **Frontend:** static test console served by FastAPI at `/`
- **Local LLM:** Qwen3.5-9B Q4_K_M GGUF via llama.cpp (`http://127.0.0.1:8000/v1`)
- **Models:** BGE-large-en-v1.5, Qwen3-Reranker-0.6B, Prompt Guard 2


## 1. Start Postgres

```powershell
docker compose -f compose.postgres.yml up -d
```

## 2. Install dependencies
```
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Create the database tables

```powershell
.\.venv\Scripts\rag-ingest.exe init-db
```

## 4. Install the models

```powershell
.\.venv\Scripts\python.exe download_bge.py
.\.venv\Scripts\python.exe download_qwen3_reranker.py
$env:HF_TOKEN="hf_your_token"; .\.venv\Scripts\python.exe download_prompt_guard.py
.\.venv\Scripts\python.exe download_qwen_gguf.py
```

- BGE model: `models/bge-large-en-v1.5`
- Reranker: `models/Qwen3-Reranker-0.6B`
- Prompt Guard 2: `models/llama-prompt-guard-2` (gated; request access on Hugging Face first)
- Local LLM: `models/Qwen_Qwen3.5-9B-Q4_K_M.gguf` (Qwen3.5-9B Q4_K_M, 6.17 GB, from `bartowski/Qwen_Qwen3.5-9B-GGUF`)

## 5. Ingest the corpus

```powershell
.\.venv\Scripts\rag-ingest.exe download --split test
.\.venv\Scripts\rag-ingest.exe ingest --split test
.\.venv\Scripts\rag-ingest.exe status
```

All splits:

```powershell
.\.venv\Scripts\python.exe scripts\ingest.py test validate train
```

## 6. Start the local LLM (llama.cpp)

Place a Windows CUDA llama.cpp release (`llama-server.exe` + DLLs) in `tools/llama.cpp/` first; the binaries are not committed.

```powershell
.\scripts\serve_llm.ps1
```

This serves the Qwen3.5-9B Q4_K_M GGUF at `http://127.0.0.1:8000/v1` with thinking off by default.

> This project uses **llama.cpp**, not vLLM. vLLM is not supported on native Windows. It exposes the same OpenAI-compatible API, so if you must use vLLM, run it under WSL/Linux on port 8000 and point `RAG_LLM_BASE_URL` there.

## 7. Start the backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn rag_query.api:create_app --factory --host 127.0.0.1 --port 8080
```

Alternative:

```powershell
.\.venv\Scripts\rag-ask.exe serve --port 8080
```

## 8. Open the frontend

```text
http://127.0.0.1:8080/
```

No separate frontend build or server is needed.

## 9. Ask from the CLI (optional)

```powershell
.\.venv\Scripts\rag-ask.exe ask "What are Apple's main risk factors in 2020?" --token dev-public
```

