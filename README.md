# Hybrid RAG (BM25 + Dense + RRF)

Adapted from the supplied QuickRAG project by drwasimahmadkhan (https://github.com/drwasimahmadkhan/QuickRAG) and the accompanying Sparse RAG implementation. This educational project combines BM25Plus keyword search and dense semantic search using Reciprocal Rank Fusion (RRF). Sentence Transformers creates embeddings; NumPy computes cosine similarity in memory (no separate vector database required for this small demo). FastAPI provides Swagger UI, and Groq generates answers from retrieved chunks.

## Flow

Upload document -> split text -> tokenize and embed chunks -> run keyword and dense search -> combine ranks using RRF -> pass best chunks to Groq -> grounded answer with citations.

## Run on Windows

Use an existing compatible Python installation (Python 3.11 or 3.12 is a good fallback if ML packages fail on newer Python). Copy `.env.example` to `.env` and set your own Groq key. Choose a Groq model available to your account. In VS Code PowerShell:

```powershell
py -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn standalone:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001/docs . Alternatively run `start.bat`. The first upload may download the public embedding model from Hugging Face; internet access is needed unless it is already cached. Groq also requires internet access and receives the question and retrieved text; use non-sensitive sample data.

## Test

Use `GET /health` to check the server. Upload all three sample files individually with `POST /upload`. Use `POST /query`:

```json
{"query":"What is a firewall?", "top_k":1, "min_score":0}
```

The best chunk is expected to be `firewall.txt`. Try `Encryption` and `Phishing` as additional tests. To see both search routes in the results, use `top_k:3` and a query such as `How can I keep a computer network safe?`.

- `sparse_rank`, `bm25_score`: rank and raw score in keyword search.
- `dense_rank`, `dense_score`: rank and cosine similarity in dense search.
- `score`: fused RRF score, sum of 1/(60 + rank) for each candidate list that contains the chunk. It is not a percentage. Null rank means the chunk did not appear in that candidate list.
- `min_score`: optional minimum fused RRF score (default 0). Not comparable to raw BM25 scores or cosine similarities.

Dense search can return a closest chunk even for unrelated questions. RRF is not a confidence score. The LLM is instructed to report missing information, but that does not guarantee abstention.

Uploaded files are stored in `data/` and reindexed at startup. Reuploading a filename replaces its chunks. Vectors are cached in memory; BM25 is rebuilt per query for simplicity. This implementation uses the original query for both searches. No model training is needed.

Never commit `.env`, virtual environments or private documents. `.gitignore` excludes them. Use `.env.example` as the public template.
