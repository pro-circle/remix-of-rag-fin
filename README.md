# RAG using LangChain

This repository contains a self-contained Python RAG application in [`rag-app/`](./rag-app).

- FastAPI backend: document parsing, semantic chunking, hybrid retrieval (vector + BM25 with RRF fusion), reranking, and streaming answers with citations.
- Vanilla HTML/CSS/JS dashboard served by the same app.

Setup, architecture, API reference, and troubleshooting are documented in [`rag-app/README.md`](./rag-app/README.md).

Quick start:

```sh
cd rag-app
cp .env.example .env   # add your GROQ_API_KEY
./run.sh
```

Then open http://localhost:8000.
