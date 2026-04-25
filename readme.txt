RAG Document Question Answering (LangChain + LangGraph)
Overview:

This project implements a Retrieval-Augmented Generation (RAG) pipeline for document question answering using LangChain and LangGraph, enhanced with cross-encoder re-ranking to improve retrieval quality and answer accuracy.

Problem Statement:

Standard RAG systems rely on vector similarity, which may retrieve semantically similar but irrelevant chunks.
This project improves retrieval precision by:

Fetching candidate documents using embeddings
Re-ranking them using a cross-encoder for better relevance scoring
Architecture:
Query → Retriever (Bi-Encoder) → Cross-Encoder Re-ranker → LLM Generator → Response

The workflow is orchestrated using LangGraph, enabling a modular and extensible pipeline.

Key Components
Document Ingestion
Chunking and preprocessing of documents
Embedding + Vector Store
Semantic retrieval using bi-encoder embeddings
Retriever (Stage 1)
Fetches top-k candidate chunks
Cross-Encoder Re-ranker (Stage 2)
Scores query-document pairs jointly
Selects most relevant context for generation
LLM Generator
Produces final grounded response
LangGraph Pipeline
Controls flow across components

Key Features
Two-stage retrieval:
Fast bi-encoder retrieval
Accurate cross-encoder re-ranking
Improved answer relevance and reduced noise
Modular pipeline using LangGraph
Clean separation of retrieval and generation
Extensible for advanced workflows


Evaluation
Recall@k
MRR (Mean Reciprocal Rank)

Cross-encoder re-ranking significantly improves ranking quality compared to standalone vector search.

⚡ API Usage (if applicable)
Endpoint:
POST /query
Request:
{
  "question": "What is ...?"
}
Response:
{
  "answer": "..."
}
 Project Structure:
├── src/
│   ├── ingestion.py
│   ├── retriever.py
│   ├── reranker.py   
│   ├── generator.py
│   ├── graph.py
│   └── utils.py
├── api/
│   └── main.py
├── sample_data/
├── requirements.txt
└── README.md

Tech Stack:
Python
LangChain
LangGraph
FAISS / Chroma
Cross-Encoder (for re-ranking)
LLM (OpenAI / local)
FastAPI


▶How to Run:
git clone <repo-url>
cd rag-langgraph-pipeline

pip install -r requirements.txt
uvicorn api.main:app --reload
