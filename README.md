# 🎓 EduRAG - RAG-based AI Teaching Assistant

Upload your course notes (PDF / TXT / MD) and chat with them. Answers are **grounded in your material**, come with **page-level citations**, and the assistant says "I couldn't find that" instead of hallucinating.

## Features
- **Retrieval-Augmented Generation** - local sentence-transformer embeddings + FAISS vector search
- **Cited answers** - every answer links back to source file and page
- **Hallucination guard** - similarity threshold; no relevant context means no LLM guess
- **Follow-up aware chat** - questions are rewritten into standalone queries using chat history
- **Adjustable level** - Beginner / Intermediate / Advanced explanations
- **Study tools** - auto-generated MCQ quizzes with scoring, and revision notes
- **Pluggable LLM** - Anthropic, OpenAI, or fully local via Ollama
- **Evaluation script** - Hit@k and MRR for retrieval quality
- **Persistent index** - documents are embedded once and saved to disk

## Architecture
```
Documents -> Loader -> Chunker (800 chars, 150 overlap) -> Embeddings (MiniLM) -> FAISS index
Question -> (rewrite w/ history) -> Embed -> Top-k search -> Threshold filter -> LLM + context -> Answer + citations
```

## Setup
```bash
git clone <your-repo-url>
cd rag-teaching-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your API key (or use Ollama)
streamlit run app.py
```
Upload `data/sample_notes.txt` (or your own PDFs) in the sidebar, click **Index uploaded files**, and start asking.

## Evaluate retrieval
```bash
python evaluate.py
```
Edit `eval_set.json` with questions and a keyword that must appear in the correct chunk.

## Project structure
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (chat, quiz, notes) |
| `teaching_assistant.py` | RAG pipeline |
| `vector_store.py` | FAISS + embeddings + persistence |
| `document_loader.py` | File parsing and chunking |
| `llm.py` | Anthropic / OpenAI / Ollama wrapper |
| `evaluate.py` | Retrieval metrics |

## Tech stack
Python, Streamlit, FAISS, Sentence-Transformers, pypdf, Anthropic / OpenAI / Ollama.

## Ideas to extend
Hybrid BM25 + vector search, a re-ranker, OCR for scanned PDFs, per-user knowledge bases, Docker deployment.
