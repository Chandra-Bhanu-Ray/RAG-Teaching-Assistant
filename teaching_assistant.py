"""Core RAG pipeline: ingest -> retrieve -> grounded answer / quiz / summary."""
import json
import re
from pathlib import Path

import config
import llm
from document_loader import load_and_chunk
from vector_store import VectorStore

LEVELS = {
    "Beginner": "Explain simply with an everyday analogy. Avoid jargon or define it.",
    "Intermediate": "Explain clearly with technical terms and one short example.",
    "Advanced": "Be precise and concise. Include edge cases, trade-offs or formulas where relevant.",
}

ANSWER_SYSTEM = """You are a patient, accurate teaching assistant.
Answer ONLY using the numbered context passages provided.
- Cite passages inline like [1], [2] after the claims they support.
- If the context does not contain the answer, say you could not find it in the course material. Never invent facts.
- Teaching style: {style}"""


def _format_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i}] ({h['source']}, p.{h['page']})\n{h['text']}" for i, h in enumerate(hits, 1))


class TeachingAssistant:
    def __init__(self):
        self.store = VectorStore.load_or_new()

    # ---------- ingestion ----------
    def ingest(self, path: Path) -> int:
        path = Path(path)
        if path.name in self.store.sources():
            return 0  # already indexed
        chunks = load_and_chunk(path)
        self.store.add(chunks)
        self.store.save()
        return len(chunks)

    def reset(self) -> None:
        self.store.reset()

    # ---------- question answering ----------
    def _condense(self, question: str, history: list[dict]) -> str:
        """Rewrite a follow-up question into a standalone one using chat history."""
        if not history:
            return question
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        prompt = (
            f"Conversation:\n{convo}\n\nFollow-up question: {question}\n\n"
            "Rewrite the follow-up as a single standalone question. Output only the question."
        )
        try:
            return llm.generate("You rewrite questions.", prompt, max_tokens=120, temperature=0).strip() or question
        except Exception:
            return question

    def ask(self, question: str, level: str = "Beginner", history: list[dict] | None = None) -> dict:
        standalone = self._condense(question, history or [])
        hits = self.store.search(standalone, config.TOP_K)
        hits = [h for h in hits if h["score"] >= config.MIN_SCORE]
        if not hits:
            return {
                "answer": "I couldn't find anything about that in the uploaded course material. "
                "Try rephrasing, or upload a document that covers this topic.",
                "sources": [],
                "standalone_question": standalone,
            }
        prompt = f"Context:\n{_format_context(hits)}\n\nQuestion: {standalone}"
        answer = llm.generate(ANSWER_SYSTEM.format(style=LEVELS[level]), prompt, max_tokens=900)
        return {"answer": answer, "sources": hits, "standalone_question": standalone}

    # ---------- study tools ----------
    def summarize(self, topic: str) -> dict:
        hits = [h for h in self.store.search(topic, 6) if h["score"] >= config.MIN_SCORE]
        if not hits:
            return {"answer": "No material found for that topic.", "sources": []}
        prompt = (
            f"Context:\n{_format_context(hits)}\n\n"
            f"Write concise revision notes on '{topic}' as 5-8 bullet points. Cite passages like [1]."
        )
        return {"answer": llm.generate(ANSWER_SYSTEM.format(style=LEVELS["Intermediate"]), prompt), "sources": hits}

    def quiz(self, topic: str, n: int = 5) -> list[dict]:
        hits = [h for h in self.store.search(topic, 6) if h["score"] >= config.MIN_SCORE]
        if not hits:
            return []
        prompt = (
            f"Context:\n{_format_context(hits)}\n\n"
            f"Create {n} multiple-choice questions on '{topic}' using ONLY this context.\n"
            'Return ONLY a JSON list. Each item: {"question": str, "options": [4 strings], '
            '"answer_index": int (0-3), "explanation": str}.'
        )
        raw = llm.generate("You write exam questions and output strict JSON only.", prompt, max_tokens=1800, temperature=0.4)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            items = json.loads(raw)
            return [q for q in items if {"question", "options", "answer_index", "explanation"} <= q.keys()]
        except json.JSONDecodeError:
            return []
