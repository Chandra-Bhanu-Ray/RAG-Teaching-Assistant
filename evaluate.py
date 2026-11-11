"""Measure retrieval quality (Hit@k and MRR) on a small hand-written question set.

Usage:  python evaluate.py
Edit eval_set.json: each item = {"question": ..., "keyword": "text that must appear in the right chunk"}
"""
import json
from pathlib import Path

import config
from vector_store import VectorStore

store = VectorStore.load_or_new()
items = json.loads(Path("eval_set.json").read_text(encoding="utf-8"))

hits_at_k, rr_total = 0, 0.0
for item in items:
    results = store.search(item["question"], config.TOP_K)
    rank = next((i for i, r in enumerate(results, 1) if item["keyword"].lower() in r["text"].lower()), None)
    if rank:
        hits_at_k += 1
        rr_total += 1 / rank
    print(f"{'OK ' if rank else 'MISS'} rank={rank} | {item['question']}")

n = len(items)
print(f"\nHit@{config.TOP_K}: {hits_at_k}/{n} = {hits_at_k / n:.0%}")
print(f"MRR: {rr_total / n:.3f}")
