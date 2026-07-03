"""
retrieve.py
Given a user query, embeds it and retrieves the top-k most similar chunks
from the FAISS index, along with their similarity scores and source metadata
(used later for citation display).
"""

import numpy as np


def retrieve(query: str, index, model, corpus: list[dict], top_k: int = 3) -> list[dict]:
    """
    Returns a list of the top_k matching chunk records, each augmented with
    a 'score' field (cosine similarity, since embeddings are normalized).
    """
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_vec.astype(np.float32), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        record = dict(corpus[idx])  # copy so we don't mutate the corpus
        record["score"] = float(score)
        results.append(record)
    return results


if __name__ == "__main__":
    from embed_index import load_index

    index, model, corpus = load_index("data/faiss_index.bin", "data/corpus.json")

    test_queries = [
        "When can I take money out of my pension?",
        "What's the difference between saving and investing?",
        "How do I improve my credit score?",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retrieve(q, index, model, corpus, top_k=3)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['title']} ({r['category']})")
