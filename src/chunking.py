"""
chunking.py
Loads the finance knowledge base and splits documents into retrieval-sized chunks.

For this dataset, each entry is already a short, self-contained Q&A style document
(one concept per entry), so chunking is light-touch: we keep each document as a
single chunk unless it exceeds a word-count threshold, in which case we split it
with a small overlap to avoid cutting sentences awkwardly across chunk boundaries.
"""

import json
from pathlib import Path


def load_documents(path: str) -> list[dict]:
    """Load the raw knowledge base JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_text(text: str, max_words: int = 120, overlap_words: int = 20) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Most entries in this KB are well under max_words, so they will return as a
    single chunk; this only kicks in for longer documents.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap_words
    return chunks


def build_chunk_corpus(kb_path: str) -> list[dict]:
    """
    Returns a flat list of chunk records ready for embedding, each with:
      - chunk_id: unique id (doc_id + chunk index)
      - doc_id: originating document id
      - title: originating document title (used for citation display)
      - category: originating document category
      - text: the chunk text itself
    """
    docs = load_documents(kb_path)
    corpus = []
    for doc in docs:
        pieces = chunk_text(doc["content"])
        for i, piece in enumerate(pieces):
            corpus.append({
                "chunk_id": f"{doc['id']}_{i}",
                "doc_id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "text": piece,
            })
    return corpus


if __name__ == "__main__":
    corpus = build_chunk_corpus("data/finance_kb.json")
    print(f"Loaded {len(corpus)} chunks from knowledge base.")
    print("Example chunk record:")
    print(json.dumps(corpus[0], indent=2))
