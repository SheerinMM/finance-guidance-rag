"""
embed_index.py
Builds sentence embeddings for the chunk corpus and stores them in a FAISS index
for fast similarity search.

Model choice: sentence-transformers/all-MiniLM-L6-v2
  - Small (~80MB), fast, runs on CPU — good for Colab free tier and reproducibility.
  - Well-established baseline embedding model, widely used in RAG literature,
    which makes it easy to justify in the technical report.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_index(corpus: list[dict], model_name: str = EMBEDDING_MODEL_NAME):
    """
    Embeds every chunk in the corpus and builds a FAISS index over the vectors.
    Returns (index, model, corpus) so the same objects can be reused at query time.
    """
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in corpus]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # normalize so inner product == cosine similarity
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine sim
    index.add(embeddings.astype(np.float32))

    return index, model, corpus


def save_index(index, corpus: list[dict], index_path: str, corpus_path: str):
    faiss.write_index(index, index_path)
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)


def load_index(index_path: str, corpus_path: str, model_name: str = EMBEDDING_MODEL_NAME):
    index = faiss.read_index(index_path)
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    model = SentenceTransformer(model_name)
    return index, model, corpus


if __name__ == "__main__":
    from chunking import build_chunk_corpus

    corpus = build_chunk_corpus("data/finance_kb.json")
    index, model, corpus = build_index(corpus)
    save_index(index, corpus, "data/faiss_index.bin", "data/corpus.json")
    print(f"Built and saved FAISS index with {index.ntotal} vectors (dim={index.d}).")
