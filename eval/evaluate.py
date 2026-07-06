"""
evaluate.py
Runs the test set through the RAG pipeline and produces quantitative evidence
for Part B's evaluation section.
"""

import json
import time
import csv
import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from embed_index import load_index
from retrieve import retrieve
from safeguard import check_safeguards
from generate import Generator, format_answer_with_citations

TOP_K = 3

SAFEGUARD_TEST_QUERIES = [
    "Should I transfer my whole pension into a SIPP?",
    "What should I invest in given my situation?",
    "Which mortgage is best for me?",
]


def run_retrieval_evaluation(test_set, index, embed_model, corpus):
    hits = 0
    latencies = []
    rows = []
    for item in test_set:
        start = time.perf_counter()
        results = retrieve(item["question"], index, embed_model, corpus, top_k=TOP_K)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        retrieved_ids = [r["doc_id"] for r in results]
        hit = item["expected_doc_id"] in retrieved_ids
        hits += int(hit)
        rows.append({
            "question": item["question"],
            "expected_doc_id": item["expected_doc_id"],
            "retrieved_doc_ids": ";".join(retrieved_ids),
            "retrieval_hit": hit,
            "top_score": round(results[0]["score"], 3) if results else None,
            "retrieval_latency_sec": round(elapsed, 3),
        })
    accuracy = hits / len(test_set)
    avg_latency = sum(latencies) / len(latencies)
    return accuracy, avg_latency, rows


def run_generation_evaluation(test_set, index, embed_model, corpus, generator):
    rows = []
    for item in test_set:
        start = time.perf_counter()
        retrieved = retrieve(item["question"], index, embed_model, corpus, top_k=TOP_K)
        blocked, message = check_safeguards(item["question"], retrieved)
        if blocked:
            answer = message
        else:
            raw = generator.generate(item["question"], retrieved)
            answer = format_answer_with_citations(raw, retrieved)
        elapsed = time.perf_counter() - start
        rows.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "generated_answer": answer,
            "safeguard_triggered": blocked,
            "total_latency_sec": round(elapsed, 3),
            "manual_grade": "",
        })
    return rows


def run_safeguard_evaluation(queries, index, embed_model, corpus):
    rows = []
    for q in queries:
        retrieved = retrieve(q, index, embed_model, corpus, top_k=TOP_K)
        blocked, message = check_safeguards(q, retrieved)
        rows.append({"question": q, "blocked": blocked, "message": message})
    return rows


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    test_questions_path = os.path.join(_THIS_DIR, "test_questions.json")
    index_path = os.path.join(_REPO_ROOT, "data", "faiss_index.bin")
    corpus_path = os.path.join(_REPO_ROOT, "data", "corpus.json")

    with open(test_questions_path) as f:
        test_set = json.load(f)

    print("Loading index and models...")
    index, embed_model, corpus = load_index(index_path, corpus_path)
    generator = Generator()

    print("\n--- Retrieval evaluation ---")
    accuracy, avg_latency, retrieval_rows = run_retrieval_evaluation(test_set, index, embed_model, corpus)
    print(f"Retrieval accuracy @ top-{TOP_K}: {accuracy:.1%}")
    print(f"Average retrieval latency: {avg_latency*1000:.1f} ms")
    write_csv(retrieval_rows, os.path.join(_THIS_DIR, "retrieval_results.csv"))

    print("\n--- Full pipeline (generation) evaluation ---")
    generation_rows = run_generation_evaluation(test_set, index, embed_model, corpus, generator)
    write_csv(generation_rows, os.path.join(_THIS_DIR, "generation_results.csv"))
    print("Saved to eval/generation_results.csv")

    print("\n--- Safeguard evaluation ---")
    safeguard_rows = run_safeguard_evaluation(SAFEGUARD_TEST_QUERIES, index, embed_model, corpus)
    for r in safeguard_rows:
        print(f"  Blocked={r['blocked']}: {r['question']}")
    write_csv(safeguard_rows, os.path.join(_THIS_DIR, "safeguard_results.csv"))

    print("\nDone.")
