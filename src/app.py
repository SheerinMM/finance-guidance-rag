"""
app.py
Gradio interface for the Finance Guidance RAG assistant.

Pipeline per query:
  1. Retrieve top-k relevant chunks from the FAISS index
  2. Run safeguard checks (advice-request detection + retrieval-confidence check)
  3. If blocked, return the safeguard message directly (no LLM call)
  4. Otherwise, generate a grounded answer using only the retrieved context
  5. Append source citations to the answer

Run with: python app.py  (after building the index with embed_index.py)
"""

import gradio as gr

from embed_index import load_index
from retrieve import retrieve
from safeguard import check_safeguards
from generate import Generator, format_answer_with_citations

INDEX_PATH = "data/faiss_index.bin"
CORPUS_PATH = "data/corpus.json"
TOP_K = 3

print("Loading FAISS index and embedding model...")
index, embed_model, corpus = load_index(INDEX_PATH, CORPUS_PATH)

print("Loading generation model (this may take a minute on first run)...")
generator = Generator()

print("Ready.")


def answer_question(query: str, history=None) -> str:
    if not query or not query.strip():
        return "Please enter a question."

    retrieved = retrieve(query, index, embed_model, corpus, top_k=TOP_K)

    blocked, message = check_safeguards(query, retrieved)
    if blocked:
        return message

    raw_answer = generator.generate(query, retrieved)
    return format_answer_with_citations(raw_answer, retrieved)


DESCRIPTION = """
# Finance Guidance Assistant (RAG Prototype)

Ask general questions about UK pensions, savings, debt, mortgages, credit, and investing.

**This is guidance, not regulated financial advice.** The assistant will not
answer personalised "what should I do" questions — it will point you toward
Pension Wise or a regulated adviser instead. Answers are grounded only in a
small knowledge base of public guidance content; see the README for scope and
limitations.
"""

demo = gr.ChatInterface(
    fn=answer_question,
    title="Finance Guidance Assistant",
    description=DESCRIPTION,
    examples=[
        "What is the difference between a defined contribution and defined benefit pension?",
        "What is a Lifetime ISA?",
        "How can I recognise a pension scam?",
        "Should I move my pension into a SIPP given my situation?",  # triggers safeguard
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True gives a public link when run in Colab
