"""
app.py
Gradio interface for the Finance Guidance RAG assistant.
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

print("Loading generation model...")
generator = Generator()


def answer_question(query, history=None):
    if not query or not query.strip():
        return "Please enter a question."
    retrieved = retrieve(query, index, embed_model, corpus, top_k=TOP_K)
    blocked, message = check_safeguards(query, retrieved)
    if blocked:
        return message
    raw_answer = generator.generate(query, retrieved)
    return format_answer_with_citations(raw_answer, retrieved)


THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="slate")

HEADER_HTML = """
<div style="background:#0f172a;padding:20px 24px;border-radius:12px;margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="font-size:28px;">&#128176;</div>
    <div>
      <div style="color:#ffffff;font-size:20px;font-weight:700;">Finance Guidance Assistant</div>
      <div style="color:#94a3b8;font-size:13px;">UK personal finance Q&amp;A - guidance, not regulated advice</div>
    </div>
  </div>
</div>
"""

DISCLAIMER_HTML = """
<div style="background:#ecfdf5;border:1px solid #a7f3d0;border-radius:8px;padding:10px 14px;margin-bottom:8px;">
  <span style="color:#065f46;font-size:13px;">
  This assistant provides general guidance only. It will not answer personalised
  "what should I do" questions - for those, it points you to Pension Wise or a regulated adviser.
  </span>
</div>
"""

EXAMPLES = [
    "What is the difference between a defined contribution and defined benefit pension?",
    "What is a Lifetime ISA?",
    "How can I recognise a pension scam?",
    "What is a loan?",
    "Should I move my pension into a SIPP given my situation?",
]

with gr.Blocks(title="Finance Guidance Assistant") as demo:
    gr.HTML(HEADER_HTML)
    gr.HTML(DISCLAIMER_HTML)
    gr.ChatInterface(
        fn=answer_question,
        examples=EXAMPLES,
        editable=True,
        save_history=True,
        fill_height=True,
    )

if __name__ == "__main__":
    demo.launch(share=True, theme=THEME)
