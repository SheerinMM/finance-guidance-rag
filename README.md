# Finance Guidance Assistant — RAG Prototype

A retrieval-augmented generation (RAG) chatbot that answers general UK personal
finance questions (pensions, savings, debt, mortgages, credit, investing,
budgeting, tax, and more) using a curated knowledge base of original UK
financial guidance content.

Built for CIS142-6 Natural Language Processing (NLP) and Generative AI — Part B
Applied Project, University of Bedfordshire.

---

## What this is (and isn't)

This assistant provides **general financial guidance**, not personalised
financial advice. UK financial regulation legally distinguishes the two: only
FCA-authorised advisers can give personalised recommendations. The system
includes a two-layer safeguard that detects advice-seeking questions and
redirects the user to appropriate regulated services instead of answering
directly. See [Safeguard](#safeguard) below.

---

## Architecture

```
User query
    │
    ▼
Embed query (sentence-transformers/all-MiniLM-L6-v2)
    │
    ▼
Retrieve top-k chunks (FAISS, cosine similarity)
    │
    ▼
Safeguard check ──── blocked? ──► Return disclaimer / low-confidence message
    │ (not blocked)
    ▼
Build grounded prompt (query + retrieved context)
    │
    ▼
Generate answer (Qwen2.5-1.5B-Instruct)
    │
    ▼
Filter and append source citations (relative-margin relevance threshold)
    │
    ▼
Return answer to user
```

## Pipeline components

| Stage | File | Technology |
|---|---|---|
| Knowledge base | `data/finance_kb.json` | 52 original documents across 16 finance categories |
| Chunking | `src/chunking.py` | Word-based chunking with overlap (light-touch — most docs are already single-topic) |
| Embedding + indexing | `src/embed_index.py` | `sentence-transformers/all-MiniLM-L6-v2` + FAISS (`IndexFlatIP`, cosine similarity via normalized vectors) |
| Retrieval | `src/retrieve.py` | Top-k similarity search |
| Safeguard | `src/safeguard.py` | Regex-based advice-request detection + retrieval-confidence threshold |
| Generation | `src/generate.py` | `Qwen/Qwen2.5-1.5B-Instruct`, with relative-margin citation filtering |
| Interface | `src/app.py` | Gradio `Blocks` + `ChatInterface`, branded header, conversation history panel, editable messages |
| Evaluation | `eval/evaluate.py` | Retrieval accuracy, latency, safeguard testing, manual answer-correctness grading |

## Why these choices

- **`all-MiniLM-L6-v2`**: a well-established, lightweight sentence embedding
  model widely used as a RAG baseline — good balance of speed and retrieval
  quality for a corpus at this scale, and CPU-viable.
- **FAISS `IndexFlatIP`**: exact (not approximate) search, appropriate given
  the corpus size (52 documents) — no need for approximate indexing
  structures (e.g. IVF, HNSW) at this scale.
- **Qwen2.5-1.5B-Instruct**: chosen over a larger or proprietary model for
  reproducibility on free-tier hardware without licensing constraints — a
  deliberate cost/access trade-off discussed critically alongside the
  GPT-5.5 vs Llama 4 comparison in the accompanying report.

## Safeguard

Two layers:
1. **Advice-request detection** — regex pattern matching on phrases like
   "should I...", "what should I do", "which [X] is best for me", "given my
   situation" — blocks the LLM call entirely and returns a disclaimer
   pointing to Pension Wise / a regulated adviser.
2. **Retrieval-confidence threshold** — if the top retrieved chunk's cosine
   similarity is below `0.35`, the system declines to answer rather than
   risk generating an ungrounded response.

This directly reflects the FCA's guidance/advice distinction. It is a
heuristic, not a guarantee — see [Evaluation results](#evaluation-results)
below for its measured accuracy, including a known precision trade-off.

---

## Evaluation results

Final results from the evaluation suite (`eval/evaluate.py`), run against the
full 52-document knowledge base:

| Metric | Result |
|---|---|
| Retrieval accuracy (top-3, 20 test questions) | **100%** |
| Average retrieval latency | **28.5 ms** |
| Answer correctness (manual grading, 20 answers) | **80% correct, 5% partial, 5% incorrect, 10% safeguard false positive** |
| Safeguard confusion matrix (23 cases: 20 general + 3 advice-seeking) | **91.3% accuracy, 100% recall, 60% precision, 75% F1** |

The safeguard is deliberately tuned for recall over precision: it never
misses a genuine advice-seeking request in testing, at the cost of
occasionally over-blocking a legitimate general question phrased similarly
(e.g. "How much should I have in an emergency fund?"). This is treated as an
acceptable trade-off in a regulated domain, where a false positive (an
over-cautious refusal) is considered less serious than a false negative
(unflagged advice-giving).

Full results, including all 20 individual test cases with manual grades, are
in `eval/retrieval_results.csv`, `eval/generation_results.csv`, and
`eval/safeguard_results.csv`.

## Limitations

- Knowledge base, despite expansion from an initial 27 to 52 documents after
  testing revealed coverage gaps, remains a curated subset — not
  comprehensive UK financial guidance.
- Safeguard pattern-matching is a heuristic: it can miss some advice-seeking
  phrasings and, as measured, over-triggers on some legitimate general
  questions (60% precision).
- Small generation model (1.5B parameters) trades some fluency/reasoning
  depth for reproducibility and cost relative to frontier models.
- Retrieval latency is largely hardware-insensitive, but generation latency
  increases substantially on CPU-only hardware relative to GPU.

---

## Setup & running (Google Colab — recommended)

1. Open `notebooks/finance_rag_colab.ipynb` in Google Colab (or open directly
   via: `colab.research.google.com/github/SheerinMM/finance-guidance-rag/blob/main/notebooks/finance_rag_colab.ipynb`).
2. Runtime → Change runtime type → select a GPU (T4 is sufficient), if available.
3. Run all cells top to bottom. This will:
   - Mount Google Drive and cache Hugging Face models there across sessions
   - Install dependencies
   - Clone/pull the repo
   - Build the FAISS index from `data/finance_kb.json`
   - Run retrieval and safeguard sanity checks
   - Load the generator and launch the Gradio app (public `share=True` link)
   - Run the evaluation suite and display results

## Setup & running (local)

```bash
pip install -r requirements.txt
python src/embed_index.py      # builds and saves the FAISS index
python src/app.py              # launches the Gradio app at http://localhost:7860
python eval/evaluate.py        # runs the evaluation suite, writes CSVs to eval/
```

Requires Python 3.10+. Runs on CPU; a CUDA-capable GPU significantly improves
generation speed.

## Project structure

```
finance-guidance-rag/
├── data/
│   └── finance_kb.json       # knowledge base (52 documents, 16 categories)
├── src/
│   ├── chunking.py
│   ├── embed_index.py
│   ├── retrieve.py
│   ├── safeguard.py
│   ├── generate.py
│   └── app.py
├── eval/
│   ├── test_questions.json     # 20-question test set
│   ├── evaluate.py
│   ├── retrieval_results.csv   # full retrieval evaluation output
│   ├── generation_results.csv  # full generation output with manual grades
│   └── safeguard_results.csv   # dedicated safeguard test output
├── notebooks/
│   └── finance_rag_colab.ipynb
├── requirements.txt
└── README.md
```

## Data sources

The knowledge base content is written in original wording, informed by
general publicly available UK financial guidance topics (of the kind covered
by services such as MoneyHelper, a UK government-backed guidance service).
No content is copied verbatim from any source; all entries are original
summaries of general, publicly known financial concepts, used here for
educational/prototype purposes only.

---

## Project context

Built as the applied project for a university NLP and Generative AI module.
The accompanying report critically compares GPT-5.5 and Llama 4 as
theoretical grounding for the design choices made here (open-weight model
selection, RAG over fine-tuning, safeguard design informed by UK financial
regulation), and evaluates this prototype's performance in detail.
