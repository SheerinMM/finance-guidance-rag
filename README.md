# Finance Guidance Assistant — RAG Prototype

A retrieval-augmented generation (RAG) chatbot that answers general UK personal
finance questions (pensions, savings, debt, mortgages, credit, investing) using
a small knowledge base of public financial guidance content.

Built for CIS142-6 Natural Language Processing (NLP) and Generative AI — Part B
Applied Project.

---

## What this is (and isn't)

This assistant provides **general financial guidance**, not personalised
financial advice. UK financial regulation legally distinguishes the two: only
FCA-authorised advisers can give personalised recommendations. The system
includes a safeguard that detects advice-seeking questions and redirects the
user to appropriate regulated services instead of answering directly. See
[Safeguard](#safeguard) below.

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
Append source citations
    │
    ▼
Return answer to user
```

## Pipeline components

| Stage | File | Technology |
|---|---|---|
| Knowledge base | `data/finance_kb.json` | 27 hand-written guidance documents across 13 finance categories |
| Chunking | `src/chunking.py` | Word-based chunking with overlap (light-touch — most docs are already single-topic) |
| Embedding + indexing | `src/embed_index.py` | `sentence-transformers/all-MiniLM-L6-v2` + FAISS (`IndexFlatIP`, cosine similarity via normalized vectors) |
| Retrieval | `src/retrieve.py` | Top-k similarity search |
| Safeguard | `src/safeguard.py` | Pattern-based advice-request detection + retrieval-confidence threshold |
| Generation | `src/generate.py` | `Qwen/Qwen2.5-1.5B-Instruct` — small enough for Colab free-tier GPU |
| Interface | `src/app.py` | Gradio `ChatInterface` |
| Evaluation | `eval/evaluate.py` | Retrieval accuracy, latency, safeguard trigger testing |

## Why these choices (for the technical report)

- **`all-MiniLM-L6-v2`**: a well-established, lightweight sentence embedding
  model widely used as a RAG baseline in the literature — good balance of
  speed and retrieval quality for a small corpus like this one.
- **FAISS `IndexFlatIP`**: exact (not approximate) search, appropriate given
  the small corpus size (27 documents / ~27-40 chunks) — no need for
  approximate indexing structures (e.g. IVF, HNSW) at this scale.
- **Qwen2.5-1.5B-Instruct**: chosen over a larger model for reproducibility on
  free-tier Colab GPU. This is a deliberate cost/latency vs. capability
  trade-off, discussed critically in the technical report alongside the
  GPT-5.5 vs Llama 4 comparison in Part A.

## Safeguard

Two layers:
1. **Advice-request detection** — regex pattern matching on phrases like
   "should I...", "what should I do", "given my situation" — blocks the LLM
   call entirely and returns a disclaimer pointing to Pension Wise / a
   regulated adviser.
2. **Retrieval-confidence threshold** — if the top retrieved chunk's cosine
   similarity is below `0.35`, the system declines to answer rather than
   risk generating an ungrounded response.

This directly reflects the FCA's guidance/advice distinction discussed in
Part A, and is evaluated quantitatively in `eval/safeguard_results.csv`.

## Limitations (be upfront about these in the report)

- Knowledge base is a small, hand-curated set (27 docs) — not comprehensive
  UK financial guidance; real deployment would need a much larger, regularly
  updated corpus.
- Safeguard pattern-matching is not exhaustive — it will miss some
  advice-seeking phrasings and may occasionally over-trigger on borderline
  factual questions. This is an explicit trade-off documented for the report.
- Small generation model (1.5B parameters) trades some fluency/reasoning
  quality for reproducibility and cost — discussed as a limitation relative
  to frontier models.

---

## Setup & running (Google Colab — recommended)

1. Open `notebooks/finance_rag_colab.ipynb` in Google Colab.
2. Runtime → Change runtime type → select a GPU (T4 is sufficient).
3. Run all cells top to bottom. This will:
   - Install dependencies
   - Build the FAISS index from `data/finance_kb.json`
   - Launch the Gradio app with a public `share=True` link
4. Run `eval/evaluate.py` (final cell) to regenerate the evaluation CSVs.

## Setup & running (local)

```bash
pip install -r requirements.txt
python src/embed_index.py      # builds and saves the FAISS index
python src/app.py              # launches the Gradio app at http://localhost:7860
python eval/evaluate.py        # runs the evaluation suite, writes CSVs to eval/
```

Requires Python 3.10+ and (ideally) a CUDA-capable GPU for reasonable
generation speed, though it will run on CPU.

## Project structure

```
finance-rag/
├── data/
│   └── finance_kb.json       # knowledge base (27 documents)
├── src/
│   ├── chunking.py
│   ├── embed_index.py
│   ├── retrieve.py
│   ├── safeguard.py
│   ├── generate.py
│   └── app.py
├── eval/
│   ├── test_questions.json   # 20-question test set
│   └── evaluate.py
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
