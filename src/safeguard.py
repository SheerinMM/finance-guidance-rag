"""
safeguard.py
Implements the required safeguard for this prototype: a guardrail that detects
requests for personalised, regulated financial advice (e.g. "should I move my
pension into X", "what should I invest in given my situation") and refuses to
answer directly, returning a standard disclaimer instead of a generated response.

Why this safeguard specifically (for the technical report):
UK financial guidance is legally distinguished from regulated financial advice.
Only FCA-authorised advisers may give personalised recommendations. A RAG chatbot
answering general questions from public guidance content is providing guidance,
not advice — but LLMs can easily drift into advice-like language if not
constrained (e.g. "you should transfer your pension to X"). This safeguard keeps
the system inside the guidance boundary rather than the advice boundary,
directly linked to the ethical/regulatory discussion in Part A.

A secondary, lower-level safeguard (grounding refusal) is also included:
if retrieval confidence is too low, the system declines to answer rather than
letting the LLM generate an ungrounded (and possibly hallucinated) response.
"""

import re

# Phrases/patterns that signal a request for personalised advice rather than
# general guidance. Not exhaustive — documented in the report as a limitation.
ADVICE_PATTERNS = [
    r"\bshould i\b",
    r"\bwhat should i do\b",
    r"\bis it (a )?good idea for me\b",
    r"\brecommend\b.*\b(for me|to me)\b",
    r"\bwhich (one|option|fund|provider)\b.*\b(should|is best for me)\b",
    r"\bgiven my (situation|circumstances|salary|age|pension)\b",
    r"\bwhat would you do\b",
]

DISCLAIMER = (
    "I can't give personalised financial advice — that requires an FCA-regulated "
    "adviser who can assess your individual circumstances. What I can do is explain "
    "general guidance on this topic. For personalised recommendations, consider a "
    "free Pension Wise appointment (if pension-related) or a regulated financial "
    "adviser. Would you like me to explain the general options instead?"
)

LOW_CONFIDENCE_MESSAGE = (
    "I don't have enough relevant information in my knowledge base to answer that "
    "confidently. Rather than guess, I'd rather say I don't know than risk giving "
    "you an inaccurate answer on a financial matter."
)

# Minimum cosine similarity for the top retrieved chunk before we trust it
# enough to generate an answer. Tuned during evaluation (see eval/).
MIN_RETRIEVAL_CONFIDENCE = 0.35


def is_advice_request(query: str) -> bool:
    """Returns True if the query pattern-matches a personalised-advice request."""
    q = query.lower()
    return any(re.search(pattern, q) for pattern in ADVICE_PATTERNS)


def check_safeguards(query: str, retrieved_chunks: list[dict]) -> tuple[bool, str | None]:
    """
    Runs both safeguard checks.
    Returns (blocked, message):
      - (True, DISCLAIMER) if the query requests personalised advice
      - (True, LOW_CONFIDENCE_MESSAGE) if retrieval confidence is too low
      - (False, None) if the query is safe to answer normally
    """
    if is_advice_request(query):
        return True, DISCLAIMER

    if not retrieved_chunks or retrieved_chunks[0]["score"] < MIN_RETRIEVAL_CONFIDENCE:
        return True, LOW_CONFIDENCE_MESSAGE

    return False, None


if __name__ == "__main__":
    test_cases = [
        "Should I transfer my pension into a SIPP given my situation?",
        "What is a Lifetime ISA?",
        "What should I do with my inheritance?",
    ]
    for q in test_cases:
        print(f"Query: {q}")
        print(f"  Advice request detected: {is_advice_request(q)}\n")
