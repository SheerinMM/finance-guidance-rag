"""
safeguard.py
Implements the required safeguard for this prototype: a guardrail that detects
requests for personalised, regulated financial advice and refuses to answer
directly, returning a standard disclaimer instead of a generated response.
"""

import re

ADVICE_PATTERNS = [
    r"\bshould i\b",
    r"\bwhat should i do\b",
    r"\bis it (a )?good idea for me\b",
    r"\brecommend\b.*\b(for me|to me)\b",
    r"\bwhich\b.*\b(should i|is best for me|is right for me|suits me)\b",
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

MIN_RETRIEVAL_CONFIDENCE = 0.35


def is_advice_request(query: str) -> bool:
    q = query.lower()
    return any(re.search(pattern, q) for pattern in ADVICE_PATTERNS)


def check_safeguards(query: str, retrieved_chunks: list[dict]) -> tuple[bool, str | None]:
    if is_advice_request(query):
        return True, DISCLAIMER
    if not retrieved_chunks or retrieved_chunks[0]["score"] < MIN_RETRIEVAL_CONFIDENCE:
        return True, LOW_CONFIDENCE_MESSAGE
    return False, None


if __name__ == "__main__":
    test_cases = [
        "Should I transfer my pension into a SIPP given my situation?",
        "What is a Lifetime ISA?",
        "Which mortgage is best for me?",
    ]
    for q in test_cases:
        print(f"Query: {q}")
        print(f"  Advice request detected: {is_advice_request(q)}\n")
