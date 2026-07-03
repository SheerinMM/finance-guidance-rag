"""
generate.py
Builds a grounded prompt from the retrieved chunks and calls a pretrained LLM
to produce the final answer, with citations back to source titles.

Model choice: a small open-weight instruction-tuned model that runs on Colab's
free-tier GPU, e.g. "Qwen/Qwen2.5-1.5B-Instruct" or "meta-llama/Llama-3.2-3B-Instruct".
Qwen2.5-1.5B-Instruct is used as the default here because it is small enough to
run reliably on free-tier hardware while still following instructions well —
an important practical constraint to discuss in the technical report (cost/
latency trade-off vs a frontier model like GPT-5.5 or a larger Llama 4 variant).
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

GENERATION_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a financial guidance assistant for UK consumers. Answer ONLY using "
    "the information provided in the context below. If the context does not "
    "contain enough information to answer, say so clearly rather than guessing. "
    "Do not give personalised financial advice or recommend specific products for "
    "the user's individual situation — explain general options instead. "
    "Keep answers concise and in plain English."
)


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['title']}]\n{c['text']}" for c in retrieved_chunks
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


class Generator:
    def __init__(self, model_name: str = GENERATION_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    def generate(self, query: str, retrieved_chunks: list[dict], max_new_tokens: int = 200) -> str:
        prompt = build_prompt(query, retrieved_chunks)
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if torch.cuda.is_available():
            inputs = inputs.to(self.model.device)

        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[-1]:], skip_special_tokens=True
        )
        return response.strip()


def format_answer_with_citations(answer: str, retrieved_chunks: list[dict]) -> str:
    """Appends a citation footer listing the source titles used."""
    sources = sorted(set(c["title"] for c in retrieved_chunks))
    citation_block = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
    return answer + citation_block


if __name__ == "__main__":
    # Quick prompt-construction sanity check (does not call the model).
    dummy_chunks = [
        {"title": "What is a Lifetime ISA?", "text": "A Lifetime ISA (LISA) is designed to help people aged 18 to 39..."}
    ]
    print(build_prompt("What is a Lifetime ISA?", dummy_chunks))
