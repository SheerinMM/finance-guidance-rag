"""
generate.py
Builds a grounded prompt from the retrieved chunks and calls a pretrained LLM
to produce the final answer, with citations back to source titles.
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


def build_prompt(query, retrieved_chunks):
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
    def __init__(self, model_name=GENERATION_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    def generate(self, query, retrieved_chunks, max_new_tokens=200):
        prompt = build_prompt(query, retrieved_chunks)
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        if torch.cuda.is_available():
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
        return response.strip()


def format_answer_with_citations(answer, retrieved_chunks, min_absolute_score=0.35, relative_margin=0.65):
    if not retrieved_chunks:
        return answer
    top_score = retrieved_chunks[0].get("score", 1.0)
    dynamic_threshold = max(min_absolute_score, relative_margin * top_score)
    relevant = [c for c in retrieved_chunks if c.get("score", 1.0) >= dynamic_threshold]
    if not relevant:
        relevant = retrieved_chunks[:1]
    sources = sorted(set(c["title"] for c in relevant))
    citation_block = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
    return answer + citation_block
