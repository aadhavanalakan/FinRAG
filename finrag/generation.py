"""
generation.py — answer a question from retrieved context, with citations (Nebius LLM).

Assembles the final chunks into a prompt per config/pipeline.yaml `context_assembly`
(best chunk first AND last — "lost in the middle"; each chunk labeled with its id so
the model can cite it), fills the versioned `rag_answer` prompt, and calls the Nebius
generator at temperature 0. The model is instructed to answer ONLY from context and
to refuse if the answer isn't there.
"""

from __future__ import annotations

from openai import OpenAI

from finrag.config import AppConfig
from finrag.retrieval import Retrieved


def _lost_in_the_middle(items: list[Retrieved]) -> list[Retrieved]:
    """Reorder a best-first list so the best chunks sit at the ENDS, weakest in the
    middle (LLMs attend most to start/end). best -> front, 2nd best -> very end."""
    evens = items[0::2]              # ranks 0,2,4 -> front, in order
    odds = items[1::2]              # ranks 1,3   -> back, reversed
    return evens + odds[::-1]


class Generator:
    def __init__(self, cfg: AppConfig) -> None:
        g = cfg.models.generation
        self.g = g
        self.prompt = cfg.prompts.rag_answer
        self.ca = cfg.pipeline.context_assembly
        self._client = OpenAI(base_url=g.base_url, api_key=g.api_key())

    def assemble_context(self, items: list[Retrieved]) -> str:
        """Select chunks that fit the char budget (in relevance order), then reorder
        for lost-in-the-middle, labeling each with its id. Returns the inner body;
        the prompt template wraps it in <context>...</context>."""
        chosen, used = [], 0
        for r in items:
            label = f"[chunk:{r.id}]\n" if self.ca.include_chunk_ids else ""
            block = f"{label}{r.text}"
            if chosen and used + len(block) > self.ca.max_context_chars:
                break
            chosen.append((r, block))
            used += len(block)
        ordered = _lost_in_the_middle([r for r, _ in chosen])
        by_id = {r.id: block for r, block in chosen}
        return "\n\n".join(by_id[r.id] for r in ordered)

    def generate(self, question: str, items: list[Retrieved]) -> str:
        context = self.assemble_context(items)
        user = self.prompt.user.format(context=context, question=question)
        resp = self._client.chat.completions.create(
            model=self.g.model,
            temperature=self.g.temperature,
            max_tokens=self.g.max_tokens,
            messages=[
                {"role": "system", "content": self.prompt.system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()
