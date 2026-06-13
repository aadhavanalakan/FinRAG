"""
faithfulness.py — LLM-as-judge faithfulness scoring.

Two stages, using the versioned judge prompt (config/prompts/faithfulness.yaml):
  1. extract the answer's atomic claims
  2. for each claim, judge whether the retrieved context supports it
Score = supported claims / total claims. A refusal (no claims) scores None and is
excluded from the faithfulness average.

Judge model = the generation model (Qwen3 via Nebius), temperature 0.
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from finrag.config import AppConfig


def _parse_json(text: str) -> dict:
    """Pull the first {...} object out of the model's reply (tolerates code fences)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


class FaithfulnessJudge:
    def __init__(self, cfg: AppConfig) -> None:
        g = cfg.models.generation
        self.model = g.model
        self.prompt = cfg.prompts.faithfulness
        self._client = OpenAI(base_url=g.base_url, api_key=g.api_key())

    def _chat(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=512,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content.strip()

    def extract_claims(self, answer: str) -> list[str]:
        out = self._chat(self.prompt.extract.system, self.prompt.extract.user.format(answer=answer))
        return _parse_json(out).get("claims", [])

    def is_supported(self, context: str, claim: str) -> bool:
        out = self._chat(self.prompt.verdict.system,
                         self.prompt.verdict.user.format(context=context, claim=claim))
        return _parse_json(out).get("verdict") == "supported"

    def score(self, answer: str, context: str) -> dict:
        """Return {faithfulness, n_claims, supported}. faithfulness is None for refusals."""
        claims = self.extract_claims(answer)
        if not claims:
            return {"faithfulness": None, "n_claims": 0, "supported": 0}
        supported = sum(self.is_supported(context, c) for c in claims)
        return {"faithfulness": supported / len(claims), "n_claims": len(claims), "supported": supported}
