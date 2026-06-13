"""Reusable judge module using the same logic as backend/routers/judge.py.
Works standalone for batch evaluation.
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of text summaries.
You will evaluate a summary against the original document on four dimensions.
You MUST respond ONLY with a valid JSON object — no markdown, no preamble, no explanation outside the JSON.

JSON schema (all fields required):
{
  "fidelity":          {"score": <1-5>, "justification": "<string>"},
  "coverage":          {"score": <1-5>, "justification": "<string>"},
  "coherence":         {"score": <1-5>, "justification": "<string>"},
  "keyword_relevance": {"score": <1-5>, "justification": "<string>"}
}

Scoring rubric:
- fidelity:          Does the summary introduce facts NOT present in the document? (5 = no hallucinations, 1 = many fabrications)
- coverage:          Are the key concepts from the document represented in the summary? (5 = excellent coverage, 1 = major gaps)
- coherence:         Is the summary fluent, well-structured, and internally consistent? (5 = excellent, 1 = incoherent)
- keyword_relevance: Are the provided keywords genuinely central to the document's topic? (5 = all highly relevant, 1 = irrelevant)
"""


def _build_judge_prompt(document: str, summary: str, keywords: list[str]) -> str:
    kw_str = ", ".join(keywords) if keywords else "(none provided)"
    # Limit document length to avoid context overflow
    doc_preview = document[:3000]
    return (
        f"DOCUMENT:\n{doc_preview}\n\n"
        f"KEYWORDS: [{kw_str}]\n\n"
        f"SUMMARY TO EVALUATE:\n{summary}\n\n"
        "Return ONLY the JSON object, no other text."
    )


def _extract_json(raw: str) -> dict:
    """Extract the first valid JSON object from a string (tolerates markdown, extra text)."""
    # Remove markdown code fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # Find the first { ... } block
    match = re.search(r"\{[\s\S]+\}", raw)
    if not match:
        raise ValueError(f"No JSON object found in: {raw[:200]}")
    json_str = match.group()
    # Fix common issues: trailing commas before } or ]
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    return json.loads(json_str)


def _call_ollama(host: str, model: str, prompt: str) -> str:
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": f"{JUDGE_SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 800},
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json().get("response", "")


class Judge:
    def __init__(self, model: str = "mistral", ollama_host: str = "http://localhost:11434"):
        self.model = model
        self.host = ollama_host

    def evaluate(self, document: str, summary: str, keywords: list[str]) -> dict[str, Any]:
        """Run judge evaluation and return structured scores (same format as API)."""
        prompt = _build_judge_prompt(document, summary, keywords)
        raw_response = _call_ollama(self.host, self.model, prompt)

        try:
            data = _extract_json(raw_response)
        except (ValueError, json.JSONDecodeError) as exc:
            # Fallback: scores neutros com justificação de erro
            data = {
                "fidelity": {"score": 3, "justification": f"Parse error: {exc}"},
                "coverage": {"score": 3, "justification": f"Parse error: {exc}"},
                "coherence": {"score": 3, "justification": f"Parse error: {exc}"},
                "keyword_relevance": {"score": 3, "justification": f"Parse error: {exc}"},
            }

        # Garantir que os scores estão dentro de 1-5
        for key in ["fidelity", "coverage", "coherence", "keyword_relevance"]:
            score = data.get(key, {}).get("score", 3)
            if not isinstance(score, (int, float)):
                score = 3
            score = max(1, min(5, int(round(score))))
            data[key] = {"score": score, "justification": data.get(key, {}).get("justification", "")[:500]}

        overall = round((data["fidelity"]["score"] + data["coverage"]["score"] +
                         data["coherence"]["score"] + data["keyword_relevance"]["score"]) / 4, 2)

        return {
            "fidelity": data["fidelity"],
            "coverage": data["coverage"],
            "coherence": data["coherence"],
            "keyword_relevance": data["keyword_relevance"],
            "overall": overall,
            "raw_response": raw_response[:1000],  # limitado
        }