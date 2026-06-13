"""Judge router — POST /api/judge

Uses Qwen (via Ollama) as an LLM-as-a-Judge to evaluate summaries on four
dimensions: fidelity, coverage, coherence, and keyword relevance.
Each dimension is scored 1–5 with a textual justification.
"""

from __future__ import annotations

import json
import re

import requests
from fastapi import APIRouter, HTTPException

from ..schemas import JudgeDimension, JudgeRequest, JudgeResult
from .deps import get_raw_config

router = APIRouter(prefix="/api", tags=["judge"])

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
    return (
        f"DOCUMENT:\n{document[:3000]}\n\n"
        f"KEYWORDS: [{kw_str}]\n\n"
        f"SUMMARY TO EVALUATE:\n{summary}\n\n"
        "Evaluate the summary and respond with the JSON object only."
    )


def _extract_json(raw: str) -> dict:
    """Extract the first JSON object from a string, tolerating markdown fences."""
    # Strip ```json ... ``` fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # Find the first { ... } block
    match = re.search(r"\{[\s\S]+\}", raw)
    if not match:
        raise ValueError(f"No JSON object found in response: {raw[:200]}")
    return json.loads(match.group())


def _call_ollama(host: str, model: str, prompt: str) -> str:
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": f"{JUDGE_SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 600},
    }
    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Ollama inacessível: {exc}")
    return resp.json().get("response", "")


@router.post("/judge", response_model=JudgeResult)
def judge(req: JudgeRequest) -> JudgeResult:
    """Evaluate a summary using Qwen as an LLM judge."""
    raw_cfg = get_raw_config()
    ollama_host = raw_cfg["llm"].get("ollama_host", "http://localhost:11434")

    prompt = _build_judge_prompt(req.document, req.summary, req.keywords)
    raw_response = _call_ollama(ollama_host, req.judge_model, prompt)

    try:
        data = _extract_json(raw_response)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Judge returned unparseable response: {exc}. Raw: {raw_response[:300]}",
        )

    def _dim(key: str) -> JudgeDimension:
        try:
            return JudgeDimension(
                score=int(data[key]["score"]),
                justification=str(data[key]["justification"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Missing or invalid field '{key}' in judge response: {exc}",
            )

    fidelity = _dim("fidelity")
    coverage = _dim("coverage")
    coherence = _dim("coherence")
    kw_rel = _dim("keyword_relevance")

    overall = round((fidelity.score + coverage.score + coherence.score + kw_rel.score) / 4, 2)

    return JudgeResult(
        fidelity=fidelity,
        coverage=coverage,
        coherence=coherence,
        keyword_relevance=kw_rel,
        overall=overall,
        model_used=req.judge_model,
        raw_response=raw_response,
    )