"""Opt-in end-to-end check against a REAL local model.

Run with:  YAKE_SUM_LIVE=1 [YAKE_SUM_MODEL=mistral] pytest tests/test_live_ollama.py -s
Skipped by default so CI stays deterministic and GPU-free.
"""

import os
from pathlib import Path

import pytest

from yake_sum import Summarizer

pytestmark = pytest.mark.skipif(not os.environ.get("YAKE_SUM_LIVE"), reason="set YAKE_SUM_LIVE=1")
MODEL = os.environ.get("YAKE_SUM_MODEL", "mistral")
SAMPLE = Path("data/samples/paper_yake.txt")


@pytest.mark.parametrize("mode", ["abstractive", "hybrid"])
def test_real_model_produces_faithful_summary(mode):
    s = Summarizer(mode=mode, backend="ollama", model=MODEL, max_context_chars=1500)
    ok, msg = s.llm_client.is_available()
    assert ok, msg
    res = s.summarize(SAMPLE.read_text(encoding="utf-8"))
    print(f"\n[{mode}/{MODEL}] {res.text}\n{res.metrics.get('faithfulness')}")
    assert len(res.text.split()) >= 10
    assert res.metrics["faithfulness"]["unsupported_numbers"] == []
    assert res.metrics["faithfulness"]["supported_ratio"] >= 0.6
    assert res.metrics.get("keyword_coverage", 0) > 0
