"""Robustness tests mapped to the plan's 'Review Focus' items. No network, no GPU."""

import json

import pytest
import requests

from yake_sum import Summarizer
from yake_sum.backends import (
    BackendConnectionError,
    BackendError,
    BackendTimeoutError,
    MockLLMClient,
    OllamaClient,
    get_llm_client,
)
from yake_sum.evaluation import compute_rouge_metrics, keyword_coverage, keyword_prf
from yake_sum.extractive.summarizer import ExtractiveSummarizer
from yake_sum.hybrid.summarizer import HybridSummarizer
from yake_sum.abstractive.summarizer import AbstractiveSummarizer
from yake_sum.text import split_into_sentences


# ---- 1. empty / whitespace / fewer sentences than requested ------------------------
@pytest.mark.parametrize("text", ["", "   \n\t  ", "\n\n"])
def test_all_engines_handle_blank_input(text):
    assert ExtractiveSummarizer().summarize(text).summary == ""
    assert AbstractiveSummarizer(MockLLMClient()).summarize(text).summary == ""
    assert HybridSummarizer(MockLLMClient()).summarize(text).summary == ""


def test_fewer_sentences_than_requested_returns_all():
    res = ExtractiveSummarizer(num_sentences=5).summarize("One idea here. Another idea there.")
    assert res.selected_sentences == ["One idea here.", "Another idea there."]


# ---- 2. YAKE returns no keywords -> safe fallback ------------------------------------
def test_stopword_only_text_falls_back_to_lead():
    text = "It is what it is. It was and it is. To be or not to be. It is so."
    res = ExtractiveSummarizer(num_sentences=2).summarize(text)
    assert len(res.selected_sentences) == 2  # no exception, lead-2 behaviour


def test_abstractive_without_keywords_uses_unconditioned_prompt():
    res = AbstractiveSummarizer(MockLLMClient()).summarize("Ok.")
    assert res.keywords == []
    assert "Keywords:" not in res.prompt_used
    assert res.summary


# ---- 3. no n-gram overlap -> ROUGE 0.0, no ZeroDivisionError -----------------------
def test_rouge_zero_overlap():
    m = compute_rouge_metrics("alpha beta gamma", "delta epsilon zeta")
    assert m["rouge1_f"] == 0.0 and m["rouge2_f"] == 0.0 and m["rougeL_f"] == 0.0


def test_keyword_metrics_edge_cases():
    assert keyword_coverage("", ["a"]) == 0.0
    assert keyword_coverage("text", []) == 0.0
    assert keyword_coverage("Recommender systems rock", ["recommender system"]) == 1.0  # stemmed
    assert keyword_prf([], ["a"])["f1"] == 0.0
    assert keyword_prf(["Graph databases", "graph databases"], ["graph database"])["precision"] == 1.0


# ---- 4. hybrid: budget respected, chronological order preserved -----------------------
def test_hybrid_budget_and_order():
    paras = [
        f"Paragraph {i} discusses neural networks and quantum computing topic {i}. "
        f"Filler words about weather number {i}." for i in range(30)
    ]
    text = "\n\n".join(paras)
    res = HybridSummarizer(MockLLMClient(), max_context_chars=500).summarize(text)
    assert 0 < res.context_char_count <= 500
    positions = [text.index(p) for p in res.retained_passages]
    assert positions == sorted(positions)


def test_hybrid_single_giant_paragraph_selects_sentences_not_just_the_head():
    sentences = [f"Sentence {i} is filler about nothing in particular." for i in range(40)]
    sentences[35] = "Federated learning preserves privacy while federated learning trains models."
    sentences[36] = "Federated learning and privacy preserving aggregation matter."
    text = " ".join(sentences)
    res = HybridSummarizer(MockLLMClient(), max_context_chars=400).summarize(text)
    assert res.context_char_count <= 400
    assert "Federated learning" in " ".join(res.retained_passages)


# ---- 5. Ollama unreachable -> specific error, bounded retries -------------------------
class _Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body or {}, text

    def json(self):
        return self._body


def _client(post, retries=2):
    sleeps = []
    c = OllamaClient(model="m", host="http://x:1", max_retries=retries, post=post, sleep=sleeps.append)
    return c, sleeps


def test_ollama_unreachable_raises_backend_connection_error_with_advice():
    calls = []

    def post(*a, **k):
        calls.append(1)
        raise requests.ConnectionError("refused")

    c, sleeps = _client(post)
    with pytest.raises(BackendConnectionError) as ei:
        c.generate("hi")
    assert len(calls) == 3 and sleeps == [1.0, 2.0]  # 1 try + 2 retries, exp. backoff
    msg = str(ei.value)
    assert "ollama serve" in msg and "http://x:1" in msg and "'m'" in msg


def test_ollama_success_after_transient_failure():
    seq = [_Resp(503), requests.ConnectionError("x"), _Resp(200, {"response": " ok "})]

    def post(*a, **k):
        r = seq.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    c, _ = _client(post, retries=3)
    assert c.generate("hi") == "ok"


def test_ollama_timeout_and_non_retryable():
    def slow(*a, **k):
        raise requests.Timeout("slow")

    c, _ = _client(slow, retries=1)
    with pytest.raises(BackendTimeoutError):
        c.generate("hi")

    calls = []

    def not_found(*a, **k):
        calls.append(1)
        return _Resp(404, text="model not found")

    c, _ = _client(not_found)
    with pytest.raises(BackendError, match="404"):
        c.generate("hi")
    assert len(calls) == 1  # never retried


def test_empty_llm_response_is_an_error_not_a_summary():
    class Empty:
        def generate(self, *a, **k):
            return "  "

        def is_available(self):
            return True, ""

    with pytest.raises(BackendError):
        AbstractiveSummarizer(Empty()).summarize("A document about transformers and attention.")


# ---- validation ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "factory",
    [
        lambda: ExtractiveSummarizer(num_sentences=0),
        lambda: ExtractiveSummarizer(top_k=0),
        lambda: HybridSummarizer(max_context_chars=5),
        lambda: AbstractiveSummarizer(temperature=-1),
        lambda: Summarizer(mode="nope"),
    ],
)
def test_invalid_settings_raise_value_error(factory):
    with pytest.raises(ValueError):
        factory()


def test_llama_cpp_requires_gguf_path():
    with pytest.raises(ValueError, match="gguf"):
        get_llm_client("llama_cpp", model="mistral")


# ---- sentence splitting + extractive quality --------------------------------------------
def test_sentence_splitter_protects_abbreviations():
    s = split_into_sentences(
        "Smith et al. showed gains (Fig. 2). Dr. Lee agrees. J. Doe found 3.5% more. Done."
    )
    assert s == [
        "Smith et al. showed gains (Fig. 2).",
        "Dr. Lee agrees.",
        "J. Doe found 3.5% more.",
        "Done.",
    ]


def test_extractive_avoids_redundant_sentences():
    text = (
        "Federated learning trains machine learning models across many devices without sharing raw data. "
        "Federated learning trains machine learning models across many devices while preserving privacy. "
        "Differential privacy adds calibrated noise to protect individual federated learning contributions. "
        "The weather was pleasant today in the valley. Lunch was served at noon in the cafeteria."
    )
    penalized = ExtractiveSummarizer(num_sentences=2, redundancy_decay=0.0).summarize(text)
    unpenalized = ExtractiveSummarizer(num_sentences=2, redundancy_decay=1.0).summarize(text)
    # Without a redundancy penalty, the two near-duplicate opening sentences both win.
    assert unpenalized.selected_sentences[0].startswith("Federated learning trains")
    assert unpenalized.selected_sentences[1].startswith("Federated learning trains")
    # With the penalty, the near-duplicate is swapped for the sentence adding new coverage.
    assert any("Differential privacy" in s for s in penalized.selected_sentences)
    assert penalized.selected_sentences != unpenalized.selected_sentences


def test_extractive_is_deterministic():
    t = "Alpha systems scale well. Beta systems fail often. Alpha systems need tuning. Gamma is unrelated."
    a = ExtractiveSummarizer(num_sentences=2).summarize(t)
    b = ExtractiveSummarizer(num_sentences=2).summarize(t)
    assert a.summary == b.summary
