"""Faithfulness signal, facade warnings and configuration precedence."""

from yake_sum import Summarizer, source_support
from yake_sum.backends import OllamaClient


class _Fixed:
    def __init__(self, text):
        self.text = text

    def generate(self, *a, **k):
        return self.text

    def is_available(self):
        return True, ""


SRC = "The model reached 91% accuracy on CIFAR with transformers. Training took 12 hours."


def test_source_support_flags_invented_numbers():
    res = source_support("The model reached 95% accuracy using 12 hours of training.", SRC)
    assert res["unsupported_numbers"] == ["95"]
    assert 0.0 < res["supported_ratio"] < 1.0


def test_source_support_faithful_summary_and_empty_cases():
    ok = source_support("Transformers reached 91% accuracy on CIFAR.", SRC)
    assert ok["unsupported_numbers"] == [] and ok["supported_ratio"] == 1.0
    assert source_support("", SRC)["supported_ratio"] == 0.0
    assert source_support("text", "")["supported_ratio"] == 0.0


def test_facade_surfaces_hallucinated_numbers_as_warning():
    s = Summarizer(mode="abstractive", llm_client=_Fixed("Accuracy was 99% on CIFAR with transformers."))
    res = s.summarize(SRC)
    assert res.metrics["faithfulness"]["unsupported_numbers"] == ["99"]
    assert any("99" in w for w in res.metrics["warnings"])


def test_facade_warns_when_abstractive_input_exceeds_budget():
    long = " ".join(f"Sentence {i} discusses graph databases and query languages." for i in range(200))
    res = Summarizer(mode="abstractive", backend="mock", max_context_chars=500).summarize(long)
    assert any("hybrid" in w for w in res.metrics["warnings"])
    hybrid = Summarizer(mode="hybrid", backend="mock", max_context_chars=500).summarize(long)
    assert not any("hybrid" in w for w in hybrid.metrics.get("warnings", []))


def test_extractive_has_no_spurious_faithfulness_or_warnings():
    res = Summarizer(mode="extractive", num_sentences=1).summarize(
        "Graph databases store edges natively. They suit fraud detection."
    )
    assert "faithfulness" not in res.metrics and "warnings" not in res.metrics


def test_ollama_host_precedence(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://envhost:1")
    assert OllamaClient(host="http://explicit:2").host == "http://explicit:2"  # explicit wins
    assert OllamaClient().host == "http://envhost:1"  # env beats default
    monkeypatch.delenv("OLLAMA_HOST")
    assert OllamaClient().host == "http://localhost:11434"
    assert OllamaClient(host="envless:3").host == "http://envless:3"
