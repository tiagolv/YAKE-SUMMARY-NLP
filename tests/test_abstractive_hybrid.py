import pytest
from yake_sum.backends.mock import MockLLMClient
from yake_sum.abstractive.summarizer import AbstractiveSummarizer
from yake_sum.hybrid.summarizer import HybridSummarizer

def test_abstractive_with_mock():
    client = MockLLMClient()
    summarizer = AbstractiveSummarizer(llm_client=client)
    doc = (
        "Transformer models use multi-head self-attention mechanisms to process text. "
        "They have replaced recurrent neural networks in natural language processing."
    )
    res = summarizer.summarize(doc)
    assert res.summary != ""
    assert len(res.keywords) > 0
    assert "Keywords:" in res.prompt_used
    assert "Document:" in res.prompt_used

def test_abstractive_empty_text():
    client = MockLLMClient()
    summarizer = AbstractiveSummarizer(llm_client=client)
    res = summarizer.summarize("")
    assert res.summary == ""
    assert res.keywords == []

def test_hybrid_long_document():
    client = MockLLMClient()
    # 20 distinct paragraphs
    paragraphs = [
        f"Paragraph {i}: Content about artificial intelligence, neural networks, and machine learning in topic {i}."
        for i in range(20)
    ]
    long_text = "\n\n".join(paragraphs)
    summarizer = HybridSummarizer(llm_client=client, max_context_chars=600)
    res = summarizer.summarize(long_text)
    assert res.summary != ""
    assert len(res.retained_passages) > 0
    assert len(res.retained_passages) < 20
    assert res.context_char_count <= 800
    # Retained passages should maintain their original chronological order
    first_p = res.retained_passages[0]
    second_p = res.retained_passages[1]
    assert long_text.index(first_p) < long_text.index(second_p)


def test_hybrid_context_budget_is_strict_for_oversized_passage():
    long_passage = "important " * 200
    summarizer = HybridSummarizer(llm_client=MockLLMClient(), max_context_chars=80)

    result = summarizer.summarize(long_passage)

    assert result.context_char_count <= 80
    assert len(result.retained_passages) == 1
    assert 0 < len(result.retained_passages[0]) <= 80
