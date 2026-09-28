import pytest
from yake_sum.extractive.summarizer import ExtractiveSummarizer

def test_extractive_summarization_basic():
    text = (
        "YAKE is a lightweight unsupervised keyword extraction algorithm. "
        "It uses statistical text features to identify important keywords in documents. "
        "Natural language processing systems often rely on keyword extractors. "
        "Today is a sunny and pleasant day in Lisbon. "
        "Evaluation of keyword quality is typically performed using gold standards."
    )
    summarizer = ExtractiveSummarizer(num_sentences=2, top_k=5)
    result = summarizer.summarize(text)
    assert len(result.selected_sentences) == 2
    assert "YAKE is a lightweight" in result.summary
    assert len(result.keywords) > 0
    assert result.compression_ratio > 0.0

def test_extractive_summarization_empty_and_short():
    summarizer = ExtractiveSummarizer(num_sentences=3)
    res_empty = summarizer.summarize("")
    assert res_empty.summary == ""
    assert res_empty.selected_sentences == []

    res_short = summarizer.summarize("Single sentence document.")
    assert res_short.summary == "Single sentence document."
    assert len(res_short.selected_sentences) == 1

def test_extractive_sentence_ordering():
    text = (
        "Sentence one introduces the machine learning model. "
        "Sentence two details experimental setups. "
        "Sentence three describes results of the machine learning model. "
        "Sentence four concludes the study."
    )
    summarizer = ExtractiveSummarizer(num_sentences=2)
    result = summarizer.summarize(text)
    # The summary sentences should appear in the original document order
    assert len(result.selected_sentences) == 2
    first_idx = text.index(result.selected_sentences[0])
    second_idx = text.index(result.selected_sentences[1])
    assert first_idx < second_idx
