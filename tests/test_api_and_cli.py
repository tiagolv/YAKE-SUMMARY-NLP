import json
import pytest
from yake_sum import Summarizer
from yake_sum.cli import main as cli_main

def test_unified_summarizer_extractive():
    sum_ext = Summarizer(mode="extractive", num_sentences=2)
    res = sum_ext.summarize("First sentence about NLP. Second sentence about AI. Third sentence about data.")
    assert res.mode == "extractive"
    assert len(res.text) > 0
    assert len(res.keywords) > 0
    assert "compression_ratio" in res.metrics

def test_unified_summarizer_abstractive_with_rouge():
    reference = "Transformer models use self-attention for NLP."
    sum_abs = Summarizer(mode="abstractive", backend="mock")
    res = sum_abs.summarize(
        text="Transformer models replace RNNs using self-attention.",
        reference_summary=reference,
    )
    assert res.mode == "abstractive"
    assert "rouge" in res.metrics
    assert res.metrics["rouge"]["rouge1_f"] > 0.0

def test_unified_summarizer_hybrid():
    sum_hyb = Summarizer(mode="hybrid", backend="mock", max_context_chars=500)
    text = "Paragraph 1 about ML.\n\nParagraph 2 about AI.\n\nParagraph 3 about statistics."
    res = sum_hyb.summarize(text)
    assert res.mode == "hybrid"
    assert len(res.text) > 0

def test_cli_extractive_execution(tmp_path, capsys):
    test_doc = tmp_path / "test.txt"
    test_doc.write_text("Sentence one on YAKE. Sentence two on natural language processing. Sentence three on metrics.", encoding="utf-8")
    
    # Run CLI in extractive mode
    exit_code = cli_main(["--input", str(test_doc), "--mode", "extractive", "--sentences", "1", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["mode"] == "extractive"
    assert len(data["summary"]) > 0
