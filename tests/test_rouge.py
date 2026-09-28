import pytest
from yake_sum.evaluation.rouge import compute_rouge_metrics

def test_compute_rouge_metrics_identical_text():
    text = "Machine learning enables computers to learn from data."
    metrics = compute_rouge_metrics(summary=text, reference=text)
    assert metrics["rouge1_f"] == pytest.approx(1.0)
    assert metrics["rouge2_f"] == pytest.approx(1.0)
    assert metrics["rougeL_f"] == pytest.approx(1.0)
    assert metrics["rouge1_p"] == pytest.approx(1.0)
    assert metrics["rouge1_r"] == pytest.approx(1.0)

def test_compute_rouge_metrics_empty_inputs():
    metrics = compute_rouge_metrics(summary="", reference="Some text")
    assert metrics["rouge1_f"] == 0.0
    assert metrics["rouge2_f"] == 0.0
    assert metrics["rougeL_f"] == 0.0

def test_compute_rouge_metrics_partial_overlap():
    summary = "Deep learning uses neural networks."
    reference = "Deep learning models are based on neural networks and gradient descent."
    metrics = compute_rouge_metrics(summary=summary, reference=reference)
    assert 0.0 < metrics["rouge1_f"] < 1.0
    assert 0.0 < metrics["rouge2_f"] <= 1.0
    assert 0.0 < metrics["rougeL_f"] <= 1.0
