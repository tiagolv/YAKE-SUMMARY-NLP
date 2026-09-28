from pathlib import Path
import pytest
from benchmarks.run_rouge_study import run_rouge_benchmark, compute_lead_3

def test_compute_lead_3():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    lead3 = compute_lead_3(text)
    assert lead3 == "Sentence one. Sentence two. Sentence three."

def test_benchmark_runner_with_mock():
    benchmark_path = Path("data/eval/benchmark_with_summaries.json")
    results = run_rouge_benchmark(
        benchmark_file=benchmark_path,
        backend="mock",
        max_samples=3,
    )
    assert "methods" in results
    assert "lead_3" in results["methods"]
    assert "yake_extractive" in results["methods"]
    assert "llm_unconditioned" in results["methods"]
    assert "yake_llm_guided" in results["methods"]
    assert "oracle_llm_guided" in results["methods"]
    assert len(results["sample_ids"]) == 3
    # Check that average metrics exist for all methods
    for method in results["methods"]:
        m = results["methods"][method]
        assert "rouge1_f" in m
        assert "rouge2_f" in m
        assert "rougeL_f" in m
