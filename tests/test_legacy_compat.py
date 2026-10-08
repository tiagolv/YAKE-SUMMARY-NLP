"""Legacy pipeline + FastAPI compatibility, running on top of ``yake_sum`` (offline)."""

import json
from pathlib import Path

import pytest
import yaml

from src.evaluator import compute_rouge, keyword_coverage, keyword_precision_recall
from src.keyword_extractor import YakeConfig, YakeKeywordExtractor
from src.llm_interface import LLMConfig, LLMError, create_llm_client
from src.pipeline import run_pipeline
from src.preprocessor import fit_context, split_sentences

DOC = (
    "Graph databases store nodes and edges natively. They suit knowledge graphs and fraud detection. "
    "Relational databases need costly joins for multi-hop queries. Cypher is a declarative query language. "
    "Scaling graph databases horizontally remains difficult."
)


@pytest.fixture()
def mock_config(tmp_path):
    cfg = {
        "yake": {"language": "en", "max_ngram_size": 3, "deduplication_threshold": 0.9, "top_k": 8},
        "llm": {"backend": "mock"},
        "prompt": {"system": "Sys.", "instruction": "Summarize.", "max_text_chars": 3000},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_legacy_pipeline_output_format(tmp_path, mock_config):
    src = tmp_path / "doc.txt"
    src.write_text(DOC, encoding="utf-8")
    out = tmp_path / "out.json"
    result, path = run_pipeline(mock_config, src, out, reference_summary_path=src)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert path == out and saved["summary"] == result["summary"] != ""
    for key in ("input", "keyword_source", "ablation_mode", "keywords", "metrics", "config", "timestamp"):
        assert key in saved
    m = saved["metrics"]
    assert set(m["keyword_coverage"]) == {"coverage", "found", "missing"}
    assert {"precision", "recall", "summary_keywords"} <= set(m["summary_alignment"])
    assert {"rouge1_f", "rougeL_f"} <= set(m["rouge"])
    assert isinstance(m["compression_ratio"], float)  # always computed now


def test_legacy_pipeline_empty_input_is_a_clear_error(tmp_path, mock_config):
    src = tmp_path / "empty.txt"
    src.write_text("  \n ", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        run_pipeline(mock_config, src, tmp_path / "o.json")


def test_legacy_extractor_never_raises_on_degenerate_text():
    ex = YakeKeywordExtractor(YakeConfig())
    assert ex.extract("") == [] and ex.extract("   ") == []


def test_keyword_coverage_is_word_boundary_and_stem_aware():
    assert keyword_coverage("A long paragraph.", ["graph"])["coverage"] == 0.0  # old bug
    assert keyword_coverage("Neural networks learn.", ["neural network"])["coverage"] == 1.0
    assert keyword_coverage("", [])["coverage"] == 0.0
    pr = keyword_precision_recall(["Graph Databases"], ["graph database"])
    assert pr["precision"] == 1.0 and pr["recall"] == 1.0
    assert compute_rouge("alpha", "beta")["rouge1_f"] == 0.0


def test_fit_context_keeps_relevant_tail_instead_of_cutting_it():
    filler = " ".join(f"Filler sentence number {i} about nothing." for i in range(60))
    tail = " Quantum annealing outperforms simulated annealing on quantum annealing benchmarks."
    text = filler + tail
    kws = [("quantum annealing", 0.001), ("simulated annealing", 0.01)]
    assert "Quantum annealing" not in fit_context(text, 400, kws, strategy="truncate")
    kept = fit_context(text, 400, kws)
    assert "Quantum annealing" in kept and len(kept) <= 400


def test_split_sentences_delegates_to_robust_splitter():
    assert split_sentences("Dr. Lee agrees. It works.") == ["Dr. Lee agrees.", "It works."]


def test_unknown_legacy_backend_raises_llm_error():
    with pytest.raises(LLMError):
        create_llm_client(LLMConfig(backend="nope"))


def test_legacy_ollama_unreachable_raises_llm_error_with_advice():
    client = create_llm_client(LLMConfig(backend="ollama", ollama_host="http://127.0.0.1:1"))
    client._client.max_retries = 0
    with pytest.raises(LLMError, match="ollama serve"):
        client.generate("hi")


# ---------------------------------------------------------------- FastAPI -------------
@pytest.fixture()
def client(mock_config):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from backend.main import app
    from backend.routers.deps import set_config_path

    set_config_path(str(mock_config))
    yield TestClient(app)
    set_config_path("")


def test_api_summarize_with_mock_backend(client):
    r = client.post("/api/summarize", json={"text": DOC})
    assert r.status_code == 200
    body = r.json()
    assert body["backend_available"] is True and body["summary"]
    assert body["keywords"]["source"].startswith("YAKE")
    assert isinstance(body["metrics"]["compression_ratio"], float)
    assert "Keywords:" in body["prompt"]


def test_api_external_keywords(client):
    r = client.post("/api/summarize", json={"text": DOC, "external_keywords": ["fraud detection"]})
    body = r.json()
    assert body["keywords"]["source"].startswith("external")
    assert [k["keyword"] for k in body["keywords"]["keywords"]] == ["fraud detection"]


def test_api_empty_or_blank_text(client):
    assert client.post("/api/summarize", json={"text": ""}).status_code == 422
    assert client.post("/api/summarize", json={"text": "   \n  "}).status_code == 422


def test_api_backend_failure_degrades_gracefully(client, mock_config):
    cfg = yaml.safe_load(mock_config.read_text())
    cfg["llm"] = {"backend": "ollama", "ollama_model": "m", "ollama_host": "http://127.0.0.1:1"}
    mock_config.write_text(yaml.safe_dump(cfg))
    from backend.routers.deps import set_config_path

    set_config_path(str(mock_config))
    r = client.post("/api/summarize", json={"text": DOC})
    assert r.status_code == 200
    body = r.json()
    assert body["backend_available"] is False and body["metrics"] is None and body["prompt"]


def test_api_stopword_only_text_falls_back_to_plain_prompt(client):
    r = client.post("/api/summarize", json={"text": "It is what it is. It was so."})
    assert r.status_code == 200
    assert "Keywords:" not in r.json()["prompt"]
