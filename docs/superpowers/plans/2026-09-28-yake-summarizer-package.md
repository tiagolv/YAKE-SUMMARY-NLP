# YAKE Summarizer (`yake-sum`) Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the experimental CLNP repository into `yake-sum`, a production-ready, pip-installable Python package offering pure extractive, LLM-guided abstractive, and hybrid long-document summarization, backed by an empirical ROUGE benchmark study.

**Architecture:** A modular library under `src/yake_sum/` with decoupled engines for zero-LLM extractive sentence scoring, prompt-conditioned abstractive generation, and passage-filtered hybrid summarization, communicating via protocol-based LLM backends (Ollama, LlamaCpp, OpenAI, Mock) and unified under a high-level `Summarizer` class and `yake-sum` CLI.

**Tech Stack:** Python 3.10+, YAKE!, Rouge-Score, Pydantic, PyYAML, Requests, Pytest, Pyproject.toml (Hatchling/Flit/Setuptools).

**Spec:** [`docs/superpowers/specs/2026-09-28-yake-summarizer-package-design.md`](file:///c:/Users/Tiago/Documents/GitHub/CLNP/docs/superpowers/specs/2026-09-28-yake-summarizer-package-design.md)

## Global Constraints

- Python versions supported: `>=3.10`
- Zero-LLM extractive mode must have zero GPU or heavy neural dependencies (runs on CPU in <15ms)
- YAKE keyword score weighting must use inversion: $W_{kw} = 1 / (S_{kw} + 10^{-6})$
- All unit and integration tests must run deterministically in CI without requiring a running Ollama server or GPU weights
- Existing `backend/` and `app.py` interfaces must maintain backward compatibility

## Review Focus

1. Input text is empty, whitespace-only, or has fewer sentences than requested: expect graceful fallback returning empty or all available sentences without exceptions.
2. YAKE returns zero keywords (e.g. extremely short or stopword-only text): extractive and abstractive engines must fall back safely to unconditioned summarization.
3. Reference summary is provided for evaluation with zero overlapping n-grams: ROUGE computation must return 0.0 without division-by-zero errors.
4. Input text exceeds context budget in hybrid mode: passage extractor must select top-scored non-contiguous paragraphs and preserve their chronological relative order.
5. Ollama backend host is unreachable: client must raise a specific `BackendConnectionError` with troubleshooting advice rather than generic unhandled exceptions.

---

### Task 1: ROUGE Evaluation Module & Benchmark Dataset

**Files:**
- Create: `data/eval/benchmark_with_summaries.json`
- Create: `src/yake_sum/evaluation/rouge.py`
- Test: `tests/test_rouge.py`

**Interfaces:**
- Produces: `compute_rouge_metrics(summary: str, reference: str) -> dict[str, float]` returning `rouge1_p`, `rouge1_r`, `rouge1_f`, `rouge2_p`, `rouge2_r`, `rouge2_f`, `rougeL_p`, `rougeL_r`, `rougeL_f`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rouge.py
import pytest
from yake_sum.evaluation.rouge import compute_rouge_metrics

def test_compute_rouge_metrics_identical_text():
    text = "Machine learning enables computers to learn from data."
    metrics = compute_rouge_metrics(summary=text, reference=text)
    assert metrics["rouge1_f"] == pytest.approx(1.0)
    assert metrics["rouge2_f"] == pytest.approx(1.0)
    assert metrics["rougeL_f"] == pytest.approx(1.0)

def test_compute_rouge_metrics_empty_inputs():
    metrics = compute_rouge_metrics(summary="", reference="Some text")
    assert metrics["rouge1_f"] == 0.0
    assert metrics["rouge2_f"] == 0.0
    assert metrics["rougeL_f"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rouge.py`  
Expected: FAIL with `ModuleNotFoundError: No module named 'yake_sum'`

- [ ] **Step 3: Implement `src/yake_sum/evaluation/rouge.py` and populate `data/eval/benchmark_with_summaries.json`**

Implement `compute_rouge_metrics` using `rouge_score.rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)` extracting precision, recall, and fmeasure. Create `benchmark_with_summaries.json` with 15 curated scientific abstracts with reference summaries.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rouge.py`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_rouge.py src/yake_sum/evaluation/rouge.py data/eval/benchmark_with_summaries.json; git commit -m "feat(evaluation): add comprehensive ROUGE metric calculation and benchmark dataset"
```

---

### Task 2: Pure Extractive Summarization Engine (Zero-LLM)

**Files:**
- Create: `src/yake_sum/extractive/scorer.py`
- Create: `src/yake_sum/extractive/summarizer.py`
- Test: `tests/test_extractive.py`

**Interfaces:**
- Produces: `YakeSentenceScorer(keywords: list[tuple[str, float]])` with `score_sentences(sentences: list[str]) -> list[float]`
- Produces: `ExtractiveSummarizer(num_sentences: int = 3, max_ngram_size: int = 3, top_k: int = 15)` with `summarize(text: str) -> ExtractiveResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extractive.py
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
    res_short = summarizer.summarize("Single sentence document.")
    assert res_short.summary == "Single sentence document."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extractive.py`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement sentence splitting, YAKE weight inversion, sentence scoring, and reordering**

In `scorer.py`:
- Invert YAKE score: $W = 1.0 / (\text{score} + 1e-6)$.
- Compute sentence score with length normalization $\text{len}^\alpha$ ($\alpha=0.8$) and position factor $1.0 + 1.0 / \sqrt{idx + 1}$.
In `summarizer.py`:
- Split text using regex sentence boundary.
- Score sentences, select top $K$, sort chronologically by original index, join with spaces.
- Compute compression ratio: $1 - (\text{words}_{\text{summary}} / \text{words}_{\text{source}})$.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_extractive.py`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_extractive.py src/yake_sum/extractive/; git commit -m "feat(extractive): implement YAKE-weighted sentence scoring and extractive summarizer"
```

---

### Task 3: Pluggable LLM Backends & Deterministic MockClient

**Files:**
- Create: `src/yake_sum/backends/base.py`
- Create: `src/yake_sum/backends/mock.py`
- Create: `src/yake_sum/backends/ollama.py`
- Create: `src/yake_sum/backends/llama_cpp.py`
- Create: `src/yake_sum/backends/openai_compatible.py`
- Test: `tests/test_backends.py`

**Interfaces:**
- Produces: `BaseLLMClient` protocol with `generate(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str` and `is_available() -> tuple[bool, str]`
- Produces: `get_llm_client(backend: str, **kwargs) -> BaseLLMClient`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backends.py
import pytest
from yake_sum.backends import get_llm_client, MockLLMClient

def test_mock_backend_generation():
    client = get_llm_client(backend="mock")
    assert isinstance(client, MockLLMClient)
    available, msg = client.is_available()
    assert available is True
    output = client.generate("Summarize this: Keywords: [ai, data]")
    assert "Mock summary" in output

def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unsupported backend"):
        get_llm_client(backend="nonexistent_backend")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backends.py`  
Expected: FAIL

- [ ] **Step 3: Implement the backends module**

In `base.py`, define `BaseLLMClient` protocol and `BackendError`. Implement `MockLLMClient` (for CI/tests), `OllamaClient` (with timeout and WSL host support), `LlamaCppClient` (guarded import of `llama_cpp`), and `OpenAICompatibleClient` (requests to `/v1/chat/completions` or `/v1/completions`). Implement factory function `get_llm_client`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backends.py`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_backends.py src/yake_sum/backends/; git commit -m "feat(backends): implement pluggable LLM backends with protocol and MockClient"
```

---

### Task 4: Guided Abstractive & Hybrid Long-Document Summarizer

**Files:**
- Create: `src/yake_sum/abstractive/prompt.py`
- Create: `src/yake_sum/abstractive/summarizer.py`
- Create: `src/yake_sum/hybrid/summarizer.py`
- Test: `tests/test_abstractive_hybrid.py`

**Interfaces:**
- Produces: `AbstractiveSummarizer(llm_client: BaseLLMClient, yake_config: YakeConfig)` with `summarize(text: str) -> AbstractiveResult`
- Produces: `HybridSummarizer(llm_client: BaseLLMClient, max_context_chars: int = 3000)` with `summarize(text: str) -> HybridResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_abstractive_hybrid.py
from yake_sum.backends.mock import MockLLMClient
from yake_sum.abstractive.summarizer import AbstractiveSummarizer
from yake_sum.hybrid.summarizer import HybridSummarizer

def test_abstractive_with_mock():
    client = MockLLMClient()
    summarizer = AbstractiveSummarizer(client=client)
    res = summarizer.summarize("This is a document about machine learning and deep learning.")
    assert res.summary != ""
    assert len(res.keywords) > 0
    assert res.prompt_used != ""

def test_hybrid_long_document():
    client = MockLLMClient()
    long_text = "\n\n".join([f"Paragraph {i}: Content about artificial intelligence and neural networks in domain {i}." for i in range(20)])
    summarizer = HybridSummarizer(client=client, max_context_chars=500)
    res = summarizer.summarize(long_text)
    assert res.summary != ""
    assert len(res.retained_passages) > 0
    assert res.context_char_count <= 600
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_abstractive_hybrid.py`  
Expected: FAIL

- [ ] **Step 3: Implement Abstractive and Hybrid Summarizers**

In `abstractive/`:
- Format prompt with system instructions, extracted YAKE keywords, and text. Call `client.generate(prompt)`.
In `hybrid/`:
- Segment long text into paragraphs/passages. Score each passage against global YAKE keywords.
- Select top-ranking passages until `max_context_chars` budget is reached, preserving original order.
- Feed selected passages + global keywords to abstractive generator.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_abstractive_hybrid.py`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_abstractive_hybrid.py src/yake_sum/abstractive/ src/yake_sum/hybrid/; git commit -m "feat(summarizers): implement abstractive and hybrid long-doc summarizers"
```

---

### Task 5: High-Level Unified API (`Summarizer`), CLI (`yake-sum`), and `pyproject.toml`

**Files:**
- Create: `pyproject.toml`
- Create: `src/yake_sum/__init__.py`
- Create: `src/yake_sum/summarizer.py`
- Create: `src/yake_sum/cli.py`
- Test: `tests/test_api_and_cli.py`

**Interfaces:**
- Produces: `from yake_sum import Summarizer` with `summarize(text: str, reference_summary: str | None = None) -> SummaryResult`
- Produces: `yake-sum` console command.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_and_cli.py
from yake_sum import Summarizer

def test_unified_summarizer_extractive():
    sum_ext = Summarizer(mode="extractive", num_sentences=2)
    res = sum_ext.summarize("First sentence about NLP. Second sentence about AI. Third sentence about data.")
    assert res.mode == "extractive"
    assert len(res.text) > 0

def test_unified_summarizer_abstractive_mock():
    sum_abs = Summarizer(mode="abstractive", backend="mock")
    res = sum_abs.summarize("Doc about transformers.", reference_summary="Doc about transformers.")
    assert res.mode == "abstractive"
    assert "rouge" in res.metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_and_cli.py`  
Expected: FAIL

- [ ] **Step 3: Implement `Summarizer` class, CLI, and `pyproject.toml`**

- `Summarizer` delegates to Extractive, Abstractive, or Hybrid based on `mode`.
- If `reference_summary` is provided, automatically compute ROUGE metrics in `res.metrics["rouge"]`.
- In `cli.py`: parse `--mode`, `--backend`, `--model`, `--sentences`, `--reference`, `--output`, `--json`.
- In `pyproject.toml`: configure packaging metadata, dependencies, and `[project.scripts] yake-sum = "yake_sum.cli:main"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_and_cli.py`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/yake_sum/ tests/test_api_and_cli.py; git commit -m "feat(package): create pyproject.toml, unified Summarizer facade, and yake-sum CLI"
```

---

### Task 6: ROUGE Benchmark Study Execution & Report Generation

**Files:**
- Create: `benchmarks/run_rouge_study.py`
- Create: `outputs/rouge_benchmark_report.md`
- Test: `tests/test_benchmark_runner.py`

**Interfaces:**
- Produces: `run_rouge_benchmark(benchmark_file: Path, backend: str = "mock") -> BenchmarkSummary`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark_runner.py
from pathlib import Path
from benchmarks.run_rouge_study import run_rouge_benchmark

def test_benchmark_runner_with_mock():
    benchmark_path = Path("data/eval/benchmark_with_summaries.json")
    results = run_rouge_benchmark(benchmark_file=benchmark_path, backend="mock", max_samples=3)
    assert len(results["methods"]) == 4  # lead_3, yake_extractive, unconditioned, yake_guided
    assert "lead_3" in results["methods"]
    assert "yake_extractive" in results["methods"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_runner.py`  
Expected: FAIL

- [ ] **Step 3: Implement `benchmarks/run_rouge_study.py` and generate Markdown report**

Compute average ROUGE-1, ROUGE-2, and ROUGE-L across all samples in `benchmark_with_summaries.json` for Lead-3, YAKE Extractive, LLM Unconditioned, and YAKE Guided. Output clean Markdown table to `outputs/rouge_benchmark_report.md`.

- [ ] **Step 4: Run benchmark and tests**

Run: `pytest tests/test_benchmark_runner.py`  
Run: `python -m benchmarks.run_rouge_study --output outputs/rouge_benchmark_report.md`  
Expected: PASS and report generated.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/ tests/test_benchmark_runner.py outputs/rouge_benchmark_report.md; git commit -m "feat(benchmarks): add automated ROUGE benchmark study runner and results report"
```

---

### Task 7: Backward Compatibility & FastAPI Integration

**Files:**
- Modify: `backend/routers/summarize.py`
- Modify: `src/pipeline.py`
- Test: `tests/test_legacy_compat.py`

**Interfaces:**
- Preserves existing FastAPI endpoint signatures while leveraging the new `yake_sum` package internally.

- [ ] **Step 1: Write test for legacy compatibility**

```python
# tests/test_legacy_compat.py
from src.pipeline import run_pipeline
from pathlib import Path

def test_legacy_pipeline_runs_with_mock(tmp_path):
    # Verify legacy CLI runner still produces valid run output
    config_path = Path("config.yaml")
    sample_path = Path("data/samples/paper_yake.txt")
    out_path = tmp_path / "out.json"
    result, path = run_pipeline(config_path=config_path, input_path=sample_path, output_path=out_path)
    assert out_path.exists()
    assert "summary" in result
```

- [ ] **Step 2: Verify and adapt backend routers and pipeline**

Refactor `backend/routers/summarize.py` and `src/pipeline.py` to use `yake_sum` components internally, eliminating duplicate logic and ensuring `compression_ratio` is always computed.

- [ ] **Step 3: Run full test suite**

Run: `pytest`  
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add backend/ src/pipeline.py tests/test_legacy_compat.py; git commit -m "refactor(compat): update FastAPI backend and legacy pipeline to consume yake_sum package"
```

---

## Completion Plan (2026-09-28)

The core package is already implemented and the current test baseline is 19 passing tests. The remaining work below closes the gaps found during the implementation review. Do not mark a phase complete until its focused tests and the full suite pass.

### Phase 1: Make the benchmark scientifically usable

**Goal:** complete the ROUGE study without presenting deterministic mock output as model-quality evidence.

- [ ] Expand `data/eval/benchmark_with_summaries.json` from 10 to at least 15 documents.
- [ ] Add a `reference_source` field to every document and document whether each reference is an author abstract, curated reference, or project-created summary.
- [ ] Validate dataset schema, unique IDs, non-empty document/reference fields, and the minimum document count in `tests/test_benchmark_runner.py`.
- [ ] Keep `oracle_llm_guided` in the comparison matrix and test all five methods.
- [ ] Keep the mock report explicitly labelled as a deterministic smoke test.
- [ ] Add a local-model run path to the benchmark documentation, without requiring a model in CI.

**Acceptance:** the dataset has at least 15 traceable entries; the runner produces Markdown and JSON; CI validates the structure without needing Ollama or GPU weights.

### Phase 2: Finish the shared package contracts

**Goal:** remove duplicated configuration and evaluation concepts from the new package.

- [ ] Add `src/yake_sum/config.py` with typed configuration models for YAKE, LLM, prompt, extractive, and hybrid settings.
- [ ] Make `Summarizer` and the lower-level engines accept these settings while preserving current constructor compatibility.
- [ ] Add `src/yake_sum/evaluation/alignment.py` for keyword coverage, precision, recall, and summary alignment.
- [ ] Add `src/yake_sum/evaluation/judge.py` with a protocol-based LLM-as-a-judge interface and deterministic mock support.
- [ ] Export the new public types from `src/yake_sum/evaluation/__init__.py` and `src/yake_sum/__init__.py`.
- [ ] Add unit tests for empty keywords, no-overlap metrics, duplicate keywords, and malformed judge responses.

**Acceptance:** package evaluation APIs are importable, deterministic tests cover zero/empty cases, and no existing public API test changes its expected behavior.

### Phase 3: Refactor legacy execution paths

**Goal:** preserve the existing FastAPI and legacy CLI contracts while consuming `yake_sum` internally.

- [ ] Add `tests/test_legacy_compat.py` covering `src.pipeline.run_pipeline` with a deterministic mock configuration.
- [ ] Refactor `src/pipeline.py` to delegate extraction, summarization, and ROUGE calculation to package components where the legacy response format permits it.
- [ ] Refactor `backend/routers/summarize.py` to use the package facade while preserving request and response schemas and ablation modes.
- [ ] Keep legacy YAML loading and output JSON field names backward compatible.
- [ ] Add API tests for empty input, external keywords, backend failure, and compression ratio.
- [ ] Add a regression test that starts the FastAPI app and exercises `POST /api/summarize` with the mock backend.

**Acceptance:** legacy pipeline output remains readable by existing consumers, FastAPI endpoint signatures remain unchanged, and all legacy plus package tests pass.

### Phase 4: Harden backends and CLI behavior

**Goal:** make operational failures explicit and retry behavior predictable.

- [ ] Add bounded retry with backoff to `OllamaClient`, retrying only connection and transient HTTP failures.
- [ ] Preserve `BackendConnectionError` with host/model/troubleshooting context after retries are exhausted.
- [ ] Add tests using a fake transport for success-after-retry, permanent connection failure, timeout, and non-retryable HTTP errors.
- [ ] Validate `max_context_chars`, sentence counts, and other numeric settings at construction time with clear `ValueError` messages.
- [ ] Make the CLI return a non-zero exit code and an actionable error when `--reference` points to a missing file.
- [ ] Add CLI tests for output files, JSON output, invalid reference paths, and backend selection.

**Acceptance:** transient backend failures are retried deterministically, permanent failures expose a specific error, invalid CLI input fails visibly, and all new behavior is covered without network access.

### Phase 5: Package, documentation, and reproducibility cleanup

**Goal:** make the repository installable and understandable from a clean checkout.

- [ ] Move runtime dependencies required by `yake_sum.config` and evaluation into the main dependency set; keep server and llama.cpp dependencies optional.
- [ ] Verify editable installation with `pip install -e .` and the `yake-sum --help` entry point.
- [ ] Update `README.md` with extractive, abstractive, hybrid, benchmark, and legacy API usage examples.
- [ ] Document that the mock backend is for tests only and that benchmark quality claims require a configured local model.
- [ ] Document the dataset provenance and how to regenerate `outputs/rouge_benchmark_report.md` and its JSON companion.
- [ ] Add a clean-environment validation command to the project documentation.
- [ ] Update this plan's earlier task checkboxes to reflect verified implementation status.

**Acceptance:** a clean virtual environment can install the package, run the deterministic test suite, invoke the CLI, and reproduce the smoke-test report.

### Final verification order

1. `.venv\\Scripts\\python.exe -m pytest -q tests/test_rouge.py tests/test_extractive.py tests/test_backends.py tests/test_abstractive_hybrid.py tests/test_api_and_cli.py tests/test_benchmark_runner.py`
2. `.venv\\Scripts\\python.exe -m pytest -q`
3. `.venv\\Scripts\\python.exe -m benchmarks.run_rouge_study --backend mock --output outputs/rouge_benchmark_report.md`
4. `.venv\\Scripts\\python.exe -m pip install -e .`
5. `.venv\\Scripts\\yake-sum.exe --help`

The project is complete when all five phases meet their acceptance criteria, the full suite is green, the benchmark report is reproducible, and no claim in the documentation exceeds what the selected backend and dataset provenance support.
