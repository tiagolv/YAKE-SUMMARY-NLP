# Technical Specification: YAKE Summarizer (`yake-sum`)

**Date**: 2026-09-28  
**Status**: Draft for Review  
**Target Package Name**: `yake-sum` (or `clnp-summarizer`)

---

## 1. Context & Motivation

The current repository ([tiagolv/CLNP](file:///c:/Users/Tiago/Documents/GitHub/CLNP)) contains an experimental pipeline that combines the **YAKE!** unsupervised keyword extractor with local LLMs (Ollama / LLaMA.cpp) to produce keyword-conditioned summaries.

Experimental ablation and batch tests ([outputs/experiment_report.md](file:///c:/Users/Tiago/Documents/GitHub/CLNP/outputs/experiment_report.md)) demonstrated that:
1. Keyword conditioning increases keyword coverage in summaries by **+75%** relative to unconditioned LLM generation.
2. An LLM judge consistently awards the highest overall rating (4.50 vs 4.18) to the keyword-guided pipeline.
3. Optimal parameter settings are $N$-gram sizes of $2-3$ and $top\_k$ between $10-15$.

### Gaps Being Solved:
1. **No ROUGE validation against gold reference summaries**: Metric coverage was previously restricted to keyword recall/precision and LLM-as-a-judge scores. ROUGE-1, ROUGE-2, and ROUGE-L are required to establish scientific validity.
2. **Rigid 3000-character blind truncation**: Long documents lose all text beyond the first ~500 words.
3. **No standalone, zero-LLM extractive mode**: Users without a local LLM or GPU cannot use the repository.
4. **Lack of Python packaging**: Currently structured as ad-hoc scripts with hardcoded relative imports and file-based YAML configuration.

---

## 2. Architecture & Package Structure

The repository will be reorganized into a clean modern Python package structure:

```text
CLNP/
├── pyproject.toml                         # Packaging, dependencies & console scripts
├── config.yaml                            # Default configuration template
├── src/
│   └── yake_sum/                          # Main library package
│       ├── __init__.py                    # Public API exports
│       ├── summarizer.py                  # High-level entrypoint: class Summarizer
│       ├── models.py                      # Data models (SummaryResult, Keyword, Metrics)
│       ├── config.py                      # Pydantic-based configuration schemas
│       ├── extractive/                    # Zero-LLM Extractive Engine
│       │   ├── __init__.py
│       │   ├── scorer.py                  # YAKE-weighted sentence scoring
│       │   └── summarizer.py              # ExtractiveSummarizer implementation
│       ├── abstractive/                   # LLM Guided Abstractive Engine
│       │   ├── __init__.py
│       │   ├── prompt.py                  # Prompt templates & conditioning
│       │   └── summarizer.py              # AbstractiveSummarizer implementation
│       ├── hybrid/                        # Long-Document Engine
│       │   ├── __init__.py
│       │   └── summarizer.py              # HybridSummarizer (Extract -> Abstract)
│       ├── backends/                      # Pluggable LLM interfaces
│       │   ├── __init__.py
│       │   ├── base.py                    # BaseLLMClient protocol
│       │   ├── ollama.py                  # OllamaClient
│       │   ├── llama_cpp.py               # LlamaCppClient
│       │   ├── openai_compatible.py       # OpenAICompatibleClient (vLLM, LMStudio, etc.)
│       │   └── mock.py                    # MockClient for deterministic CI tests
│       ├── evaluation/                    # Evaluation & Metrics
│       │   ├── __init__.py
│       │   ├── rouge.py                   # ROUGE-1, 2, L (P, R, F1)
│       │   ├── alignment.py               # Keyword coverage & alignment
│       │   └── judge.py                   # LLM-as-a-Judge protocol
│       └── cli.py                         # Console CLI (`yake-sum`)
├── tests/                                 # Pytest test suite
│   ├── test_extractive.py
│   ├── test_rouge.py
│   ├── test_backends.py
│   └── test_summarizer.py
├── data/                                  # Datasets & gold standards
│   ├── eval/
│   │   ├── inspec_gold.json
│   │   └── benchmark_with_summaries.json  # Documents with human reference summaries
│   └── samples/                           # Sample txt files
└── backend/                               # Existing FastAPI service (refactored to consume yake_sum)
```

---

## 3. Algorithm & Component Details

### 3.1. Extractive Engine (`yake_sum.extractive`)

The extractive engine runs on CPU in milliseconds without downloading LLM weights.

#### Keyword Weight Inversion:
In YAKE!, smaller scores signify higher significance:
$$\text{Score}_{\text{YAKE}} \in [0, \infty), \quad 0 = \text{most important}$$

Weight assignment for keyword $kw$:
$$\text{Weight}(kw) = \frac{1}{\text{Score}_{\text{YAKE}}(kw) + 10^{-6}}$$

#### Sentence Scoring Formula:
For a candidate sentence $S_i$ at index $i \in \{0, \dots, M-1\}$:
$$\text{RawScore}(S_i) = \sum_{kw \in S_i} \text{Weight}(kw)$$
$$\text{NormScore}(S_i) = \frac{\text{RawScore}(S_i)}{(\text{WordCount}(S_i) + 1)^\alpha}, \quad \text{with } \alpha = 0.8$$
$$\text{PositionBonus}(i) = 1.0 + \frac{1.0}{\sqrt{i + 1}}$$
$$\text{FinalScore}(S_i) = \text{NormScore}(S_i) \times \text{PositionBonus}(i)$$

Sentences are ranked by $\text{FinalScore}(S_i)$. The top $K$ sentences (or sentences up to target compression ratio) are selected and **reordered chronologically** by their appearance in the original document.

---

### 3.2. ROUGE Benchmark Specification (`yake_sum.evaluation.rouge`)

To provide empirical proof of the package's utility:
- **ROUGE-1**: Evaluates unigram overlap (concept retention).
- **ROUGE-2**: Evaluates bigram overlap (grammatical coherence and phrase consistency).
- **ROUGE-L**: Evaluates Longest Common Subsequence (discourse flow).

#### Benchmark Dataset:
`data/eval/benchmark_with_summaries.json` will contain at least 15 curated scientific and technical documents paired with authentic human author summaries (*abstracts/gold summaries*).

#### Experimental Comparison Matrix:
A dedicated runner script `benchmarks/run_rouge_study.py` will execute across all dataset entries:
1. `lead_3`: First 3 sentences (standard NLP extractive baseline).
2. `yake_extractive`: Pure YAKE extractive summarization (our lightweight method).
3. `llm_unconditioned`: Prompt with text only (`no_keywords`).
4. `yake_llm_guided`: Full pipeline with YAKE keywords (`full`).
5. `oracle_llm_guided`: LLM conditioned on gold human keywords.

Outputs will produce an automated summary table comparing Mean Precision, Recall, and F1 across ROUGE-1, ROUGE-2, and ROUGE-L.

---

### 3.3. Hybrid Long-Document Summarization (`yake_sum.hybrid`)

To eliminate the 3000-character blind cut:
1. Input document $D$ of arbitrary length is analyzed by YAKE! to extract document-level global keywords.
2. The document is segmented into semantic passages/paragraphs $P_1, P_2, \dots, P_n$.
3. Each passage $P_j$ is scored using the YAKE keyword density score.
4. Top passages fitting within the LLM's context budget (e.g. 2000-4000 characters) are selected in original order.
5. The LLM receives the salient passages and the global keywords to generate the final abstractive summary.

---

### 3.4. Pluggable LLM Backends (`yake_sum.backends`)

All backends conform to a common `BaseLLMClient` protocol:

```python
class BaseLLMClient(Protocol):
    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        ...

    def is_available(self) -> tuple[bool, str]:
        ...
```

Supported implementations:
- `OllamaClient`: HTTP calls to local Ollama server with automatic retry and host resolution (supports WSL `OLLAMA_HOST`).
- `LlamaCppClient`: Direct in-process GGUF model execution via `llama-cpp-python`.
- `OpenAICompatibleClient`: Works with any standard OpenAI-compatible API (vLLM, LM Studio, Ollama OpenAI endpoint, LocalAI).
- `MockClient`: Deterministic rule-based client returning predictable text for fast, reproducible unit tests without local models.

---

### 3.5. High-Level Python API Design

```python
from yake_sum import Summarizer

# Extractive Mode (CPU only, ultra-fast)
ext_summarizer = Summarizer(mode="extractive", num_sentences=3)
result = ext_summarizer.summarize(text)
print(result.text)
print(result.keywords)

# Guided Abstractive Mode
abs_summarizer = Summarizer(
    mode="abstractive",
    backend="ollama",
    model="mistral",
    top_k=12
)
result = abs_summarizer.summarize(text, reference_summary=gold_summary)
print(result.text)
print(result.metrics.rouge)  # {"rouge1_f": 0.45, "rouge2_f": 0.22, "rougeL_f": 0.41}

# Hybrid Mode for long documents
hyb_summarizer = Summarizer(mode="hybrid", backend="ollama", model="mistral")
result = hyb_summarizer.summarize(long_text)
```

---

## 4. Testing & Reliability Strategy

- **Deterministic testing**: Test suite in `tests/` runs in CI without requiring Ollama or GPU weights, using `MockClient`.
- **Edge cases covered**: Empty text, single-sentence text, repetitive text, text shorter than target sentence count, non-ASCII/unicode characters.
- **Metric validation**: Unit tests validating ROUGE calculation against known sample strings.

---

## 5. Review & Feedback Request

Please review this specification. Once approved, we will proceed to create the step-by-step implementation plan.
