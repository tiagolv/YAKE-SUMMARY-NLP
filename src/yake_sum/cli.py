"""Command line interface for YAKE Summarizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backends import get_llm_client
from .backends.base import BackendConnectionError, BackendError, BackendTimeoutError
from .summarizer import Summarizer


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="yake-sum",
        description="Fast, keyword-guided text summarization with YAKE! and LLMs.",
    )
    parser.add_argument(
        "-i", "--input",
        default=None,
        help="Path to input text file, or '-' to read from stdin.",
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["extractive", "abstractive", "hybrid"],
        default="extractive",
        help="Summarization mode: 'extractive' (zero-LLM), 'abstractive' (LLM-guided), or 'hybrid' (long doc).",
    )
    parser.add_argument(
        "-b", "--backend",
        choices=["mock", "ollama", "llama_cpp", "openai_compatible"],
        default="ollama",
        help="LLM backend for abstractive/hybrid modes.",
    )
    parser.add_argument(
        "--model",
        default="mistral",
        help="Model name (for Ollama/OpenAI) or model path (for llama_cpp).",
    )
    parser.add_argument(
        "-s", "--sentences",
        type=int,
        default=3,
        help="Number of sentences for extractive mode (default: 3).",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=12,
        help="Number of top keywords to extract with YAKE (default: 12).",
    )
    parser.add_argument(
        "--host",
        help="Ollama host or OpenAI-compatible base URL (e.g. http://localhost:11434).",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=3000,
        help="Context budget for hybrid mode (default: 3000).",
    )
    parser.add_argument(
        "-r", "--reference",
        help="Path to reference summary file for automated ROUGE evaluation.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: prints to stdout).",
    )
    parser.add_argument(
        "--check-backend",
        action="store_true",
        help="Only check that the selected backend/model is reachable, then exit (0 = ready).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Format output as JSON including keywords and metrics.",
    )
    return parser.parse_args(args)


def _fail(message: str, code: int) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return code


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    if parsed.check_backend:
        extra_check: dict[str, str] = {}
        if parsed.host:
            extra_check["base_url" if parsed.backend in ("openai_compatible", "openai") else "host"] = parsed.host
        try:
            ok, msg = get_llm_client(parsed.backend, model=parsed.model, **extra_check).is_available()
        except (ValueError, BackendError) as exc:
            ok, msg = False, str(exc)
        print(("OK: " if ok else "NOT READY: ") + msg, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 3
    if not parsed.input:
        return _fail("the following arguments are required: -i/--input", 2)

    if parsed.input == "-":
        text = sys.stdin.read()
    else:
        input_path = Path(parsed.input)
        if not input_path.is_file():
            return _fail(f"input file '{parsed.input}' not found.", 1)
        text = input_path.read_text(encoding="utf-8")

    ref_text = None
    if parsed.reference:
        ref_path = Path(parsed.reference)
        if not ref_path.is_file():
            return _fail(
                f"reference file '{parsed.reference}' not found. "
                "Check the path or drop --reference to skip ROUGE evaluation.",
                2,
            )
        ref_text = ref_path.read_text(encoding="utf-8")

    extra: dict[str, str] = {}
    if parsed.host:
        extra["base_url" if parsed.backend in ("openai_compatible", "openai") else "host"] = parsed.host

    try:
        summarizer = Summarizer(
            mode=parsed.mode,
            backend=parsed.backend,
            model=parsed.model,
            num_sentences=parsed.sentences,
            top_k=parsed.top_k,
            max_context_chars=parsed.max_context_chars,
            **extra,
        )
        result = summarizer.summarize(text=text, reference_summary=ref_text)
    except ValueError as exc:
        return _fail(str(exc), 2)
    except (BackendConnectionError, BackendTimeoutError) as exc:
        return _fail(f"LLM backend unreachable. {exc}", 3)
    except BackendError as exc:
        return _fail(f"LLM backend failed. {exc}", 3)

    for warning in result.metrics.get("warnings", []):
        print(f"Warning: {warning}", file=sys.stderr)

    if parsed.json:
        payload = {
            "mode": result.mode,
            "summary": result.text,
            "keywords": [{"keyword": k.keyword, "score": k.score} for k in result.keywords],
            "metrics": result.metrics,
        }
        output_str = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        output_str = result.text

    if parsed.output:
        out_path = Path(parsed.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_str, encoding="utf-8")
        print(f"Summary written to {parsed.output}")
    else:
        print(output_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
