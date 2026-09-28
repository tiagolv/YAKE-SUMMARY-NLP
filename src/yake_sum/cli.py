"""Command line interface for YAKE Summarizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .summarizer import Summarizer


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="yake-sum",
        description="Fast, keyword-guided text summarization with YAKE! and LLMs.",
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input text file.",
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
        "-r", "--reference",
        help="Path to reference summary file for automated ROUGE evaluation.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: prints to stdout).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Format output as JSON including keywords and metrics.",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    input_path = Path(parsed.input)
    if not input_path.exists():
        print(f"Error: Input file '{parsed.input}' not found.", file=sys.stderr)
        return 1

    text = input_path.read_text(encoding="utf-8")

    ref_text = None
    if parsed.reference:
        ref_path = Path(parsed.reference)
        if ref_path.exists():
            ref_text = ref_path.read_text(encoding="utf-8")

    summarizer = Summarizer(
        mode=parsed.mode,
        backend=parsed.backend,
        model=parsed.model,
        num_sentences=parsed.sentences,
        top_k=parsed.top_k,
    )

    result = summarizer.summarize(text=text, reference_summary=ref_text)

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
