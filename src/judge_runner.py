"""Judge runner for ablation results.

Evaluates each successful ablation run with the judge module (backend logic).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.judge import Judge   # agora usa a versão robusta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-results", default="outputs/ablation_results.json", help="Input ablation JSON")
    parser.add_argument("--samples-dir", default="data/samples", help="Directory with source texts")
    parser.add_argument("--output", default="outputs/judge_results.json", help="Output JSON")
    parser.add_argument("--model", default="mistral", help="Ollama model for judge")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host")
    args = parser.parse_args()

    with open(args.ablation_results, encoding="utf-8") as f:
        ablation_data = json.load(f)

    results = ablation_data.get("results", [])
    successful = [r for r in results if "error" not in r]

    print(f"Evaluating {len(successful)} ablation runs with judge...")

    judge = Judge(model=args.model, ollama_host=args.ollama_host)
    judge_outputs = []

    for run in successful:
        doc_id = run["document"]
        mode = run["mode"]
        summary = run["summary"]
        keywords = run.get("keywords", [])
        # Load source document
        doc_path = Path(args.samples_dir) / f"{doc_id}.txt"
        if not doc_path.exists():
            print(f"Warning: document {doc_id} not found, skipping.")
            continue
        document = doc_path.read_text(encoding="utf-8")

        print(f"  Judging {doc_id} / {mode}...")
        try:
            result = judge.evaluate(document, summary, keywords)
            judge_outputs.append({
                "document": doc_id,
                "mode": mode,
                "judge": result,
            })
        except Exception as e:
            print(f"    Error: {e}")
            judge_outputs.append({
                "document": doc_id,
                "mode": mode,
                "error": str(e),
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(judge_outputs, f, indent=2)

    print(f"Saved judge results to {output_path}")


if __name__ == "__main__":
    main()