#!/usr/bin/env python3
"""Fetch YAKE! demo samples from the live website and save as .txt files.

Usage:
    python scripts/fetch_yake_demo_samples.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests

DEMO_JS_URL = "http://yake.inesctec.pt/assets/js/demo.js"
SAMPLES_DIR = Path("data/samples")
GOLD_JSON_PATH = Path("data/eval/inspec_gold.json")


def extract_demo_samples(js_content: str) -> dict[str, str]:
    """
    Extract demoSamples using regex to isolate the object, then parse as JSON.
    Returns a dict mapping SampleX -> text content.
    """
    # Find the demoSamples object definition
    pattern = r'const demoSamples\s*=\s*(\{[\s\S]*?\n\});'
    match = re.search(pattern, js_content)
    if not match:
        raise ValueError("Could not find demoSamples object")

    samples_str = match.group(1)

    # Try to parse as JSON
    try:
        samples_obj = json.loads(samples_str)
    except json.JSONDecodeError as e:
        # If fails, try to fix common issues: remove trailing commas, etc.
        # Simple fix: remove commas before } or ]
        samples_str = re.sub(r',\s*}', '}', samples_str)
        samples_str = re.sub(r',\s*]', ']', samples_str)
        samples_obj = json.loads(samples_str)

    # Filter only entries that start with "Sample" and extract the "text" field
    samples = {}
    for key, value in samples_obj.items():
        if key.startswith("Sample") and isinstance(value, dict) and "text" in value:
            samples[key] = value["text"].strip()

    if not samples:
        raise ValueError("No sample texts extracted (expected keys like 'Sample1', 'Sample2', ...)")

    return samples


def main() -> None:
    print("Fetching YAKE! demo samples...")
    resp = requests.get(DEMO_JS_URL, timeout=10)
    resp.raise_for_status()
    js_content = resp.text

    samples = extract_demo_samples(js_content)
    print(f"Found {len(samples)} sample documents: {list(samples.keys())}")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Guarda cada sample como ficheiro .txt
    for sample_id, content in samples.items():
        file_path = SAMPLES_DIR / f"{sample_id}.txt"
        file_path.write_text(content, encoding="utf-8")
        print(f"Saved: {file_path}")

    # Adiciona entradas ao inspec_gold.json se não existirem
    if GOLD_JSON_PATH.exists():
        import json
        gold_data = json.loads(GOLD_JSON_PATH.read_text(encoding="utf-8"))
        existing_ids = {doc["id"] for doc in gold_data["documents"]}
        added = 0
        for sample_id in samples.keys():
            if sample_id not in existing_ids:
                gold_data["documents"].append({
                    "id": sample_id,
                    "gold_keywords": []   # placeholder, preencher manualmente depois
                })
                added += 1
        if added:
            GOLD_JSON_PATH.write_text(json.dumps(gold_data, indent=2), encoding="utf-8")
            print(f"Added {added} placeholder gold entries to {GOLD_JSON_PATH}")
        else:
            print("All sample IDs already have gold entries.")
    else:
        print(f"Warning: {GOLD_JSON_PATH} not found; gold entries not updated.")

    print("Done.")


if __name__ == "__main__":
    main()