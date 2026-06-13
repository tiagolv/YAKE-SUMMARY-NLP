"""Load external keywords from JSON files for comparison experiments."""

from __future__ import annotations

import json
from pathlib import Path


def load_external_keywords(path: Path, doc_id: str | None = None) -> list[str]:
    """Load external keywords for a specific document.

    If path is a directory and doc_id is provided, loads from <path>/<doc_id>.json.
    If path is a file, loads from it. If the file contains a unified list,
    extracts the keywords for doc_id if provided.
    """
    if path.is_dir():
        if not doc_id:
            raise ValueError("doc_id is required when path is a directory.")
        file_path = path / f"{doc_id}.json"
    else:
        file_path = path

    if not file_path.exists():
        raise FileNotFoundError(f"External keywords file not found at {file_path}")

    data = json.loads(file_path.read_text(encoding="utf-8"))
    
    if isinstance(data, dict):
        # Check if it has a unified "documents" structure
        if "documents" in data:
            if doc_id and doc_id in data["documents"]:
                return data["documents"][doc_id].get("keywords", [])
            elif doc_id:
                raise KeyError(f"Document '{doc_id}' not found in unified data.")
        # Check if it has document-specific structure
        return data.get("keywords", [])
    elif isinstance(data, list):
        return data
    return []


def load_all_external_keywords(path: Path) -> dict[str, list[str]]:
    """Load all external keywords.

    If path is a directory, loads from all JSON files in that directory
    (using filename stem as the document ID).
    If path is a file, loads from the unified JSON structure.
    """
    if path.is_dir():
        results = {}
        for file_path in path.glob("*.json"):
            try:
                kws = load_external_keywords(file_path)
                results[file_path.stem] = kws
            except Exception:
                pass
        return results

    data = json.loads(path.read_text(encoding="utf-8"))
    documents = data.get("documents", {})
    return {
        doc_id: entry.get("keywords", [])
        for doc_id, entry in documents.items()
    }
