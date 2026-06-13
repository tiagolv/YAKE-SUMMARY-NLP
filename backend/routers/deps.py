"""Shared FastAPI dependencies — config access and pipeline helpers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

from src.keyword_extractor import YakeConfig, YakeKeywordExtractor
from src.llm_interface import LLMConfig
from src.prompt_builder import PromptConfig
from src.prompt_templates import TEMPLATES, build_from_template
from src.prompt_builder import build_prompt, build_prompt_keywords_only, build_prompt_no_keywords


@lru_cache(maxsize=1)
def _load_raw_config(config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env_host = os.environ.get("OLLAMA_HOST")
    if env_host:
        cfg.setdefault("llm", {})["ollama_host"] = env_host
    return cfg


# Module-level mutable reference — set once at startup from main.py
_config_path: str = ""


def set_config_path(path: str) -> None:
    global _config_path
    _config_path = path
    _load_raw_config.cache_clear()


def get_raw_config() -> dict:
    path = _config_path or os.environ.get("CLNP_CONFIG", "config.yaml")
    return _load_raw_config(path)


def get_llm_config() -> LLMConfig:
    return LLMConfig(**get_raw_config()["llm"])


def get_yake_config(top_k: int, max_ngram_size: int) -> YakeConfig:
    base = get_raw_config()["yake"]
    return YakeConfig(
        language=base["language"],
        max_ngram_size=max_ngram_size,
        deduplication_threshold=base.get("deduplication_threshold", 0.9),
        top_k=top_k,
    )


def get_prompt_config() -> PromptConfig:
    return PromptConfig(**get_raw_config()["prompt"])


def build_prompt_for_mode(
    mode: str,
    template_key: str,
    keywords: list[str],
    truncated_text: str,
    prompt_config: PromptConfig,
) -> str:
    """Build a prompt string given ablation mode and template."""
    if mode == "no_keywords":
        return build_prompt_no_keywords(truncated_text, prompt_config)
    if mode == "keywords_only":
        return build_prompt_keywords_only(keywords, prompt_config)
    # full — check template
    if template_key and template_key != "zero_shot":
        template = TEMPLATES.get(template_key)
        if template:
            return build_from_template(template, keywords, truncated_text)
    return build_prompt(keywords, truncated_text, prompt_config)