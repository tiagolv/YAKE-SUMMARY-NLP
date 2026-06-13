"""Prompt template strategies for comparing different summarization approaches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class PromptTemplate:
    name: str
    description: str
    system: str
    instruction_template: str  # uses {keywords} and {text} placeholders


TEMPLATES: dict[str, PromptTemplate] = {
    "zero_shot": PromptTemplate(
        name="zero_shot",
        description="Direct summarization with keyword guidance (baseline)",
        system="You are a concise scientific summarizer.",
        instruction_template=(
            "Given these keywords extracted from the document, write a concise "
            "summary that covers the key concepts.\n\n"
            "Keywords: [{keywords}]\n\n"
            "Document:\n{text}"
        ),
    ),
    "structured": PromptTemplate(
        name="structured",
        description="Bullet-point summary covering each keyword topic",
        system="You are a precise scientific summarizer that produces structured outputs.",
        instruction_template=(
            "The following keywords were extracted from a scientific document. "
            "Write a structured summary using bullet points, where each bullet "
            "addresses one or more of the keywords.\n\n"
            "Keywords: [{keywords}]\n\n"
            "Document:\n{text}"
        ),
    ),
    "analytical": PromptTemplate(
        name="analytical",
        description="Analytical synthesis connecting keyword relationships",
        system="You are an analytical scientific writer.",
        instruction_template=(
            "Analyze how the following keywords relate to each other in the "
            "context of the document below. Write a synthesis that explains "
            "the connections between the key concepts.\n\n"
            "Keywords: [{keywords}]\n\n"
            "Document:\n{text}"
        ),
    ),
    "chain_of_thought": PromptTemplate(
        name="chain_of_thought",
        description="Step-by-step reasoning before summarization",
        system="You are a methodical scientific summarizer.",
        instruction_template=(
            "Step 1: Group the following keywords by theme.\n"
            "Step 2: For each theme, identify the main claim from the document.\n"
            "Step 3: Write a concise summary that integrates all themes.\n\n"
            "Keywords: [{keywords}]\n\n"
            "Document:\n{text}"
        ),
    ),
}


def build_from_template(
    template: PromptTemplate,
    keywords: Iterable[str],
    text: str,
) -> str:
    keywords_str = ", ".join(keywords)
    body = template.instruction_template.format(keywords=keywords_str, text=text)
    return f"{template.system}\n\n{body}"


def get_template(name: str) -> PromptTemplate:
    if name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Unknown template '{name}'. Available: {available}")
    return TEMPLATES[name]


def list_templates() -> list[str]:
    return list(TEMPLATES.keys())
