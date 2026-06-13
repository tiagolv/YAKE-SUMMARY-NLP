"""Gradio interface for the YAKE + LLM local pipeline.

Provides a web-based UI for interactive summarization with keyword extraction.

Usage:
    python app.py
    python app.py --config config.yaml --port 7860
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio is required for the web interface. "
        "Install it with: pip install gradio"
    )

from src.evaluation import evaluate_metrics
from src.keyword_extractor import YakeConfig, YakeKeywordExtractor
from src.llm_interface import LLMConfig, LLMError, check_backend, create_llm_client
from src.preprocessor import normalize_text, truncate_text
from src.prompt_builder import (
    PromptConfig,
    build_prompt,
    build_prompt_keywords_only,
    build_prompt_no_keywords,
)
from src.prompt_templates import TEMPLATES, build_from_template


# Global config — loaded at startup
_config: dict = {}


def load_config(config_path: str = "config.yaml") -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    # Allow overriding the Ollama host via environment (useful on WSL, where
    # the Windows-hosted Ollama is not always reachable via "localhost").
    env_host = os.environ.get("OLLAMA_HOST")
    if env_host:
        config.setdefault("llm", {})["ollama_host"] = env_host
    return config


def backend_status_md() -> str:
    """Render the LLM backend status as a Markdown banner."""
    available, message = check_backend(LLMConfig(**_config["llm"]))
    icon = "🟢" if available else "🔴"
    css_class = "status-ok" if available else "status-down"
    return f"<div class='{css_class}'>{icon} {message}</div>"


def summarize(
    text: str,
    top_k: int,
    max_ngram_size: int,
    temperature: float,
    prompt_template: str,
    ablation_mode: str,
    external_keywords_text: str,
) -> tuple[str, dict, dict, str, str]:
    """Core summarization function for the Gradio interface.

    Returns: (summary, keywords, metrics, prompt_used, status_md)
    """
    # Gradio may pass None for empty textboxes — normalize before any .strip().
    text = text or ""
    external_keywords_text = external_keywords_text or ""

    if not text.strip():
        empty = {"info": "Introduza texto para resumir."}
        return "Introduza texto para resumir.", empty, empty, "", backend_status_md()

    clean_text = normalize_text(text)

    # Keyword extraction
    yake_config = YakeConfig(
        language=_config["yake"]["language"],
        max_ngram_size=max_ngram_size,
        deduplication_threshold=_config["yake"].get("deduplication_threshold", 0.9),
        top_k=top_k,
    )
    extractor = YakeKeywordExtractor(yake_config)

    # Decide keyword source
    if external_keywords_text.strip():
        keywords = [kw.strip() for kw in external_keywords_text.split(",") if kw.strip()]
        keywords_scored = [(kw, 0.0) for kw in keywords]
        keyword_source = "external (user-provided)"
    else:
        keywords_scored = extractor.extract(clean_text)
        keywords = [kw for kw, _ in keywords_scored]
        keyword_source = "YAKE! (local)"

    max_chars = _config["prompt"].get("max_text_chars", 3000)
    truncated_text = truncate_text(clean_text, max_chars)

    # Build prompt
    prompt_config = PromptConfig(**_config["prompt"])

    if ablation_mode == "No Keywords (text only)":
        prompt = build_prompt_no_keywords(truncated_text, prompt_config)
    elif ablation_mode == "Keywords Only (no text)":
        prompt = build_prompt_keywords_only(keywords, prompt_config)
    elif prompt_template != "zero_shot (default)":
        template_key = prompt_template.split(" ")[0]
        template = TEMPLATES.get(template_key)
        if template:
            prompt = build_from_template(template, keywords, truncated_text)
        else:
            prompt = build_prompt(keywords, truncated_text, prompt_config)
    else:
        prompt = build_prompt(keywords, truncated_text, prompt_config)

    # Keywords are computed locally (offline) and shown regardless of the LLM.
    kw_display = {
        "source": keyword_source,
        "keywords": [
            {"keyword": kw, "score": round(score, 6)}
            for kw, score in keywords_scored
        ],
    }

    # Generate summary — the only step that needs the (local) LLM backend.
    llm_params = {**_config["llm"], "temperature": temperature}
    llm_config = LLMConfig(**llm_params)
    try:
        llm = create_llm_client(llm_config)
        summary = llm.generate(prompt)
    except (LLMError, Exception) as exc:  # noqa: BLE001 — never crash the UI
        summary = (
            f"⚠️ Backend LLM inacessível: {exc}\n\n"
            "As keywords e o prompt foram gerados localmente. "
            "Inicie o Ollama (ollama serve) ou configure llama_cpp em "
            "config.yaml para gerar o resumo. Em WSL, defina OLLAMA_HOST "
            "para o IP do host Windows se necessário."
        )
        metrics_display = {"info": "Métricas indisponíveis sem resumo do LLM."}
        status = (
            "<div class='status-down'>🔴 Resumo não gerado — "
            "backend LLM inacessível.</div>"
        )
        return summary, kw_display, metrics_display, prompt, status

    # Evaluate (only when a summary was produced)
    metrics = evaluate_metrics(
        summary=summary,
        source_keywords=keywords,
        extractor=extractor,
    )
    metrics_display = {
        "keyword_coverage": {
            "coverage": round(metrics["keyword_coverage"]["coverage"], 4),
            "found": metrics["keyword_coverage"]["found"],
            "missing": metrics["keyword_coverage"]["missing"],
        },
        "summary_alignment": {
            "precision": round(metrics["summary_alignment"]["precision"], 4),
            "recall": round(metrics["summary_alignment"]["recall"], 4),
        },
    }
    status = "<div class='status-ok'>🟢 Resumo gerado com sucesso.</div>"

    return summary, kw_display, metrics_display, prompt, status


CSS = """
.gradio-container { max-width: 1400px !important; margin: auto; }
#status-banner { padding: 8px 12px; border-radius: 8px; font-size: .9em; }
.status-ok   { background: #d1fae5; color: #065f46; }
.status-down { background: #fee2e2; color: #991b1b; }
"""

HEADER = (
    "# 🔑 YAKE + LLM Local Pipeline\n"
    "Extrai keywords com YAKE! e gera resumos guiados por keywords com um LLM local. "
    "Totalmente offline — sem chamadas a APIs externas."
)


def build_interface() -> gr.Blocks:
    """Build the Gradio interface with Sidebar + Tabs layout."""
    template_choices = [
        "zero_shot (default)",
        "structured (bullet points)",
        "analytical (synthesis)",
        "chain_of_thought (step-by-step)",
    ]
    ablation_choices = [
        "Full Pipeline",
        "No Keywords (text only)",
        "Keywords Only (no text)",
    ]

    sample_dir = Path("data/samples")
    sample_names: list[str] = []
    if sample_dir.exists():
        sample_names = [s.stem for s in sorted(sample_dir.glob("*.txt"))]

    def load_sample(name: str) -> str:
        path = sample_dir / f"{name}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""

    with gr.Blocks(
        title="YAKE + LLM Local Pipeline",
        fill_height=True,
    ) as demo:

        # ── Sidebar: todos os controlos ──────────────────────────────────────
        with gr.Sidebar(open=True, width=340):

            gr.Markdown("### 📚 Exemplos")
            sample_dropdown = gr.Dropdown(
                choices=sample_names,
                label="Carregar documento de exemplo",
                info="Seleciona um exemplo para preencher o campo de texto.",
            )

            gr.Markdown("### ⚙️ Parâmetros")
            top_k = gr.Slider(
                minimum=3, maximum=30, value=12, step=1,
                label="Top-K Keywords",
            )
            max_ngram = gr.Slider(
                minimum=1, maximum=5, value=3, step=1,
                label="Tamanho máx. N-grama",
            )
            temperature = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                label="Temperatura do LLM",
            )

            with gr.Accordion("🧪 Opções Avançadas", open=False):
                prompt_template = gr.Dropdown(
                    choices=template_choices,
                    value="zero_shot (default)",
                    label="Template de Prompt",
                )
                ablation_mode = gr.Dropdown(
                    choices=ablation_choices,
                    value="Full Pipeline",
                    label="Modo de Ablação",
                )
                external_kw = gr.Textbox(
                    label="Keywords Externas",
                    placeholder="keyword1, keyword2, keyword3...",
                    info="Separadas por vírgula. Deixa vazio para usar o YAKE!.",
                )

            run_btn = gr.Button("🚀 Gerar Resumo", variant="primary", size="lg")

        # ── Área principal ───────────────────────────────────────────────────
        with gr.Column():
            gr.Markdown(HEADER)
            status_banner = gr.Markdown(elem_id="status-banner")

            with gr.Row(equal_height=True):
                # Coluna esquerda: input
                with gr.Column(scale=1, min_width=320):
                    text_input = gr.Textbox(
                        label="📄 Texto de Entrada",
                        placeholder="Cola aqui o texto do documento...",
                        lines=18,
                        max_lines=22,
                    )

                # Coluna direita: outputs em separadores
                with gr.Column(scale=1, min_width=320):
                    with gr.Tabs():
                        with gr.Tab("📝 Resumo"):
                            summary_output = gr.Textbox(
                                label="Resumo gerado",
                                lines=16,
                            )
                        with gr.Tab("🔑 Keywords"):
                            keywords_output = gr.JSON(label="Keywords extraídas")
                        with gr.Tab("📊 Métricas"):
                            metrics_output = gr.JSON(label="Métricas de avaliação")
                        with gr.Tab("🔍 Prompt"):
                            prompt_output = gr.Code(
                                label="Prompt completo",
                                language="markdown",
                                lines=16,
                            )

        # ── Wiring ───────────────────────────────────────────────────────────
        sample_dropdown.change(
            fn=load_sample,
            inputs=[sample_dropdown],
            outputs=[text_input],
        )

        run_btn.click(
            fn=summarize,
            inputs=[
                text_input,
                top_k,
                max_ngram,
                temperature,
                prompt_template,
                ablation_mode,
                external_kw,
            ],
            outputs=[
                summary_output,
                keywords_output,
                metrics_output,
                prompt_output,
                status_banner,
            ],
        )

        # Mostra o estado do backend ao abrir a página
        demo.load(fn=backend_status_md, outputs=[status_banner])

    return demo


def main() -> None:
    global _config

    parser = argparse.ArgumentParser(description="YAKE + LLM Gradio Interface")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    _config = load_config(args.config)
    demo = build_interface()
    demo.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(primary_hue="indigo", spacing_size="sm"),
        css=CSS,
    )


if __name__ == "__main__":
    main()