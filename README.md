# YAKE + LLM Local Pipeline

Projeto para extrair keywords com YAKE! e gerar um resumo guiado por essas keywords usando um LLM local (llama.cpp ou Ollama), sem chamadas a APIs externas.

## `yake-sum` — pacote e CLI

Resumo extrativo (sem LLM), abstrativo guiado por keywords YAKE, ou híbrido (documentos longos).

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -e ".[server,dev]"
pytest -q                                                # offline, sem GPU nem Ollama

yake-sum -i data/samples/paper_yake.txt -m extractive -s 3          # zero-LLM
yake-sum --check-backend -b ollama --model mistral                  # verifica o Ollama
yake-sum -i doc.txt -m abstractive -b ollama --model mistral --json # resumo + métricas
yake-sum -i longo.txt -m hybrid -b ollama --max-context-chars 3000  # documentos longos
```

```python
from yake_sum import Summarizer
res = Summarizer(mode="hybrid", backend="ollama", model="mistral").summarize(text)
res.text, res.metrics["faithfulness"], res.metrics.get("warnings")
```

**Confiança:** o backend `mock` é só para testes (o CLI/API avisam). Para resultados reais usa um modelo local;
`YAKE_SUM_LIVE=1 pytest tests/test_live_ollama.py -s` faz um teste ponta-a-ponta com Ollama.
`metrics["faithfulness"]` assinala números/termos do resumo que não existem no texto-fonte (heurística de revisão, não prova).

**Benchmark:** `python -m benchmarks.run_rouge_study --backend mock` (smoke test) ou `--backend ollama --model mistral`
(resultados reais). Referências com `reference_source: unverified` devem ser revistas antes de publicar números.

## Objetivo
- Pipeline 100% local, capaz de operar offline
- Combinar extracao estatistica (YAKE!) com geracao via LLM
- Avaliar cobertura das keywords no resumo (metrica principal)

## Estrutura
- data/: documentos de exemplo e dados de avaliacao
- src/: codigo reutilizavel do pipeline
- notebooks/: demonstracoes e figuras para o relatorio
- models/: modelos locais (apenas referencia no repo)
- outputs/: resultados gerados
- report/: template LaTeX do relatorio

## Requisitos
- Python 3.10+
- (Opcional) Modelo GGUF em models/ (ex: Mistral 7B instruct quantizado)
- (Opcional) Ollama local instalado com modelo baixado

## Instalacao
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuracao
Edite o [config.yaml](config.yaml) para ajustar:
- parametros do YAKE! (n-gramas, top_k)
- backend do LLM (llama_cpp ou ollama)
- limite de texto para o prompt

## Execucao rapida
```powershell
python -m src.pipeline --input data/samples/paper_yake.txt --config config.yaml --output outputs/run.json
```

## Notebooks
- [notebooks/01_exploracao_yake.ipynb](notebooks/01_exploracao_yake.ipynb)
- [notebooks/02_llm_local_setup.ipynb](notebooks/02_llm_local_setup.ipynb)
- [notebooks/03_pipeline_completo.ipynb](notebooks/03_pipeline_completo.ipynb)

## Guia de Demonstração
- [docs/guia_demonstracao.md](docs/guia_demonstracao.md)

## Localidade (sem APIs externas)
O sistema assume modelos carregados localmente (GGUF via llama-cpp-python) ou um servidor Ollama local. Nenhuma chamada a APIs externas e nenhuma dependencia de internet durante a execucao.