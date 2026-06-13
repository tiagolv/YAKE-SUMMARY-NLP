# YAKE + LLM Local Pipeline

Projeto para extrair keywords com YAKE! e gerar um resumo guiado por essas keywords usando um LLM local (llama.cpp ou Ollama), sem chamadas a APIs externas.

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