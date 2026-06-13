# Guia de Demonstração

Este guia mostra como demonstrar o uso da aplicação YAKE + LLM local, tanto pela linha de comando como pela interface web.

## 1. Objetivo da demonstração

A aplicação recebe um texto, extrai keywords com YAKE e gera um resumo com um LLM local. No fim, apresenta:

- o resumo gerado
- as keywords extraídas
- métricas de cobertura e alinhamento
- o prompt usado na geração

## 2. Pré-requisitos

- Python 3.10 ou superior
- Dependências instaladas com `pip install -r requirements.txt`
- Um backend de LLM local ativo:
  - `ollama` a correr em `http://localhost:11434`, ou
  - `llama_cpp` configurado com um modelo GGUF local

## 3. Demonstração rápida pela linha de comando

Use um dos ficheiros de exemplo incluídos no repositório:

```powershell
python -m src.pipeline --input data/samples/paper_yake.txt --config config.yaml --output outputs/run.json
```

O comando acima faz o seguinte:

1. lê o texto de entrada
2. normaliza o conteúdo
3. extrai keywords com YAKE
4. constrói o prompt
5. gera o resumo com o backend local
6. guarda o resultado em `outputs/run.json`

Se quiseres mostrar uma variante da demo, podes usar keywords externas ou modos de ablação:

```powershell
python -m src.pipeline --input data/samples/paper_yake.txt --config config.yaml --external-keywords "yake, keyword extraction, summarization"
python -m src.pipeline --input data/samples/paper_yake.txt --config config.yaml --ablation no_keywords
python -m src.pipeline --input data/samples/paper_yake.txt --config config.yaml --ablation keywords_only
```

## 4. Demonstração pela interface web

Executa a app Gradio com:

```powershell
python app.py --config config.yaml --port 7860
```

Depois abre a interface no browser e segue este fluxo:

1. cola um texto na caixa de entrada, ou escolhe um documento de exemplo
2. ajusta `Top-K Keywords`, `Max N-gram Size` e `LLM Temperature`
3. opcionalmente escolhe um template de prompt ou um modo de ablação
4. clica em `Generate Summary`
5. apresenta o resumo, as keywords e as métricas na coluna da direita

## 5. Roteiro sugerido para apresentação

Uma sequência simples de demonstração é:

1. abrir a interface web
2. carregar `paper_yake.txt` a partir dos exemplos
3. mostrar as keywords extraídas automaticamente
4. gerar o resumo com o modo `Full Pipeline`
5. alternar para `No Keywords (text only)` e comparar o resultado
6. alternar para `Keywords Only (no text)` e explicar o impacto da remoção do texto
7. voltar ao pipeline completo e mostrar as métricas finais

## 6. O que observar na demo

- se as keywords capturam os conceitos centrais do texto
- se o resumo mantém cobertura das keywords principais
- como os modos de ablação alteram a qualidade do resultado
- como o prompt influencia o estilo da resposta

## 7. Problemas comuns

- Se o comando falhar com erro de ligação, confirma que o Ollama está a correr localmente.
- Se estiveres a usar `llama_cpp`, confirma que `model_path` está definido em `config.yaml`.
- Se a interface não abrir, verifica se a porta `7860` está livre ou muda o valor de `--port`.

## 8. Próximo passo

Para uma demonstração mais completa, consulta também o notebook [notebooks/03_pipeline_completo.ipynb](../notebooks/03_pipeline_completo.ipynb).