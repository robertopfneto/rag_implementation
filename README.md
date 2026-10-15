# RAG simples

Implementação mínima de um sistema **RAG** (Retrieval-Augmented Generation) em um único notebook, [simple_rag.ipynb](simple_rag.ipynb).
Tudo roda localmente com o [Ollama](https://ollama.com), sem API externa.

Baseado em: https://huggingface.co/blog/ngxson/make-your-own-rag

## O que é RAG, em uma frase

Em vez de perguntar direto ao modelo de linguagem, primeiro **buscamos** os trechos do nosso texto mais parecidos com a pergunta e os entregamos ao modelo como contexto para ele responder.

## Como o código funciona

O fluxo é: **indexação → recuperação → geração**.

| # | Célula | O que faz |
|---|--------|-----------|
| 1 | Carregar dataset | Lê `dataset/cat-facts.txt`. Cada linha vira um *chunk* (trecho). |
| 2 | Banco de vetores | Para cada chunk, gera um *embedding* (vetor numérico que representa o sentido do texto) com o modelo de embedding e guarda `(chunk, embedding)` na lista `VECTOR_DB`. |
| 3 | Recuperação | `recuperacao(query, top_n=3)` gera o embedding da pergunta, calcula a similaridade de cosseno contra todos os chunks e devolve os `top_n` mais parecidos. |
| 4 | Prompt | Pede uma pergunta ao usuário, recupera os chunks e monta o `instruction_prompt` mandando o modelo responder **somente** com aquele contexto. |
| 5 | Geração | Envia o prompt + a pergunta ao modelo de linguagem e imprime a resposta em *streaming*. |

### Modelos usados

| Constante | Modelo | Papel |
|-----------|--------|-------|
| `EMBEDDING_MODEL` | `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf` | transforma texto em vetor |
| `LANGUAGE_MODEL` | `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF` | gera a resposta final |

### Principais funções e variáveis

- `VECTOR_DB`: lista em memória com tuplas `(chunk, embedding)`. Some ao reiniciar o kernel.
- `add_chunk_to_database(chunk)`: gera o embedding de um chunk e o adiciona ao `VECTOR_DB`.
- `similaridade_cosseno(a, b)`: similaridade de cosseno entre dois vetores (1 = muito parecidos, 0 = sem relação).
- `recuperacao(query, top_n=3)`: retorna lista de `(chunk, similaridade)` ordenada da mais para a menos relevante.

## Tutorial: como executar

### 1. Pré-requisitos

- Python **3.12 ou superior** (o f-string da célula 4 usa `'\n'` dentro de `{}`, o que só funciona a partir do 3.12).
- [Ollama](https://ollama.com/download) instalado e em execução.
- Jupyter (Jupyter Lab/Notebook ou a extensão do VS Code).

### 2. Instalar as dependências Python

Em um ambiente (conda, por exemplo) com Python 3.12+:

```bash
pip install -r requirements.txt
```

O `requirements.txt` contém `ollama`, `numpy` e `scikit-learn`.
Se for usar Jupyter Lab, instale também com `pip install jupyterlab`.

### 3. Baixar os modelos no Ollama

Com o Ollama rodando (`ollama serve`, ou o app aberto):

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

### 4. Conferir o dataset

O notebook espera o arquivo `dataset/cat-facts.txt`, com um fato por linha. Se a pasta `dataset/` estiver vazia, adicione o arquivo antes de continuar.

### 5. Rodar o notebook

```bash
jupyter lab simple_rag.ipynb
```

Execute as células **em ordem**, de cima para baixo:

1. Células 1 a 3 preparam o dataset, o banco de vetores e a função de busca. A célula 2 é a mais demorada, porque gera um embedding por linha.
2. Célula 4 pede a pergunta (em inglês, pois o dataset e os modelos estão em inglês) e mostra os trechos recuperados.
3. Célula 5 imprime a resposta do chatbot.

Para fazer outra pergunta, basta executar de novo as células 4 e 5. Não é preciso reindexar.

## Problemas comuns

- `connection refused` / erro do Ollama: o servidor não está rodando. Inicie com `ollama serve`.
- `model not found`: faltou o `ollama pull` do passo 3.
- `SyntaxError` na célula 4: Python anterior ao 3.12.
- Respostas fracas: o modelo de 1B parâmetros é pequeno; é esperado para um exemplo didático.
