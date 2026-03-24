# Laboratório 6 — Construindo um Tokenizador BPE e Explorando o WordPiece

**Disciplina:** Tópicos em Inteligência Artificial – 2026.1  
**Professor:** Prof. Dimmy Magalhães  
**Instituição:** iCEV - Instituto de Ensino Superior  
**Aluno:** Sean Victor Machado de Moraes  
**GitHub:** [SeanVictor](https://github.com/SeanVictor)  

---

## Descrição

Implementação do algoritmo **Byte Pair Encoding (BPE)** do zero e exploração
do algoritmo **WordPiece** usado pelo BERT, demonstrando como os modelos de
linguagem modernos transformam texto bruto em tensores numéricos de forma
eficiente.

---

## Estrutura do Repositório

```
lab6_bpe/
│
├── main.py              # Pipeline completo — executa as 3 tarefas
├── task1_get_stats.py   # Tarefa 1: get_stats — frequência de pares
├── task2_merge_vocab.py # Tarefa 2: merge_vocab + loop K=5 fusões
├── task3_wordpiece.py   # Tarefa 3: WordPiece com BERT (Hugging Face)
├── requirements.txt     # Dependências do projeto
└── README.md            # Este arquivo
```

---

## Como rodar

### 1. Clone o repositório

```bash
git clone https://github.com/SeanVictor/lab6-tokenizador-bpe-wordpiece.git
cd lab6-tokenizador-bpe-wordpiece
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Execute o pipeline completo

```bash
python main.py
```

### 4. Execute as tarefas individualmente (opcional)

```bash
python task1_get_stats.py    # Tarefa 1 — frequências dos pares
python task2_merge_vocab.py  # Tarefa 2 — loop de fusão BPE
python task3_wordpiece.py    # Tarefa 3 — WordPiece com BERT
```

---

## Algoritmo BPE — Resumo das 5 Iterações

| # | Par fundido | Token criado | Frequência |
|---|-------------|--------------|------------|
| 1 | `('e', 's')` | `es` | 9 |
| 2 | `('es', 't')` | `est` | 9 |
| 3 | `('est', '</w>')` | `est</w>` | 9 |
| 4 | `('l', 'o')` | `lo` | 7 |
| 5 | `('lo', 'w')` | `low` | 7 |

✓ Após 5 iterações, o sufixo morfológico **`est</w>`** é formado,
provando que o BPE aprende estruturas linguísticas reais a partir de frequências.

---

## Relatório: O que significa `##` no WordPiece?

Ao tokenizar a frase de teste com o BERT multilingual:

> *"Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."*

palavras longas como **"inconstitucionalmente"** são divididas em sub-palavras:

```
in  +  ##consti  +  ##tucion  +  ##almente
```

O símbolo **`##`** (cerquilha dupla) é o marcador de **continuação de palavra**
do algoritmo WordPiece. Ele indica que aquele token **não inicia uma palavra nova**
— ele é a sequência de uma palavra que já começou. Sem o `##`, o tokenizador
não saberia se `almente` é o começo de uma nova palavra ou o final de
"inconstitucionalmente".

### Por que sub-palavras impedem o travamento do modelo?

Um vocabulário de palavras completas (ex: 500.000 entradas) inevitavelmente
encontrará palavras desconhecidas (*out-of-vocabulary*, OOV) em textos novos.
Quando isso acontece, o modelo recebe um token genérico `[UNK]` e perde
completamente a informação semântica daquela palavra.

Com sub-palavras (BPE ou WordPiece), qualquer palavra jamais vista pode ser
decomposta em partes menores que **já existem no vocabulário**. A palavra
inventada *"transformerização"*, por exemplo, seria segmentada como:

```
transform  +  ##er  +  ##iza  +  ##ção
```

O modelo recebe informação útil sobre cada pedaço, em vez de um `[UNK]` vazio.
Isso permite vocabulários menores (32.000–37.000 tokens, como no paper original)
sem perder cobertura sobre qualquer texto, tornando os modelos robustos e eficientes.

---

## Diferença BPE vs WordPiece

| Característica | BPE | WordPiece |
|---|---|---|
| Critério de fusão | maior frequência do par | maximiza P(corpus) |
| Score | `freq(AB)` | `freq(AB) / (freq(A) × freq(B))` |
| Marcador especial | `</w>` (fim de palavra) | `##` (continuação) |
| Usado em | GPT, RoBERTa | BERT, mBERT |

---

## Nota de Integridade Acadêmica

Partes geradas/complementadas com IA (Claude), revisadas por **Sean Victor Machado de Moraes**.

A função `merge_vocab` utiliza expressões regulares (`re.sub`) para substituição
de pares isolados no vocabulário — este trecho foi construído com auxílio de IA
para garantir a correção do padrão regex, conforme permitido pelo contrato pedagógico.
A lógica do algoritmo BPE, a função `get_stats` e o loop de treinamento foram
compreendidos e documentados pelo aluno.

---

## Referências

- Sennrich et al. (2016). *Neural Machine Translation of Rare Words with Subword Units*.
  [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*.
  [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
- Tokenizador: [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)
