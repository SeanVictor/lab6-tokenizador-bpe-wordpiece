"""
Tarefa 3: Integração Industrial e WordPiece

Usa o tokenizador BERT multilingual (bert-base-multilingual-cased)
do Hugging Face para explorar o algoritmo WordPiece na prática.

O WordPiece difere do BPE clássico:
    - BPE  : funde o par mais FREQUENTE
    - WP   : funde o par que MAXIMIZA a probabilidade do corpus
              score(A,B) = freq(AB) / (freq(A) × freq(B))

O símbolo ## indica que o token é uma sub-palavra CONTINUAÇÃO
(não inicia uma palavra), permitindo reconstrução do texto original.

Aluno      : Sean Victor Machado de Moraes
GitHub     : https://github.com/SeanVictor
Disciplina : Tópicos em Inteligência Artificial – 2026.1
Instituição: iCEV
"""

from transformers import AutoTokenizer


def load_wordpiece_tokenizer(model_name: str = "bert-base-multilingual-cased"):
    """Carrega e retorna o tokenizador WordPiece do BERT multilingual."""
    print(f"\n  Carregando tokenizador: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"  Vocab size : {tokenizer.vocab_size:,} tokens")
    print(f"  Tipo       : {type(tokenizer).__name__}")
    return tokenizer


def analyze_wordpiece(tokenizer, sentence: str) -> list:
    """
    Tokeniza a frase usando WordPiece e imprime análise detalhada.

    Parâmetros:
        tokenizer : tokenizador BERT
        sentence  : frase de teste

    Retorna:
        tokens : list[str]
    """
    tokens = tokenizer.tokenize(sentence)
    ids    = tokenizer.convert_tokens_to_ids(tokens)

    print(f"\n  Frase original:")
    print(f"    '{sentence}'")
    print(f"\n  Tokens WordPiece ({len(tokens)} tokens):")
    print(f"  {'#':<5} {'Token':<25} {'ID':>8}  {'Tipo'}")
    print(f"  {'-'*55}")

    for i, (tok, tid) in enumerate(zip(tokens, ids)):
        tipo = "continuação (##)" if tok.startswith("##") else "início de palavra"
        print(f"  {i:<5} {tok:<25} {tid:>8}  {tipo}")

    # Agrupa tokens por palavra original
    print(f"\n  Reconstrução das palavras:")
    palavras = []
    palavra_atual = []
    for tok in tokens:
        if tok.startswith("##"):
            palavra_atual.append(tok[2:])   # remove ##
        else:
            if palavra_atual:
                palavras.append("".join(palavra_atual))
            palavra_atual = [tok]
    if palavra_atual:
        palavras.append("".join(palavra_atual))

    for p in palavras:
        print(f"    → '{p}'")

    # Destaca tokens com ##
    tokens_continuacao = [t for t in tokens if t.startswith("##")]
    print(f"\n  Tokens de continuação (##): {tokens_continuacao}")

    return tokens


def wordpiece_vs_bpe_comparison():
    """Imprime uma comparação didática entre BPE e WordPiece."""
    print("\n" + "=" * 60)
    print("  BPE vs WordPiece — Comparação")
    print("=" * 60)
    print("""
  ┌─────────────────┬──────────────────────────────────────┐
  │ Característica  │ BPE              │ WordPiece          │
  ├─────────────────┼──────────────────┼────────────────────┤
  │ Critério merge  │ maior frequência │ maior P(corpus)    │
  │ Score           │ freq(AB)         │ freq(AB)/freq(A)   │
  │                 │                  │         ×freq(B)   │
  │ Marcador        │ </w> (fim)       │ ## (continuação)   │
  │ Usado em        │ GPT, RoBERTa     │ BERT, mBERT        │
  └─────────────────┴──────────────────┴────────────────────┘
    """)


# ─────────────────────────────────────────────────────────────
# Execução direta
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Tarefa 3 — WordPiece (Hugging Face BERT)")
    print("=" * 60)

    tokenizer = load_wordpiece_tokenizer()

    # Frase de teste obrigatória do enunciado
    frase_teste = (
        "Os hiper-parâmetros do transformer são "
        "inconstitucionalmente difíceis de ajustar."
    )

    tokens = analyze_wordpiece(tokenizer, frase_teste)

    wordpiece_vs_bpe_comparison()

    print("  ✓ Tarefa 3 — WordPiece analisado com sucesso!")
    print("=" * 60)
