"""
Tarefa 1: O Motor de Frequências

O algoritmo BPE inicia com um corpus onde as palavras estão
separadas em caracteres com símbolo especial de fim </w>.
Esta função varre o corpus contando a frequência de cada
par de símbolos adjacentes.

Aluno      : Sean Victor Machado de Moraes
GitHub     : https://github.com/SeanVictor
Disciplina : Tópicos em Inteligência Artificial – 2026.1
Instituição: iCEV
"""

# ─────────────────────────────────────────────────────────────
# Corpus de treinamento (exatamente como definido no enunciado)
# ─────────────────────────────────────────────────────────────
vocab = {
    'l o w </w>':       5,
    'l o w e r </w>':  2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3,
}


def get_stats(vocab: dict) -> dict:
    """
    Varre o corpus e conta a frequência de cada par adjacente
    de símbolos (caracteres ou tokens já fundidos).

    Parâmetro:
        vocab : dict  —  { 'símbolo1 símbolo2 ... </w>': frequência }

    Retorna:
        pairs : dict  —  { ('a', 'b'): contagem_total }

    Exemplo:
        'n e w e s t </w>': 6
        → pares: (n,e):6  (e,w):6  (w,e):6  (e,s):6  (s,t):6  (t,</w>):6
    """
    pairs = {}

    for word, freq in vocab.items():
        symbols = word.split()          # divide a string em lista de símbolos

        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq

    return pairs


# ─────────────────────────────────────────────────────────────
# Execução / Validação
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Tarefa 1 — Motor de Frequências BPE")
    print("=" * 60)

    print("\n  Corpus de treinamento:")
    for word, freq in vocab.items():
        print(f"    '{word}' : {freq}")

    pairs = get_stats(vocab)

    # Ordena por frequência decrescente para facilitar leitura
    sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  Frequências de todos os pares adjacentes:")
    print(f"  {'Par':<20} {'Contagem':>8}")
    print(f"  {'-'*30}")
    for pair, count in sorted_pairs:
        destaque = " ← MÁXIMO" if pair == ('e', 's') else ""
        print(f"  {str(pair):<20} {count:>8}{destaque}")

    # ── Validação obrigatória do enunciado ────────────────────
    print(f"\n  Validação: par ('e', 's') = {pairs[('e', 's')]}")
    assert pairs[('e', 's')] == 9, \
        f"FALHOU: esperado 9, obtido {pairs[('e', 's')]}"
    print("  ✓ APROVADO — ('e', 's') retorna contagem máxima de 9")
    print("    (6 de 'newest' + 3 de 'widest' = 9)")
    print("=" * 60)
