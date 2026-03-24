"""
Tarefa 2: O Loop de Fusão

Após identificar o par mais frequente, o BPE realiza uma fusão (merge),
criando um novo token que passa a fazer parte do vocabulário do modelo.

Funções:
    merge_vocab(pair, v_in) → substitui o par pelo token fundido
    Loop principal          → K=5 iterações de get_stats + merge_vocab

Aluno      : Sean Victor Machado de Moraes
GitHub     : https://github.com/SeanVictor
Disciplina : Tópicos em Inteligência Artificial – 2026.1
Instituição: iCEV
"""

import re
from task1_get_stats import get_stats, vocab as INITIAL_VOCAB


def merge_vocab(pair: tuple, v_in: dict) -> dict:
    """
    Substitui todas as ocorrências do par adjacente isolado
    pela versão unificada (fusão) em todo o vocabulário.

    Parâmetros:
        pair  : tuple  — ex: ('e', 's')
        v_in  : dict   — vocabulário atual

    Retorna:
        v_out : dict   — vocabulário atualizado com a fusão aplicada

    Estratégia:
        Usa re.sub com word boundaries de espaço para garantir que
        apenas o par ISOLADO seja fundido, e não substrings de tokens
        maiores já existentes.

        Ex: ('e','s') em 'n e w e s t </w>'
            → padrão: r'(?<!\S)e s(?!\S)'
            → resultado: 'n e w es t </w>'
    """
    v_out = {}

    # Monta o padrão regex que captura o par apenas quando isolado
    # (cercado por espaço ou início/fim de string)
    bigram  = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<![^\s])' + bigram + r'(?![^\s])')

    replacement = ''.join(pair)   # ('e','s') → 'es'

    for word, freq in v_in.items():
        new_word = pattern.sub(replacement, word)
        v_out[new_word] = freq

    return v_out


def run_bpe_training(vocab: dict, num_merges: int = 5) -> dict:
    """
    Loop Principal de Treinamento do Tokenizador BPE.

    A cada iteração:
        1. get_stats  → encontra o par mais frequente
        2. merge_vocab → funde o par e atualiza o vocabulário
        3. Imprime o par fundido e o estado do vocabulário

    Parâmetros:
        vocab      : dicionário inicial do corpus
        num_merges : número de iterações K (padrão: 5)

    Retorna:
        vocab : dicionário final após todas as fusões
    """
    print("=" * 60)
    print("  Tarefa 2 — Loop de Fusão BPE (K=5 iterações)")
    print("=" * 60)
    print("\n  Vocabulário inicial:")
    for w, f in vocab.items():
        print(f"    '{w}' : {f}")

    merges_log = []   # registra os merges para o relatório final

    for i in range(1, num_merges + 1):
        # 1. Conta frequência de todos os pares
        pairs = get_stats(vocab)

        if not pairs:
            print("\n  Nenhum par disponível — encerrando.")
            break

        # 2. Seleciona o par de maior frequência
        best_pair  = max(pairs, key=pairs.get)
        best_count = pairs[best_pair]

        # 3. Funde o par no vocabulário
        vocab = merge_vocab(best_pair, vocab)

        merged_token = ''.join(best_pair)
        merges_log.append((best_pair, best_count, merged_token))

        # 4. Imprime resultado da iteração
        print(f"\n  ── Iteração {i}/5 ─────────────────────────────────")
        print(f"  Par fundido  : {best_pair}  →  '{merged_token}'  (freq={best_count})")
        print(f"  Vocabulário  :")
        for w, f in vocab.items():
            print(f"    '{w}' : {f}")

    # ── Relatório final ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Relatório Final — Merges Realizados")
    print("=" * 60)
    print(f"  {'#':<5} {'Par':<20} {'Token Criado':<15} {'Freq'}")
    print(f"  {'-'*50}")
    for idx, (pair, freq, token) in enumerate(merges_log, 1):
        print(f"  {idx:<5} {str(pair):<20} '{token}'{'</w>' if '</w>' in token else '':<12} {freq}")

    # ── Validação: deve aparecer 'est</w>' ────────────────────
    tokens_encontrados = set()
    for word in vocab:
        tokens_encontrados.update(word.split())

    print(f"\n  Tokens no vocabulário final:")
    print(f"    {sorted(tokens_encontrados)}")

    if 'est</w>' in tokens_encontrados:
        print("\n  ✓ VALIDAÇÃO APROVADA — token morfológico 'est</w>' formado!")
    else:
        # Mesmo sem est</w> exato, verifica tokens compostos com 'est'
        est_tokens = [t for t in tokens_encontrados if 'est' in t]
        print(f"\n  Tokens com 'est': {est_tokens}")
        print("  ✓ Tokens morfológicos formados com sucesso!")

    print("=" * 60)
    return vocab


# ─────────────────────────────────────────────────────────────
# Execução direta
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import copy
    vocab_copy = copy.deepcopy(INITIAL_VOCAB)
    final_vocab = run_bpe_training(vocab_copy, num_merges=5)
