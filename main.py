"""
main.py  —  Laboratório 6: Construindo um Tokenizador BPE e Explorando o WordPiece
Disciplina : Tópicos em Inteligência Artificial – 2026.1
Professor  : Prof. Dimmy Magalhães
Instituição: iCEV
Aluno      : Sean Victor Machado de Moraes
GitHub     : https://github.com/SeanVictor

Pipeline:
    Tarefa 1 → get_stats: frequência de pares adjacentes
    Tarefa 2 → merge_vocab: loop de fusão BPE (K=5)
    Tarefa 3 → WordPiece com BERT multilingual (Hugging Face)
"""

import copy
from task1_get_stats   import get_stats, vocab as INITIAL_VOCAB
from task2_merge_vocab import run_bpe_training
from task3_wordpiece   import load_wordpiece_tokenizer, analyze_wordpiece, wordpiece_vs_bpe_comparison

print("=" * 60)
print("  Lab 06 — Tokenizador BPE e WordPiece")
print("  Aluno : Sean Victor Machado de Moraes")
print("  GitHub: SeanVictor")
print("=" * 60)


# ════════════════════════════════════════════════════════════
# TAREFA 1 — Motor de Frequências
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  TAREFA 1 — Motor de Frequências")
print("═" * 60)

pairs = get_stats(INITIAL_VOCAB)
sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)

print(f"\n  {'Par':<20} {'Contagem':>8}")
print(f"  {'-'*30}")
for pair, count in sorted_pairs:
    mark = " ← MÁXIMO" if pair == ('e', 's') else ""
    print(f"  {str(pair):<20} {count:>8}{mark}")

assert pairs[('e', 's')] == 9
print(f"\n  ✓ Validação OK — ('e','s') = 9")


# ════════════════════════════════════════════════════════════
# TAREFA 2 — Loop de Fusão
# ════════════════════════════════════════════════════════════
vocab_copy  = copy.deepcopy(INITIAL_VOCAB)
final_vocab = run_bpe_training(vocab_copy, num_merges=5)


# ════════════════════════════════════════════════════════════
# TAREFA 3 — WordPiece
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  TAREFA 3 — Integração Industrial e WordPiece")
print("═" * 60)

tokenizer = load_wordpiece_tokenizer()

frase_teste = (
    "Os hiper-parâmetros do transformer são "
    "inconstitucionalmente difíceis de ajustar."
)

tokens = analyze_wordpiece(tokenizer, frase_teste)
wordpiece_vs_bpe_comparison()


# ════════════════════════════════════════════════════════════
# Resumo final
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ✓ Lab 06 — Pipeline completo executado com sucesso!")
print("  Aluno: Sean Victor Machado de Moraes | GitHub: SeanVictor")
print("=" * 60)
