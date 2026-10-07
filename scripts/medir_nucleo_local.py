#!/usr/bin/env python3
"""Mede, na sua máquina, quão rápido o núcleo de linguagem local responde.

Fala direto com o llama-server, sem passar pelo chat do projeto, para separar a
velocidade do modelo da lógica do produto. Em CPU o custo dominante é ler o
prompt (prefill), então são três cenários:

  1. pergunta curta, a frio;
  2. conversa longa, a frio (leitura de todo o histórico);
  3. a mesma conversa com um seguimento: mostra o ganho do cache de prefixo.

Uso (o servidor precisa estar de pé; veja Documentacoes/NUCLEO_LINGUAGEM_LOCAL.md):
  IA_LOCAL_CORE_URL=http://127.0.0.1:3102 python3 scripts/medir_nucleo_local.py
  python3 scripts/medir_nucleo_local.py --url http://127.0.0.1:8080 --history-chars 3500 --json medida.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from dialogue_api import CHECKPOINT_DIALOGUE_GUIDANCE, DialogueProviderError  # noqa: E402
from language_core import DEFAULT_PORT, CoreSettings, LocalCoreProvider, validate_core_url  # noqa: E402

SENTENCES = (
    "Estamos organizando um aplicativo para dividir as tarefas da casa entre quatro pessoas.",
    "Cada pessoa tem dias em que trabalha fora e dias em que fica em casa, e isso muda a divisão.",
    "Queremos que o aplicativo avise quando uma tarefa estiver atrasada, sem ficar chato.",
    "Também precisamos de uma forma simples de trocar tarefas entre duas pessoas.",
    "A ideia é que funcione no celular e também em um computador comum, sem instalar nada.",
    "Ainda não decidimos se os dados ficam no aparelho ou em um servidor que a gente mantém.",
)


def long_history(chars: int) -> list[dict]:
    """Conversa alternada de usuário e assistente com cerca de ``chars`` caracteres."""
    messages, size, index = [], 0, 0
    while size < chars:
        role = "user" if index % 2 == 0 else "assistant"
        text = SENTENCES[index % len(SENTENCES)] + (" Entendi, vamos considerar isso." if role == "assistant" else "")
        messages.append({"role": role, "content": text})
        size += len(text)
        index += 1
    if messages[-1]["role"] != "user":
        messages.pop()
    return messages


def measure(provider: LocalCoreProvider, messages: list[dict]) -> tuple[dict, str]:
    answer = provider.complete_stream(messages, lambda fragment: None)
    stats = dict(provider.last_stats)
    tokens = stats.get("completion_tokens")
    generation = stats.get("completion_tokens_per_second")
    first = stats.get("time_to_first_delta_s")
    if generation is None and tokens and first is not None and stats["elapsed_s"] > first:
        generation = round((tokens - 1) / (stats["elapsed_s"] - first), 2)
    stats["completion_tokens_per_second"] = generation
    return stats, answer


def fmt(value, unit="", digits=1) -> str:
    return "?" if value is None else f"{value:.{digits}f}{unit}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("IA_LOCAL_CORE_URL") or f"http://127.0.0.1:{DEFAULT_PORT}")
    parser.add_argument("--history-chars", type=int, default=3500, help="tamanho do histórico do cenário 2")
    parser.add_argument("--max-tokens", type=int, default=128, help="limite de geração por cenário")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json", type=Path, help="grava as medidas neste arquivo")
    args = parser.parse_args(argv)

    try:
        url = validate_core_url(args.url)
    except ValueError as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 2
    settings = CoreSettings(url=url, max_tokens=args.max_tokens, timeout=args.timeout,
                            history_chars=max(6000, args.history_chars + 2500))
    provider = LocalCoreProvider(settings)
    if not provider.configured:
        print(f"O servidor em {url} não respondeu ao /health. Inicie o núcleo (IA_LOCAL_CORE_MODEL no start.sh, "
              "ou o llama-server à mão) e espere terminar de carregar.", file=sys.stderr)
        return 2

    system = {"role": "system", "content": CHECKPOINT_DIALOGUE_GUIDANCE}
    history = long_history(args.history_chars)
    try:
        provider.complete([system, {"role": "user", "content": "Diga oi."}])  # aquecimento, não é medido
        rows = []
        stats, _ = measure(provider, [system, {"role": "user", "content": "Em duas frases: o que é uma API?"}])
        rows.append(("1 pergunta curta", stats))
        stats, reply = measure(provider, [system, *history])
        rows.append((f"2 conversa de ~{sum(len(m['content']) for m in history)} caracteres", stats))
        followup = [system, *history, {"role": "assistant", "content": reply}, {"role": "user", "content": "E qual seria o primeiro passo?"}]
        stats, _ = measure(provider, followup)
        rows.append(("3 mesmo histórico + seguimento", stats))
    except (DialogueProviderError, OSError) as error:
        print(f"Falha durante a medição: {error}", file=sys.stderr)
        return 2

    print(f"\nNúcleo: {provider.model}\n")
    print(f"{'cenário':42} {'1º texto':>9} {'total':>8} {'tokens':>7} {'geração':>10} {'leitura do prompt':>19}")
    for label, stats in rows:
        print(f"{label:42} {fmt(stats.get('time_to_first_delta_s'), ' s'):>9} {fmt(stats.get('elapsed_s'), ' s'):>8} "
              f"{str(stats.get('completion_tokens') or '?'):>7} {fmt(stats.get('completion_tokens_per_second'), ' tok/s'):>10} "
              f"{fmt(stats.get('prompt_tokens_per_second'), ' tok/s'):>19}")

    generation = rows[1][1].get("completion_tokens_per_second")
    cold, warm = rows[1][1].get("time_to_first_delta_s"), rows[2][1].get("time_to_first_delta_s")
    print("\nComo ler (faixas de referência, não garantias):")
    if generation is not None:
        verdict = ("folga para voz" if generation >= 15 else
                   "bom para texto com streaming; na voz a latência será perceptível" if generation >= 6 else
                   "lento: considere um modelo menor ou menos contexto")
        print(f"  geração a {generation:.1f} tok/s: {verdict}.")
    if cold and warm:
        if cold < 1.0:
            print("  cache de prefixo: o prompt a frio foi rápido demais para concluir; aumente --history-chars.")
        else:
            print("  cache de prefixo: " + ("funcionando (o seguimento começou bem mais rápido)." if warm < 0.5 * cold else
                  "não está ajudando; confira o contexto (-c) e se o histórico coube no orçamento."))
    if args.json:
        args.json.write_text(json.dumps({"model": provider.model, "url": url, "scenarios": [
            {"scenario": label, **stats} for label, stats in rows]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nMedidas gravadas em {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
