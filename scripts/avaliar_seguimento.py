#!/usr/bin/env python3
"""Mede se o chat entende um segundo pedido que depende do primeiro.

Usa conversas NOVAS (tests/data/avaliacao_seguimento.json), não as da bancada
fixa de 50 perguntas: ali as respostas de continuidade vêm de regras escritas
para aquelas mesmas frases, então o resultado não prova nada. Aqui cada caso
exige um termo que só existe nos turnos anteriores, e o relatório diz de onde
veio cada resposta (gerador ou regras prontas).

Rotas:
  chat      POST /api/chat                 mesma chamada da bancada de 50
  dialogue  POST /api/v1/dialogue/turn     contrato agent-dialogue/v1 (aceita --provider)
  agent     POST /api/v1/agent/pursue      caminho da interface para a maioria das mensagens

Exemplos:
  python3 scripts/avaliar_seguimento.py --route chat
  python3 scripts/avaliar_seguimento.py --route dialogue --provider local-core --show
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avaliar_dialogo_50 import normalize, request_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "tests/data/avaliacao_seguimento.json"
DEFAULT_OUTPUT_DIR = ROOT / "logs/dialogue-evaluations"
RUN_SCHEMA = "follow-up-evaluation-run/v1"
ROUTES = ("chat", "dialogue", "agent")

# Respostas que a própria interface trata como "não respondeu".
NON_ANSWERS = (
    "estou acompanhando. pode me contar um pouco mais",
    "nao consegui produzir uma resposta confiavel",
    "a tarefa nao foi concluida",
    "a qual item, biblioteca ou opcao voce esta se referindo",
    "como modelo de linguagem",
    "como uma ia",
)

# Backends que respondem por regras ou memória curada, não por um gerador.
RULE_BACKENDS = frozenset({
    "local-conversation-rules", "local-conversation", "session-memory", "curated-memory", "local-knowledge",
    "quality-gate", "diagnostic-reasoning-fallback", "diagnostic-failure-report", "local-planning", "local-task",
    "deterministic-reasoning", "requirements-gate", "local-capabilities", "cognitive-router", "tool-router",
    "static-analysis", "action-proposal", "execution-engine", "local-product-planning", "proactive-learning",
})


def term_matches(normalized_text: str, term: str) -> bool:
    """Compara sem acentos nem maiúsculas. Termo terminado em ``*`` casa como prefixo."""
    term = normalize(term).strip()
    prefix = term.endswith("*")
    term = term.rstrip("*").strip()
    if not term:
        return False
    pattern = re.escape(term).replace(r"\ ", r"\s+")
    if term[0].isalnum():
        pattern = r"(?<![a-z0-9])" + pattern
    if not prefix and term[-1].isalnum():
        pattern += r"(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def validate_suite(suite: object) -> list[dict]:
    if not isinstance(suite, dict) or not isinstance(suite.get("cases"), list) or not suite["cases"]:
        raise ValueError("arquivo inválido: esperava-se um objeto com a lista cases")
    ids = set()
    for index, case in enumerate(suite["cases"], start=1):
        if not isinstance(case, dict):
            raise ValueError(f"caso {index} não é um objeto")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError(f"id ausente ou repetido no caso {index}")
        ids.add(case_id)
        if not isinstance(case.get("kind"), str) or not case["kind"]:
            raise ValueError(f"{case_id}: falta o tipo (kind)")
        messages = case.get("messages")
        if not isinstance(messages, list) or len(messages) < 3 or len(messages) % 2 == 0:
            raise ValueError(f"{case_id}: messages precisa terminar em fala do usuário e ter ao menos um turno anterior")
        for position, message in enumerate(messages):
            expected = "user" if position % 2 == 0 else "assistant"
            if (not isinstance(message, dict) or message.get("role") != expected
                    or not isinstance(message.get("content"), str) or not message["content"].strip()):
                raise ValueError(f"{case_id}: a mensagem {position + 1} deveria ser do papel {expected}")
        if not isinstance(case.get("checks"), list) or not case["checks"]:
            raise ValueError(f"{case_id}: cada caso precisa de critérios")
        for check in case["checks"]:
            if (not isinstance(check, dict) or not isinstance(check.get("name"), str)
                    or not isinstance(check.get("any"), list) or not check["any"]
                    or not all(isinstance(term, str) and term.strip() for term in check["any"])):
                raise ValueError(f"{case_id}: cada critério precisa de name e any (lista de termos)")
        forbidden = case.get("forbidden", [])
        if not isinstance(forbidden, list) or not all(isinstance(term, str) for term in forbidden):
            raise ValueError(f"{case_id}: forbidden precisa ser uma lista de termos")
    return suite["cases"]


def build_request(route: str, case: dict, run_id: str, provider: str | None = None) -> tuple[str, dict]:
    messages = case["messages"]
    request_id = f"follow-up-{run_id}-{case['id']}"
    if route == "chat":
        return "/api/chat", {"request_id": request_id, "messages": messages}
    if route == "dialogue":
        body = {"schema": "agent-dialogue-request/v1", "request_id": request_id, "messages": messages}
        if provider:
            body["provider"] = provider
        return "/api/v1/dialogue/turn", body
    # Como a interface: o histórico enviado já inclui o pedido atual como última fala.
    return "/api/v1/agent/pursue", {
        "schema": "agent-request/v2", "prompt": messages[-1]["content"], "objective": "auto",
        "history": messages, "operationId": f"agent-core-follow-up-{run_id}-{case['id']}"[:110],
        "request_id": request_id, "conversation_id": request_id, "attachments": [],
    }


def extract(route: str, status: int, body: dict) -> dict:
    """Normaliza a resposta das três rotas em texto, sucesso e origem."""
    if route == "agent":
        report = body.get("report") if isinstance(body.get("report"), dict) else {}
        text = str(report.get("finalText") or "")
        awaiting = body.get("awaitingClarification") is True
        if awaiting and not text:
            text = "[pediu esclarecimento] " + json.dumps(body.get("understanding") or {}, ensure_ascii=False)[:400]
        ok = status == 200 and body.get("ok") is not False and bool(text.strip()) and not awaiting
        return {"text": text, "ok": ok, "backend": "agent-core", "problem": "pediu esclarecimento" if awaiting else None,
                "error": body.get("error")}
    text = str(body.get("text") or "")
    asked_for_tool = bool(body.get("tool_call") or body.get("tool_calls"))
    ok = status == 200 and body.get("ok") is True and bool(text.strip()) and not asked_for_tool
    return {"text": text, "ok": ok, "backend": body.get("backend"),
            "problem": "pediu uma ferramenta em vez de responder" if asked_for_tool else None,
            "error": body.get("error")}


def origin(backend: str | None) -> str:
    if not backend:
        return "desconhecida"
    if backend in RULE_BACKENDS:
        return "regras"
    if backend.startswith("dialogue-") or backend == "local-neural":
        return "gerador"
    return "outra"


def evaluate(case: dict, answer: str, response_ok: bool) -> dict:
    normalized = normalize(answer)
    checks = [{"name": check["name"], "passed": any(term_matches(normalized, term) for term in check["any"])}
              for check in case["checks"]]
    forbidden_terms = [*NON_ANSWERS, *case.get("forbidden", [])]
    forbidden_hits = [term for term in forbidden_terms if term_matches(normalized, term)]
    length = len(answer.strip())
    minimum, maximum = int(case.get("min_chars", 20)), case.get("max_chars")
    length_ok = length >= minimum and (maximum is None or length <= int(maximum))
    passed = response_ok and all(item["passed"] for item in checks) and not forbidden_hits and length_ok
    return {"passed": passed, "checks": checks, "forbidden_hits": forbidden_hits, "length_ok": length_ok, "chars": length}


def summarize(results: list[dict]) -> dict:
    by_kind: dict[str, list[bool]] = defaultdict(list)
    for row in results:
        by_kind[row["kind"]].append(row["passed"])
    origins = Counter(row["origin"] for row in results)
    total = len(results)
    passed = sum(row["passed"] for row in results)
    return {
        "total": total, "passed": passed, "pass_rate": round(100.0 * passed / total, 1) if total else 0.0,
        "by_kind": {kind: {"passed": sum(values), "total": len(values)} for kind, values in sorted(by_kind.items())},
        "origins": dict(origins),
        "rules_share": round(100.0 * origins.get("regras", 0) / total, 1) if total else 0.0,
        "elapsed_s_median": round(sorted(row["elapsed_s"] for row in results)[total // 2], 2) if total else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:3000", help="runtime local (padrão: %(default)s)")
    parser.add_argument("--route", choices=ROUTES, default="chat")
    parser.add_argument("--provider", choices=("local", "own-checkpoint", "local-core"),
                        help="só na rota dialogue: escolhe o gerador")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--only", nargs="+", metavar="ID", help="roda apenas estes casos")
    parser.add_argument("--timeout", type=float, default=300.0, help="segundos por pedido (CPU é lenta)")
    parser.add_argument("--pause-seconds", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true", help="imprime cada resposta")
    parser.add_argument("--fail-under", type=float, help="sai com código 1 se a taxa de acerto ficar abaixo (em %%)")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.provider and args.route != "dialogue":
        parser.error("--provider só vale na rota dialogue")

    try:
        cases = validate_suite(json.loads(args.suite.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Erro ao carregar a bateria: {error}", file=sys.stderr)
        return 2
    if args.only:
        unknown = sorted(set(args.only) - {case["id"] for case in cases})
        if unknown:
            print(f"Casos desconhecidos: {', '.join(unknown)}", file=sys.stderr)
            return 2
        cases = [case for case in cases if case["id"] in set(args.only)]
    if args.validate_only:
        print(f"Bateria válida: {len(cases)} casos em {len({case['kind'] for case in cases})} tipos.")
        return 0

    base_url = args.base_url.rstrip("/")
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    print(f"Rota {args.route} em {base_url}" + (f" (gerador: {args.provider})" if args.provider else "")
          + f" · {len(cases)} casos em série", flush=True)

    results = []
    for index, case in enumerate(cases, start=1):
        path, payload = build_request(args.route, case, run_id, args.provider)
        started = time.monotonic()
        try:
            status, body = request_json(base_url + path, method="POST", payload=payload, timeout=args.timeout)
        except ConnectionError as error:
            print(f"Runtime indisponível em {base_url}: {error}", file=sys.stderr)
            return 2
        elapsed = time.monotonic() - started
        got = extract(args.route, status, body)
        verdict = evaluate(case, got["text"], got["ok"])
        row = {"id": case["id"], "kind": case["kind"], "route": args.route, "http_status": status,
               "backend": got["backend"], "origin": origin(got["backend"]), "problem": got["problem"],
               "error": got["error"], "elapsed_s": round(elapsed, 2), "answer": got["text"],
               "last_user_message": case["messages"][-1]["content"], **verdict}
        results.append(row)
        label = "ok   " if row["passed"] else "FALHA"
        detail = got["problem"] or (", ".join(c["name"] for c in verdict["checks"] if not c["passed"]) or
                                    ("termo proibido: " + ", ".join(verdict["forbidden_hits"]) if verdict["forbidden_hits"]
                                     else "" if verdict["length_ok"] else "tamanho fora do limite"))
        print(f"[{index:02d}/{len(cases)}] {label} {case['id']} · {elapsed:.1f}s · {row['backend'] or 'sem backend'}"
              + (f" · {detail}" if not row["passed"] and detail else ""), flush=True)
        if args.show:
            print("        pergunta: " + case["messages"][-1]["content"])
            print("        resposta: " + (got["text"].strip().replace("\n", "\n                  ")[:600] or "(vazia)"))
        if args.pause_seconds and index < len(cases):
            time.sleep(args.pause_seconds)

    summary = summarize(results)
    print(f"\nAcertos: {summary['passed']}/{summary['total']} ({summary['pass_rate']}%)")
    for kind, value in summary["by_kind"].items():
        print(f"  {kind:18} {value['passed']}/{value['total']}")
    print("Origem das respostas: " + " · ".join(f"{name} {count}" for name, count in sorted(summary["origins"].items())))
    if summary["rules_share"] >= 50:
        print("AVISO: a maioria das respostas veio de regras prontas, não de um gerador. "
              "Um acerto vindo de regra não prova que o seguimento foi entendido.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"seguimento-{args.route}-{run_id}.json"
    report_path.write_text(json.dumps({
        "schema": RUN_SCHEMA, "run_id": run_id, "created_at": now.isoformat(), "base_url": base_url,
        "route": args.route, "provider": args.provider, "suite": str(args.suite), "summary": summary,
        "results": results,
        "note": "Critérios automáticos amplos; leia as respostas. Não mede nuance nem naturalidade.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Relatório: {report_path}")
    return 1 if args.fail_under is not None and summary["pass_rate"] < args.fail_under else 0


if __name__ == "__main__":
    sys.exit(main())
