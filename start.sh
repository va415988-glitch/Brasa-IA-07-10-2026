#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${PROJECT_DIR}/runtime/target}"

# Lê somente configurações permitidas, sem executar o .env como shell.
# Variáveis já definidas no ambiente de quem iniciou o script têm prioridade.
load_local_env() {
  local env_file="${PROJECT_DIR}/.env"
  local line key value first_char last_char line_number=0

  [[ -f "$env_file" ]] || return 0
  if [[ ! -r "$env_file" ]]; then
    echo "Erro: não consigo ler ${env_file}. Confira as permissões do arquivo." >&2
    return 1
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    ((line_number += 1))
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    if [[ "$line" == export[[:space:]]* ]]; then
      line="${line#export}"
    fi
    if [[ "$line" != *=* ]]; then
      echo "Aviso: linha ${line_number} de .env ignorada (use CHAVE=VALOR)." >&2
      continue
    fi

    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    case "$key" in
      IA_LOCAL_BRAVE_SEARCH_API_KEY|IA_LOCAL_BRAVE_ALLOW_STORAGE) ;;
      # Núcleo de linguagem local (ver Documentacoes/NUCLEO_LINGUAGEM_LOCAL.md).
      # IA_LOCAL_CORE_ARGS fica de fora de propósito: argumentos livres para um
      # binário só valem vindos do ambiente do processo, nunca de um arquivo.
      IA_LOCAL_CORE_MODEL|IA_LOCAL_CORE_MMPROJ|IA_LOCAL_CORE_BIN|IA_LOCAL_CORE_URL) ;;
      IA_LOCAL_CORE_PORT|IA_LOCAL_CORE_CTX|IA_LOCAL_CORE_THREADS|IA_LOCAL_CORE_MAX_TOKENS) ;;
      IA_LOCAL_CORE_HISTORY_CHARS|IA_LOCAL_CORE_TEMPERATURE|IA_LOCAL_CORE_TIMEOUT) ;;
      *) continue ;;
    esac

    if [[ ${#value} -ge 2 ]]; then
      first_char="${value:0:1}"
      last_char="${value: -1:1}"
      if [[ ( "$first_char" == "'" && "$last_char" == "'" ) || ( "$first_char" == '"' && "$last_char" == '"' ) ]]; then
        value="${value:1:${#value}-2}"
      fi
    fi

    [[ -v "$key" ]] && continue
    printf -v "$key" '%s' "$value"
    export "$key"
  done < "$env_file"
}

load_local_env

# A inferência diária usa o checkpoint neural desenvolvido pelo projeto e, quando
# IA_LOCAL_CORE_MODEL ou IA_LOCAL_CORE_URL está definido, um núcleo de linguagem
# local de pesos abertos (somente loopback). Provedores remotos continuam fora:
# configurações antigas do Colab não ativam nada externo neste iniciador.
unset BRASA_COLAB_URL BRASA_COLAB_TOKEN

if ! command -v cargo >/dev/null 2>&1; then
  echo "Erro: cargo não foi encontrado. Instale o Rust antes de iniciar o projeto." >&2
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Erro: .venv/bin/python não foi encontrado. Crie o ambiente Python do projeto." >&2
  exit 1
fi

# O AgentCore executa TypeScript diretamente. Algumas distribuições entregam
# Node 22 sem o suporte compilado para --experimental-strip-types.
NODE_CANDIDATES=()
if [[ -n "${IA_LOCAL_NODE:-}" ]]; then
  NODE_CANDIDATES+=("${IA_LOCAL_NODE}")
else
  if command -v node >/dev/null 2>&1; then
    NODE_CANDIDATES+=("$(command -v node)")
  fi
  for candidate in "$HOME"/.nvm/versions/node/*/bin/node; do
    [[ -x "$candidate" ]] && NODE_CANDIDATES+=("$candidate")
  done
fi
NODE_BIN=""
for candidate in "${NODE_CANDIDATES[@]}"; do
  if [[ -x "$candidate" ]] && "$candidate" --experimental-strip-types --input-type=module \
      -e "import {stripTypeScriptTypes} from 'node:module'; stripTypeScriptTypes('const ready: boolean = true')" >/dev/null 2>&1; then
    NODE_BIN="$candidate"
    break
  fi
done
if [[ -z "$NODE_BIN" ]]; then
  echo "Erro: o AgentCore precisa de um Node.js com suporte a --experimental-strip-types. Defina IA_LOCAL_NODE com o caminho do executável compatível." >&2
  exit 1
fi
export PATH="$(dirname "$NODE_BIN"):$PATH"

if ! command -v ss >/dev/null 2>&1; then
  echo "Erro: ss não foi encontrado; não posso identificar com segurança os serviços que ocupam as portas locais." >&2
  exit 1
fi

echo "Compilando o runtime Rust..."
cargo build --offline --manifest-path runtime/Cargo.toml --bin local_ai_runtime

RUNTIME="${CARGO_TARGET_DIR}/debug/local_ai_runtime"
[[ "$RUNTIME" == /* ]] || RUNTIME="${PROJECT_DIR}/${RUNTIME}"

listener_rows() {
  ss -H -ltnp "sport = :$1" || {
    echo "Erro: não consegui verificar a porta $1 com ss. Nenhum serviço adicional será encerrado." >&2
    return 1
  }
}

listener_pids() {
  local rows matches status
  rows="$(listener_rows "$1")" || return 1
  matches="$(grep -oE 'pid=[0-9]+' <<< "$rows")" || {
    status=$?
    [[ "$status" == 1 ]] && return 0
    return "$status"
  }
  printf '%s\n' "$matches" \
    | cut -d= -f2 \
    | sort -u
}

is_project_listener() {
  local port="$1"
  local pid="$2"
  local command_line

  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline")" || return 1

  case "$port" in
    3000)
      [[ "$command_line" == *"${RUNTIME}"* && "$command_line" == *"--web"* ]]
      ;;
    3101)
      [[ "$command_line" == *"${PROJECT_DIR}/python/model_server.py"* && "$command_line" == *"--port 3101"* ]]
      ;;
    3200)
      [[ "$command_line" == *"${PROJECT_DIR}/agent-core/src/server.ts"* && "$command_line" == *"--experimental-strip-types"* ]]
      ;;
    *)
      return 1
      ;;
  esac
}

check_project_listeners() {
  local port="$1"
  local rows pids pid
  rows="$(listener_rows "$port")" || return 1
  [[ -n "$rows" ]] || return 0
  pids="$(listener_pids "$port")" || return 1
  if [[ -z "$pids" ]]; then
    echo "Erro: a porta ${port} está ocupada, mas não consegui identificar o processo. Nada foi encerrado." >&2
    return 1
  fi
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    if ! is_project_listener "$port" "$pid"; then
      echo "Erro: a porta ${port} está ocupada por um processo que não pertence a esta cópia da IA Local (PID ${pid}). Nada foi encerrado." >&2
      return 1
    fi
  done <<< "$pids"
}

stop_project_listeners() {
  local port="$1"
  local pids pid pgid attempt rows
  pids="$(listener_pids "$port")" || return 1
  [[ -n "$pids" ]] || return 0

  while read -r pid; do
    [[ -z "$pid" ]] && continue
    [[ -d "/proc/${pid}" ]] || continue
    if ! is_project_listener "$port" "$pid"; then
      echo "Erro: o processo na porta ${port} mudou durante a substituição (PID ${pid}); não vou encerrá-lo." >&2
      return 1
    fi

    echo "Encerrando serviço anterior da IA Local na porta ${port} (PID ${pid})."
    if [[ "$port" == "3000" ]]; then
      pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
      if [[ -n "$pgid" && "$pgid" == "$pid" && "$pgid" != "$$" ]]; then
        kill -TERM -- "-${pgid}" 2>/dev/null || true
      else
        kill -TERM "$pid" 2>/dev/null || true
      fi
    else
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done <<< "$pids"

  for attempt in {1..50}; do
    rows="$(listener_rows "$port")" || return 1
    [[ -z "$rows" ]] && return 0
    sleep 0.1
  done

  pids="$(listener_pids "$port")" || return 1
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    [[ -d "/proc/${pid}" ]] || continue
    if ! is_project_listener "$port" "$pid"; then
      echo "Erro: surgiu outro processo na porta ${port} durante a substituição; nada mais será encerrado." >&2
      return 1
    fi
    echo "Forçando o encerramento do serviço anterior na porta ${port} (PID ${pid})."
    kill -KILL "$pid" 2>/dev/null || true
  done <<< "$pids"

  for attempt in {1..20}; do
    rows="$(listener_rows "$port")" || return 1
    [[ -z "$rows" ]] && return 0
    sleep 0.1
  done
  echo "Erro: a porta ${port} continuou ocupada após encerrar o serviço anterior da IA Local." >&2
  return 1
}

replace_existing_runtime() {
  local port
  for port in 3000 3101 3200; do
    check_project_listeners "$port" || return 1
  done
  for port in 3000 3101 3200; do
    stop_project_listeners "$port" || return 1
  done
}

CHECKPOINT_PATH="$("${PROJECT_DIR}/.venv/bin/python" - <<'PY'
import sys
sys.path.insert(0, "python")
from active_checkpoint import selected_checkpoint
print(selected_checkpoint())
PY
)"
export IA_LOCAL_CHECKPOINT="${CHECKPOINT_PATH}"
CHECKPOINT_CONTEXT_INFO="$("${PROJECT_DIR}/.venv/bin/python" - "$CHECKPOINT_PATH" <<'PY'
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, "python")
from checkpoint_io import load_checkpoint
from model_server import ModelService
import torch

checkpoint = load_checkpoint(sys.argv[1])
if not isinstance(checkpoint.get("config"), dict) or not checkpoint.get("state_dict"):
    raise ValueError("Checkpoint sem configuração ou pesos válidos; os serviços existentes foram preservados.")
# Reutiliza o carregador real, inclusive a origem e as posições do contexto ampliado,
# sem construir adaptadores, escrever traces ou iniciar um servidor.
service = ModelService.__new__(ModelService)
service.checkpoint_path = sys.argv[1]
service.local_model = service.local_tokenizer = service.local_config = service.local_model_error = None
service._load_local_model()
if service.local_model_error or service.local_model is None or service.local_tokenizer is None:
    raise ValueError("Modelo não está pronto; os serviços existentes foram preservados: " + str(service.local_model_error))
config = service.local_config
tokenizer_path = Path(config.get("tokenizer_path") or "model/tokenizer.json")
tokenizer = service.local_tokenizer
expected = checkpoint.get("tokenizer_sha256")
if expected and hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != expected:
    raise ValueError("Tokenizer não corresponde ao hash do checkpoint.")
vocab_size = int(config["vocab_size"])
ids = list(tokenizer.vocab.values()) + list(tokenizer.special_tokens.values())
if not ids or any(type(token_id) is not int or not 0 <= token_id < vocab_size for token_id in ids):
    raise ValueError("Tokenizer incompatível com o vocabulário dos pesos.")
if any(bytes([value]).hex() not in tokenizer.vocab for value in range(256)):
    raise ValueError("Tokenizer não cobre todos os bytes de entrada.")
probe = tokenizer.encode_fast("Olá", add_bos=True)
with torch.inference_mode():
    logits = service.local_model(torch.tensor([probe], dtype=torch.long))
if logits.shape[-1] != vocab_size or not torch.isfinite(logits).all():
    raise ValueError("Forward de prontidão do modelo falhou.")
runtime = config.get("runtime_context_tokens") or config["context_length"]
trained = config.get("training_context_length", config["context_length"])
print(f"execução {runtime} tokens; treino {trained} tokens; pesos, tokenizer e forward verificados")
PY
)"
echo "Iniciando IA Local do Zero em http://127.0.0.1:3000"
echo "Node do AgentCore: ${NODE_BIN}"
echo "Checkpoint próprio: ${CHECKPOINT_PATH} · contexto: ${CHECKPOINT_CONTEXT_INFO} · elegibilidade semântica exige avaliação separada"
if [[ -n "${IA_LOCAL_BRAVE_SEARCH_API_KEY:-}" ]]; then
  echo "Busca web: Brave LLM Context habilitada (trechos transitórios; chave mantida no runtime)"
  BRAVE_STORAGE_SETTING="${IA_LOCAL_BRAVE_ALLOW_STORAGE:-false}"
  BRAVE_STORAGE_SETTING="${BRAVE_STORAGE_SETTING,,}"
  if [[ "$BRAVE_STORAGE_SETTING" =~ ^(1|true|yes)$ ]]; then
    echo "Acervo Brave: armazenamento habilitado por configuração; confirme direitos explícitos no plano contratado"
  else
    echo "Acervo Brave: armazenamento desabilitado (padrão seguro)"
  fi
else
  echo "Busca web: DuckDuckGo local scraper (adicione IA_LOCAL_BRAVE_SEARCH_API_KEY para usar Brave LLM Context)"
fi
echo "Pressione Ctrl+C para encerrar o runtime e o worker do modelo."

replace_existing_runtime

setsid "$RUNTIME" --web &
RUNTIME_PID=$!

cleanup() {
  trap - INT TERM EXIT
  kill -TERM -- "-${RUNTIME_PID}" 2>/dev/null || true
  wait "$RUNTIME_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

wait "$RUNTIME_PID"
