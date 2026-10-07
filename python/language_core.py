"""Núcleo de linguagem local de pesos abertos, servido pelo ``llama-server`` (llama.cpp).

O núcleo é um provedor do contrato ``agent-dialogue/v1``: recebe mensagens
``{role, content}`` e devolve texto. Ele só conversa com um servidor em loopback,
então nenhuma conversa sai da máquina. O checkpoint próprio continua na cadeia
como reserva (ver ``DialogueAPI``).

Desligado por padrão. Só existe quando uma destas variáveis está definida:

- ``IA_LOCAL_CORE_MODEL``: caminho de um GGUF; o worker inicia o ``llama-server``;
- ``IA_LOCAL_CORE_URL``: servidor que você já mantém (somente http em loopback).

O servidor aplica o template de chat do próprio GGUF, que é o que dá ao modelo o
formato multi-turno em que ele foi treinado. Este módulo cuida do que é do
projeto: higiene dos turnos, orçamento de contexto para CPU, prefixo de prompt
estável (para o cache de KV do servidor) e um portão de qualidade próprio.
"""
from __future__ import annotations

import atexit
import ctypes
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dialogue_api import CHECKPOINT_DIALOGUE_GUIDANCE, DialogueProviderError

ROOT = Path(__file__).resolve().parents[1]
PROVIDER_NAME = 'local-core'
DEFAULT_PORT = 3102
LOOPBACK_HOSTS = frozenset({'127.0.0.1', 'localhost', '::1'})


# --------------------------------------------------------------------------- configuração

def validate_core_url(url):
    """Aceita só HTTP em loopback e sem credenciais: o núcleo nunca é um serviço remoto."""
    parts = urlsplit(str(url or '').strip())
    if parts.scheme != 'http' or parts.hostname not in LOOPBACK_HOSTS or parts.username or parts.password:
        raise ValueError('IA_LOCAL_CORE_URL deve ser http://127.0.0.1:<porta> (somente loopback, sem credenciais)')
    host = f'[{parts.hostname}]' if ':' in parts.hostname else parts.hostname
    return f'http://{host}:{parts.port or 80}'


def _bounded_int(text, default, minimum, maximum):
    try:
        return max(minimum, min(maximum, int(str(text).strip())))
    except ValueError:
        return default


def _bounded_float(text, default, minimum, maximum):
    try:
        value = float(str(text).strip())
    except ValueError:
        return default
    return max(minimum, min(maximum, value)) if value == value else default


@dataclass(frozen=True)
class CoreSettings:
    url: str
    model: str | None = None        # GGUF que o supervisor inicia; None = servidor já ativo
    mmproj: str | None = None       # projetor multimodal (visão e áudio), opcional
    binary: str | None = None
    port: int = DEFAULT_PORT
    context: int = 4096
    threads: int | None = None      # None deixa o llama-server escolher
    max_tokens: int = 512
    history_chars: int = 6000       # orçamento do histórico; o prefill em CPU é o custo dominante
    temperature: float = 0.7
    timeout: float = 180.0
    extra_args: tuple[str, ...] = ()

    @property
    def label(self):
        return Path(self.model).name if self.model else self.url

    @classmethod
    def from_environment(cls, environ=None):
        """Lê ``IA_LOCAL_CORE_*``. Devolve None quando o núcleo não está configurado."""
        environ = os.environ if environ is None else environ

        def get(key):
            return str(environ.get(key, '') or '').strip()

        url, model = get('IA_LOCAL_CORE_URL'), get('IA_LOCAL_CORE_MODEL')
        if not url and not model:
            return None
        port = _bounded_int(get('IA_LOCAL_CORE_PORT'), DEFAULT_PORT, 1024, 65535)
        return cls(
            url=validate_core_url(url) if url else f'http://127.0.0.1:{port}',
            model=None if url else model,
            mmproj=get('IA_LOCAL_CORE_MMPROJ') or None,
            binary=get('IA_LOCAL_CORE_BIN') or None,
            port=port,
            context=_bounded_int(get('IA_LOCAL_CORE_CTX'), 4096, 512, 131072),
            threads=_bounded_int(get('IA_LOCAL_CORE_THREADS'), 0, 0, 256) or None,
            max_tokens=_bounded_int(get('IA_LOCAL_CORE_MAX_TOKENS'), 512, 16, 8192),
            history_chars=_bounded_int(get('IA_LOCAL_CORE_HISTORY_CHARS'), 6000, 500, 200000),
            temperature=_bounded_float(get('IA_LOCAL_CORE_TEMPERATURE'), 0.7, 0.0, 2.0),
            timeout=_bounded_float(get('IA_LOCAL_CORE_TIMEOUT'), 180.0, 5.0, 1800.0),
            # Argumentos extras só vêm do ambiente do processo, nunca do arquivo .env.
            extra_args=tuple(shlex.split(str(environ.get('IA_LOCAL_CORE_ARGS', '') or ''))),
        )


# --------------------------------------------------------------------------- preparo dos turnos

_ROLES = frozenset({'system', 'user', 'assistant'})


def normalize_turns(messages):
    """Deixa os turnos aceitáveis por templates de chat estritos (como o do Gemma).

    Esses templates exigem alternância user/assistant começando por user; uma
    sequência fora disso faz o servidor recusar o pedido. Junta falas seguidas do
    mesmo papel, descarta assistente no início e garante que o último turno seja do
    usuário (a resposta nasce de uma fala dele).
    """
    system, turns = [], []
    for item in messages or []:
        if not isinstance(item, dict) or item.get('role') not in _ROLES:
            continue
        content = str(item.get('content') or '').strip()
        if not content:
            continue
        if item['role'] == 'system':
            system.append(content)
        elif turns and turns[-1]['role'] == item['role']:
            turns[-1]['content'] += '\n\n' + content
        else:
            turns.append({'role': item['role'], 'content': content})
    while turns and turns[0]['role'] != 'user':
        turns.pop(0)
    while turns and turns[-1]['role'] != 'user':
        turns.pop()
    head = [{'role': 'system', 'content': '\n\n'.join(system)}] if system else []
    return head + turns


def _head_tail(text, limit):
    if len(text) <= limit:
        return text
    marker = '\n[trecho intermediário omitido]\n'
    room = max(0, limit - len(marker))
    head = room * 2 // 5
    return text[:head] + marker + text[len(text) - (room - head):]


def fit_messages(messages, budget_chars):
    """Mantém as falas mais recentes dentro do orçamento; a fala atual nunca é descartada.

    Em CPU o tempo de leitura do prompt cresce com o histórico, então o orçamento
    é pequeno por padrão. O system fica de fora da conta (é um prefixo estável).
    """
    normalized = normalize_turns(messages)
    head = [item for item in normalized if item['role'] == 'system']
    turns = [item for item in normalized if item['role'] != 'system']
    kept, used = [], 0
    for item in reversed(turns):
        size = len(item['content'])
        if not kept:
            content = _head_tail(item['content'], budget_chars)
            kept.append({**item, 'content': content})
            used += len(content)
        elif used + size <= budget_chars:
            kept.append(item)
            used += size
        else:
            break
    kept.reverse()
    while kept and kept[0]['role'] != 'user':
        kept.pop(0)
    return head + kept


def stabilize_system(messages):
    """Mantém o prefixo do prompt idêntico entre turnos.

    O servidor só reaproveita o cache de KV enquanto o início do prompt não muda. O
    guia-base é constante, mas a orientação por pedido (``system_context``) varia a
    cada pergunta; ela passa para o fim da última fala do usuário.
    """
    if not messages or messages[0].get('role') != 'system':
        return list(messages)
    system = str(messages[0].get('content') or '')
    if not system.startswith(CHECKPOINT_DIALOGUE_GUIDANCE):
        return list(messages)
    variable = system[len(CHECKPOINT_DIALOGUE_GUIDANCE):].strip()
    stable = {'role': 'system', 'content': CHECKPOINT_DIALOGUE_GUIDANCE}
    rest = [dict(item) for item in messages[1:]]
    if not variable:
        return [stable, *rest]
    for item in reversed(rest):
        if item.get('role') == 'user':
            item['content'] = f"{item['content']}\n\n[Orientação para esta resposta: {variable}]"
            return [stable, *rest]
    return list(messages)


# --------------------------------------------------------------------------- saída do modelo

_END_MARKERS = re.compile(r'(?:\s*(?:<end_of_turn>|<eos>|<\|im_end\|>|<\|eot_id\|>|<\|endoftext\|>))+\s*$')
_LEAK = re.compile(
    r'<\|(?:im_start|im_end|endoftext|eot_id|start_header_id|end_header_id|system|user|assistant)\|>'
    r'|<(?:start|end)_of_turn>|<bos>|<eos>'
)


def clean_answer(text):
    """Remove raciocínio interno (``<think>``) e marcadores de fim de turno soltos."""
    text = re.sub(r'<think>.*?</think>', '', str(text or ''), flags=re.S)
    text = text.split('<think>', 1)[0]  # raciocínio sem fechamento: a resposta não chegou
    return _END_MARKERS.sub('', text).strip()


def core_answer_gate(answer):
    """Portão de segurança e degeneração para respostas de um modelo capaz.

    Não exige que a resposta repita palavras da pergunta: em um seguimento
    ("Web, com login.") a resposta correta quase nunca as repete, e esse critério
    lexical existe para o checkpoint minúsculo, não para um modelo de instruções.
    """
    text = str(answer or '').strip()
    if len(text) < 2:
        return False, 'empty-answer'
    if any(ord(char) < 32 and char not in '\n\r\t' for char in text):
        return False, 'control-character'
    if _LEAK.search(text):
        return False, 'prompt-leak'
    words = text.split()
    if len(words) >= 24:
        grams = Counter(tuple(words[index:index + 6]) for index in range(len(words) - 5))
        if grams.most_common(1)[0][1] >= 4:
            return False, 'repetition-loop'
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 12]
    if len(lines) >= 6 and Counter(lines).most_common(1)[0][1] >= 4:
        return False, 'repetition-loop'
    return True, 'accepted'


def _partial_suffix(buffer, tag):
    """Maior sufixo de ``buffer`` que ainda pode virar o começo de ``tag``."""
    for size in range(min(len(tag) - 1, len(buffer)), 0, -1):
        if buffer.endswith(tag[:size]):
            return size
    return 0


class _ThinkFilter:
    """Esconde ``<think>...</think>`` num fluxo, mesmo com a tag partida entre fragmentos."""

    OPEN, CLOSE = '<think>', '</think>'

    def __init__(self):
        self.buffer = ''
        self.hidden = False

    def feed(self, text):
        self.buffer += text
        visible = []
        while True:
            tag = self.CLOSE if self.hidden else self.OPEN
            index = self.buffer.find(tag)
            if index < 0:
                break
            if not self.hidden:
                visible.append(self.buffer[:index])
            self.buffer = self.buffer[index + len(tag):]
            self.hidden = not self.hidden
        keep = _partial_suffix(self.buffer, self.CLOSE if self.hidden else self.OPEN)
        emit_until = len(self.buffer) - keep
        if not self.hidden:
            visible.append(self.buffer[:emit_until])
        self.buffer = self.buffer[emit_until:]
        return ''.join(visible)

    def flush(self):
        tail = '' if self.hidden else self.buffer
        self.buffer = ''
        return tail


def _sse_events(response):
    """Produz o JSON de cada linha ``data:`` de uma resposta em streaming (SSE)."""
    for raw in response:
        line = raw.decode('utf-8', 'replace').strip()
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            return
        try:
            yield json.loads(data)
        except ValueError:
            continue


# --------------------------------------------------------------------------- provedor

class LocalCoreProvider:
    """Provedor ``local-core``: conversa com um ``llama-server`` em loopback."""

    name = PROVIDER_NAME

    def __init__(self, settings, *, opener=None, supervisor=None, clock=time.monotonic):
        self.settings = settings
        self.supervisor = supervisor
        # Loopback não deve passar por proxy do ambiente.
        self._opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._clock = clock
        self._health = (float('-inf'), False)
        self.last_stats = {}

    @property
    def model(self):
        return self.settings.label

    @property
    def configured(self):
        """Verdadeiro quando o servidor responde ao /health (resultado em cache por poucos segundos)."""
        now = self._clock()
        checked_at, healthy = self._health
        if now - checked_at < (5.0 if healthy else 2.0):
            return healthy
        healthy = self._probe()
        self._health = (now, healthy)
        return healthy

    def _probe(self):
        try:
            with self._opener.open(self.settings.url + '/health', timeout=1.5) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            return False

    def state(self):
        if self.configured:
            return 'ready'
        if self.supervisor is not None:
            return self.supervisor.state()
        return 'unreachable'

    def status(self):
        return {'id': PROVIDER_NAME, 'configured': self.configured, 'model': self.model,
                'server': self.settings.url, 'state': self.state(), 'managed': self.supervisor is not None}

    def answer_gate(self, answer):
        return core_answer_gate(answer)

    # -- geração

    def _payload(self, messages, stream):
        prepared = stabilize_system(fit_messages(messages, self.settings.history_chars))
        if not any(item['role'] == 'user' for item in prepared):
            raise DialogueProviderError('núcleo local: o pedido não tem fala do usuário')
        payload = {
            'messages': prepared,
            'stream': stream,
            'max_tokens': self.settings.max_tokens,
            'temperature': self.settings.temperature,
            'top_p': 0.9,
            'cache_prompt': True,
        }
        if stream:
            payload['stream_options'] = {'include_usage': True}
        return payload

    def _open(self, payload):
        request = urllib.request.Request(
            self.settings.url + '/v1/chat/completions',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            return self._opener.open(request, timeout=self.settings.timeout)
        except urllib.error.HTTPError as error:
            raise DialogueProviderError(f'núcleo local respondeu HTTP {error.code}: {_error_detail(error)}') from error
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            raise DialogueProviderError(f'núcleo local indisponível: {error}') from error

    def _record(self, body, started, finish_reason, first_delta=None):
        timings = body.get('timings') if isinstance(body, dict) and isinstance(body.get('timings'), dict) else {}
        usage = body.get('usage') if isinstance(body, dict) and isinstance(body.get('usage'), dict) else {}
        now = self._clock()
        self.last_stats = {
            'elapsed_s': round(now - started, 3),
            'time_to_first_delta_s': round(first_delta - started, 3) if first_delta is not None else None,
            'finish_reason': finish_reason,
            'prompt_tokens': usage.get('prompt_tokens', timings.get('prompt_n')),
            'completion_tokens': usage.get('completion_tokens', timings.get('predicted_n')),
            'prompt_tokens_per_second': timings.get('prompt_per_second'),
            'completion_tokens_per_second': timings.get('predicted_per_second'),
        }

    def complete(self, messages):
        started = self._clock()
        with self._open(self._payload(messages, False)) as response:
            try:
                body = json.loads(response.read().decode('utf-8'))
            except ValueError as error:
                raise DialogueProviderError('núcleo local devolveu uma resposta inválida') from error
        choice = (body.get('choices') or [{}])[0] if isinstance(body, dict) else {}
        self._record(body, started, choice.get('finish_reason'))
        return clean_answer((choice.get('message') or {}).get('content') or '')

    def complete_stream(self, messages, on_delta):
        started, first_delta = self._clock(), None
        hidden, parts, finish_reason, last = _ThinkFilter(), [], None, {}
        with self._open(self._payload(messages, True)) as response:
            for event in _sse_events(response):
                if event.get('timings') or event.get('usage'):
                    last = event
                choice = (event.get('choices') or [{}])[0]
                finish_reason = choice.get('finish_reason') or finish_reason
                visible = hidden.feed((choice.get('delta') or {}).get('content') or '')
                if visible:
                    first_delta = first_delta if first_delta is not None else self._clock()
                    parts.append(visible)
                    on_delta(visible)
        tail = hidden.flush()
        if tail:
            parts.append(tail)
            on_delta(tail)
        self._record(last, started, finish_reason, first_delta)
        return clean_answer(''.join(parts))

    def stop(self):
        if self.supervisor is not None:
            self.supervisor.stop()


def _error_detail(error):
    try:
        body = json.loads(error.read(2000).decode('utf-8', 'replace'))
        detail = body.get('error', body) if isinstance(body, dict) else body
        return str(detail.get('message', detail) if isinstance(detail, dict) else detail)[:300]
    except (ValueError, OSError):
        return str(error.reason or '')[:300]


# --------------------------------------------------------------------------- supervisor

try:  # carregado no import: depois do fork, o filho só deve chamar prctl
    _LIBC = ctypes.CDLL('libc.so.6', use_errno=True) if sys.platform.startswith('linux') else None
except OSError:
    _LIBC = None


def _die_with_parent():
    """Se o worker morrer, o llama-server (alguns GB de RAM) não deve ficar órfão (Linux)."""
    if _LIBC is not None:
        _LIBC.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


class CoreSupervisor:
    """Inicia e encerra o ``llama-server`` a partir do GGUF configurado."""

    def __init__(self, settings, *, which=shutil.which, popen=subprocess.Popen, log_dir=None):
        self.settings = settings
        self._which = which
        self._popen = popen
        self.process = None
        self.error = ''
        self.log_path = Path(log_dir or ROOT / '.agent-state') / 'local-core.log'

    def resolve_binary(self):
        name = self.settings.binary or 'llama-server'
        if os.sep in name:
            return name if os.access(name, os.X_OK) else None
        return self._which(name)

    def command(self, binary):
        settings = self.settings
        command = [binary, '-m', settings.model, '--host', '127.0.0.1', '--port', str(settings.port),
                   '-c', str(settings.context), '--jinja', '-np', '1']
        if settings.threads:
            command += ['-t', str(settings.threads)]
        if settings.mmproj:
            command += ['--mmproj', settings.mmproj]
        return command + list(settings.extra_args)

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return True
        if not Path(self.settings.model or '').is_file():
            self.error = f'GGUF não encontrado: {self.settings.model}'
            return False
        binary = self.resolve_binary()
        if not binary:
            self.error = 'llama-server não encontrado (ponha no PATH ou defina IA_LOCAL_CORE_BIN)'
            return False
        extra = {'preexec_fn': _die_with_parent} if sys.platform.startswith('linux') else {}
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'ab') as log:
                self.process = self._popen(self.command(binary), stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, **extra)
        except OSError as error:
            self.error = f'falha ao iniciar o llama-server: {error}'
            return False
        self.error = ''
        atexit.register(self.stop)
        return True

    def stop(self):
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def state(self):
        if self.error:
            return 'start-failed'
        if self.process is None:
            return 'stopped'
        return 'loading' if self.process.poll() is None else 'exited'


def local_core_from_environment(environ=None, *, start=True, stderr=sys.stderr):
    """Monta o provedor a partir de ``IA_LOCAL_CORE_*``. Devolve None quando desligado."""
    try:
        settings = CoreSettings.from_environment(environ)
    except ValueError as error:
        print(f'Núcleo local desativado: {error}', file=stderr)
        return None
    if settings is None:
        return None
    if not settings.model:
        return LocalCoreProvider(settings)
    supervisor = CoreSupervisor(settings)
    provider = LocalCoreProvider(settings, supervisor=supervisor)
    if start and not provider.configured:  # um servidor já saudável na porta é reaproveitado
        if supervisor.start():
            print(f'Núcleo local: carregando {settings.label} em {settings.url} (log: {supervisor.log_path})', file=stderr)
        else:
            print(f'Núcleo local indisponível: {supervisor.error}', file=stderr)
    return provider
