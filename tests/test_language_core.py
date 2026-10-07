import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import language_core as core
import medir_nucleo_local as speed
from dialogue_api import CHECKPOINT_DIALOGUE_GUIDANCE, DialogueAPI, DialogueProviderError, REQUEST_SCHEMA

TIMINGS = {'prompt_n': 30, 'prompt_per_second': 40.0, 'predicted_n': 12, 'predicted_per_second': 8.5}


class FakeLlamaServer:
    """Imita o recorte do llama-server que o provedor usa: /health e /v1/chat/completions."""

    def __init__(self, reply='Resposta do núcleo.', pieces=None, health=200, chat_status=200):
        self.requests = []
        self.reply, self.pieces, self.health, self.chat_status = reply, pieces, health, chat_status
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'  # o llama-server transmite com Transfer-Encoding: chunked

            def log_message(self, *args):
                pass

            def _json(self, status, value):
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == '/health':
                    self._json(outer.health, {'status': 'ok' if outer.health == 200 else 'loading model'})
                else:
                    self.send_error(404)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                outer.requests.append(payload)
                if outer.chat_status != 200:
                    return self._json(outer.chat_status, {'error': {'message': 'template quebrado'}})
                if not payload.get('stream'):
                    return self._json(200, {'choices': [{'message': {'content': outer.reply}, 'finish_reason': 'stop'}],
                                            'timings': TIMINGS})
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()

                def chunk(text):
                    data = text.encode()
                    self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n')
                    self.wfile.flush()

                for piece in (outer.pieces or [outer.reply]):
                    chunk('data: ' + json.dumps({'choices': [{'delta': {'content': piece}}]}) + '\n\n')
                chunk('data: ' + json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'timings': TIMINGS}) + '\n\n')
                chunk('data: [DONE]\n\n')
                self.wfile.write(b'0\r\n\r\n')

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f'http://127.0.0.1:{self.httpd.server_port}'

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class Ticker:
    """Relógio que anda 10 s por leitura: o cache de saúde sempre expira."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 10.0
        return self.now


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def provider_for(url, **overrides):
    settings = core.CoreSettings(url=url, **overrides)
    return core.LocalCoreProvider(settings, clock=Ticker())


CONVERSATION = [
    {'role': 'user', 'content': 'Quero uma função em Python que some dois números.'},
    {'role': 'assistant', 'content': 'def soma(a, b):\n    return a + b'},
    {'role': 'user', 'content': 'Agora faça ela aceitar uma lista.'},
]


class LocalCoreProviderTests(unittest.TestCase):
    def setUp(self):
        self.servers = []

    def tearDown(self):
        for server in self.servers:
            server.close()

    def serve(self, **options):
        server = FakeLlamaServer(**options)
        self.servers.append(server)
        return server

    def test_complete_cleans_the_answer_and_sends_the_expected_payload(self):
        server = self.serve(reply='<think>raciocínio interno</think>Use sum(lista).<end_of_turn>')
        provider = provider_for(server.url)
        answer = provider.complete([
            {'role': 'system', 'content': 'Seja breve.'}, *CONVERSATION,
        ])
        self.assertEqual(answer, 'Use sum(lista).')
        sent = server.requests[0]
        self.assertIs(sent['stream'], False)
        self.assertIs(sent['cache_prompt'], True)
        self.assertEqual(sent['max_tokens'], 512)
        self.assertEqual([item['role'] for item in sent['messages']], ['system', 'user', 'assistant', 'user'])
        self.assertEqual(sent['messages'][-1]['content'], 'Agora faça ela aceitar uma lista.')
        self.assertEqual(provider.last_stats['completion_tokens_per_second'], 8.5)
        self.assertEqual(provider.last_stats['prompt_tokens_per_second'], 40.0)
        self.assertEqual(provider.last_stats['finish_reason'], 'stop')

    def test_stream_forwards_visible_fragments_and_hides_split_thinking_tags(self):
        server = self.serve(pieces=['Olá, ', '<th', 'ink>pensando</think>', 'como vai?'])
        provider = provider_for(server.url)
        fragments = []
        answer = provider.complete_stream(CONVERSATION, fragments.append)
        self.assertEqual(fragments, ['Olá, ', 'como vai?'])
        self.assertEqual(answer, 'Olá, como vai?')
        sent = server.requests[0]
        self.assertIs(sent['stream'], True)
        self.assertEqual(sent['stream_options'], {'include_usage': True})
        self.assertEqual(provider.last_stats['completion_tokens_per_second'], 8.5)
        self.assertIsNotNone(provider.last_stats['time_to_first_delta_s'])

    def test_server_error_becomes_a_provider_error_with_the_reason(self):
        server = self.serve(chat_status=500)
        provider = provider_for(server.url)
        with self.assertRaisesRegex(DialogueProviderError, r'HTTP 500.*template quebrado'):
            provider.complete(CONVERSATION)

    def test_unreachable_server_is_not_configured_and_raises_a_provider_error(self):
        provider = provider_for(f'http://127.0.0.1:{free_port()}')
        self.assertFalse(provider.configured)
        self.assertEqual(provider.state(), 'unreachable')
        with self.assertRaisesRegex(DialogueProviderError, 'indisponível'):
            provider.complete(CONVERSATION)

    def test_health_decides_whether_the_provider_is_configured(self):
        loading = self.serve(health=503)
        self.assertFalse(provider_for(loading.url).configured)
        ready = self.serve()
        provider = provider_for(ready.url)
        self.assertTrue(provider.configured)
        self.assertEqual(provider.status()['state'], 'ready')

    def test_request_without_a_user_turn_is_refused_before_calling_the_server(self):
        server = self.serve()
        provider = provider_for(server.url)
        with self.assertRaisesRegex(DialogueProviderError, 'fala do usuário'):
            provider.complete([{'role': 'assistant', 'content': 'Só eu falando.'}])
        self.assertEqual(server.requests, [])

    def test_per_request_guidance_moves_to_the_end_so_the_prompt_prefix_stays_stable(self):
        server = self.serve()
        provider = provider_for(server.url)
        for guidance in ('Modo engenharia: A.', 'Modo conversa: B.'):
            provider.complete([
                {'role': 'system', 'content': CHECKPOINT_DIALOGUE_GUIDANCE + '\n\n' + guidance}, *CONVERSATION,
            ])
        first, second = (request['messages'] for request in server.requests)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[0]['content'], CHECKPOINT_DIALOGUE_GUIDANCE)
        self.assertEqual(first[:-1], second[:-1])
        self.assertIn('Modo engenharia: A.', first[-1]['content'])
        self.assertIn('Modo conversa: B.', second[-1]['content'])


class PromptPreparationTests(unittest.TestCase):
    def test_turns_are_made_acceptable_to_strict_chat_templates(self):
        turns = core.normalize_turns([
            {'role': 'assistant', 'content': 'começo solto'},
            {'role': 'user', 'content': 'a'},
            {'role': 'user', 'content': 'b'},
            {'role': 'assistant', 'content': 'r1'},
            {'role': 'tool', 'content': 'ignorado'},
            {'role': 'user', 'content': '   '},
            {'role': 'user', 'content': 'c'},
            {'role': 'assistant', 'content': 'sobra no fim'},
        ])
        self.assertEqual([(item['role'], item['content']) for item in turns],
                         [('user', 'a\n\nb'), ('assistant', 'r1'), ('user', 'c')])

    def test_budget_drops_old_history_but_never_the_current_turn(self):
        messages = [
            {'role': 'system', 'content': 'S'},
            {'role': 'user', 'content': 'x' * 3000},
            {'role': 'assistant', 'content': 'y' * 3000},
            {'role': 'user', 'content': 'pergunta atual'},
        ]
        fitted = core.fit_messages(messages, 4000)
        self.assertEqual([(item['role'], item['content'][:14]) for item in fitted],
                         [('system', 'S'), ('user', 'pergunta atual')])
        long_current = core.fit_messages([{'role': 'user', 'content': 'z' * 10000}], 1000)
        self.assertLessEqual(len(long_current[0]['content']), 1000)
        self.assertIn('trecho intermediário omitido', long_current[0]['content'])

    def test_history_that_fits_is_kept_in_order(self):
        fitted = core.fit_messages(CONVERSATION, 6000)
        self.assertEqual([item['role'] for item in fitted], ['user', 'assistant', 'user'])

    def test_answer_gate_does_not_demand_lexical_overlap_with_the_question(self):
        # Respostas corretas a seguimentos quase nunca repetem as palavras da pergunta.
        self.assertEqual(core.core_answer_gate('Perfeito! Vou planejar um app para a web com autenticação.'),
                         (True, 'accepted'))
        self.assertEqual(core.core_answer_gate('Em Elixir use o pipe: x |> f() |> g()'), (True, 'accepted'))

    def test_answer_gate_rejects_leaks_loops_and_empty_answers(self):
        self.assertEqual(core.core_answer_gate('x'), (False, 'empty-answer'))
        self.assertEqual(core.core_answer_gate('Resposta <start_of_turn>user'), (False, 'prompt-leak'))
        self.assertEqual(core.core_answer_gate('Resposta <|im_end|>'), (False, 'prompt-leak'))
        self.assertEqual(core.core_answer_gate('texto\x07quebrado'), (False, 'control-character'))
        loop = ' '.join(['uma frase repetida muitas vezes sem parar'] * 8)
        self.assertEqual(core.core_answer_gate(loop), (False, 'repetition-loop'))
        lines = '\n'.join(['esta linha se repete sem parar'] * 6)
        self.assertEqual(core.core_answer_gate(lines), (False, 'repetition-loop'))

    def test_clean_answer_drops_unfinished_reasoning(self):
        self.assertEqual(core.clean_answer('<think>ainda pensando e acabou o limite'), '')
        self.assertEqual(core.clean_answer('<think>a</think>Resposta<eos>'), 'Resposta')


class SettingsTests(unittest.TestCase):
    def test_core_is_off_unless_configured(self):
        self.assertIsNone(core.CoreSettings.from_environment({}))
        self.assertIsNone(core.local_core_from_environment({}))

    def test_model_mode_starts_a_server_on_loopback_with_safe_defaults(self):
        settings = core.CoreSettings.from_environment({'IA_LOCAL_CORE_MODEL': '/m/gemma-4-E4B-it-Q4_K_M.gguf'})
        self.assertEqual(settings.url, 'http://127.0.0.1:3102')
        self.assertEqual(settings.label, 'gemma-4-E4B-it-Q4_K_M.gguf')
        self.assertEqual((settings.context, settings.max_tokens, settings.history_chars), (4096, 512, 6000))
        self.assertIsNone(settings.threads)

    def test_url_mode_wins_and_is_loopback_only(self):
        settings = core.CoreSettings.from_environment({
            'IA_LOCAL_CORE_URL': 'http://localhost:8081/v1', 'IA_LOCAL_CORE_MODEL': '/m/outro.gguf'})
        self.assertEqual(settings.url, 'http://localhost:8081')
        self.assertIsNone(settings.model)
        for remote in ('https://127.0.0.1:8080', 'http://example.com:8080', 'http://user:senha@127.0.0.1:8080'):
            with self.subTest(url=remote), self.assertRaises(ValueError):
                core.CoreSettings.from_environment({'IA_LOCAL_CORE_URL': remote})

    def test_invalid_remote_url_disables_the_core_with_a_message(self):
        class Sink:
            text = ''

            def write(self, value):
                self.text += value

            def flush(self):
                pass

        sink = Sink()
        self.assertIsNone(core.local_core_from_environment({'IA_LOCAL_CORE_URL': 'http://example.com:1'}, stderr=sink))
        self.assertIn('desativado', sink.text)

    def test_numbers_are_clamped_and_garbage_falls_back_to_defaults(self):
        settings = core.CoreSettings.from_environment({
            'IA_LOCAL_CORE_MODEL': '/m/x.gguf', 'IA_LOCAL_CORE_CTX': '999999999', 'IA_LOCAL_CORE_MAX_TOKENS': 'abc',
            'IA_LOCAL_CORE_THREADS': '6', 'IA_LOCAL_CORE_TEMPERATURE': 'nan', 'IA_LOCAL_CORE_ARGS': '--flash-attn "on"'})
        self.assertEqual(settings.context, 131072)
        self.assertEqual(settings.max_tokens, 512)
        self.assertEqual(settings.threads, 6)
        self.assertEqual(settings.temperature, 0.7)
        self.assertEqual(settings.extra_args, ('--flash-attn', 'on'))


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = Path(self.directory.name, 'modelo.gguf')
        self.model.write_bytes(b'GGUF')

    def supervisor(self, which=lambda name: '/usr/local/bin/llama-server', **overrides):
        settings = core.CoreSettings(url='http://127.0.0.1:3102', model=str(self.model), **overrides)
        self.launched = []

        def popen(command, **kwargs):
            self.launched.append((command, kwargs))
            return FakeProcess()

        return core.CoreSupervisor(settings, which=which, popen=popen, log_dir=self.directory.name)

    def test_command_binds_loopback_only_and_adds_optional_flags(self):
        supervisor = self.supervisor(threads=6, mmproj='/m/mmproj.gguf', extra_args=('--flash-attn', 'on'))
        self.assertTrue(supervisor.start())
        command, kwargs = self.launched[0]
        self.assertEqual(command[:3], ['/usr/local/bin/llama-server', '-m', str(self.model)])
        self.assertEqual(command[command.index('--host') + 1], '127.0.0.1')
        self.assertEqual(command[command.index('--port') + 1], '3102')
        self.assertEqual(command[command.index('-t') + 1], '6')
        self.assertEqual(command[command.index('--mmproj') + 1], '/m/mmproj.gguf')
        self.assertIn('--jinja', command)
        self.assertEqual(command[-2:], ['--flash-attn', 'on'])
        self.assertEqual(supervisor.state(), 'loading')
        self.assertTrue(Path(self.directory.name, 'local-core.log').is_file())

    def test_missing_binary_or_model_is_reported_not_raised(self):
        supervisor = self.supervisor(which=lambda name: None)
        self.assertFalse(supervisor.start())
        self.assertIn('llama-server não encontrado', supervisor.error)
        self.assertEqual(supervisor.state(), 'start-failed')
        self.model.unlink()
        missing = self.supervisor()
        self.assertFalse(missing.start())
        self.assertIn('GGUF não encontrado', missing.error)
        self.assertEqual(self.launched, [])

    def test_explicit_binary_must_be_executable(self):
        binary = Path(self.directory.name, 'llama-server')
        binary.write_text('#!/bin/sh\n')
        supervisor = self.supervisor(binary=str(binary))
        self.assertIsNone(supervisor.resolve_binary())
        binary.chmod(0o755)
        self.assertEqual(supervisor.resolve_binary(), str(binary))

    def test_start_is_idempotent_and_stop_terminates_the_process(self):
        supervisor = self.supervisor()
        self.assertTrue(supervisor.start())
        self.assertTrue(supervisor.start())
        self.assertEqual(len(self.launched), 1)
        process = supervisor.process
        supervisor.stop()
        self.assertTrue(process.terminated)
        self.assertEqual(supervisor.state(), 'stopped')

    def test_an_unwritable_log_directory_disables_the_core_instead_of_crashing(self):
        blocked = Path(self.model, 'sub')  # o modelo é um arquivo: criar pasta dentro dele falha
        settings = core.CoreSettings(url='http://127.0.0.1:3102', model=str(self.model))
        supervisor = core.CoreSupervisor(settings, which=lambda name: '/usr/local/bin/llama-server',
                                         popen=lambda *a, **k: self.fail('não deveria iniciar'), log_dir=blocked)
        self.assertFalse(supervisor.start())
        self.assertIn('falha ao iniciar o llama-server', supervisor.error)

    @unittest.skipIf(sys.platform == 'win32', 'o servidor falso é um script com shebang')
    def test_a_real_process_is_started_answers_and_is_stopped(self):
        script = Path(self.directory.name, 'llama-server')
        script.write_text(f'''#!{sys.executable}
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
port = int(args[args.index('--port') + 1])
assert args[args.index('--host') + 1] == '127.0.0.1' and '-m' in args and '--jinja' in args
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, code, value):
        body = json.dumps(value).encode()
        self.send_response(code); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self): self.reply(200, {{'status': 'ok'}})
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.reply(200, {{'choices': [{{'message': {{'content': 'Resposta do processo real.'}}, 'finish_reason': 'stop'}}]}})
HTTPServer(('127.0.0.1', port), Handler).serve_forever()
''')
        script.chmod(0o755)
        port = free_port()
        settings = core.CoreSettings(url=f'http://127.0.0.1:{port}', model=str(self.model), binary=str(script), port=port)
        supervisor = core.CoreSupervisor(settings, log_dir=self.directory.name)
        provider = core.LocalCoreProvider(settings, supervisor=supervisor, clock=Ticker())
        self.addCleanup(supervisor.stop)
        self.assertTrue(supervisor.start())
        deadline = time.time() + 15
        while not provider.configured and time.time() < deadline:
            time.sleep(0.1)
        self.assertTrue(provider.configured, Path(self.directory.name, 'local-core.log').read_text())
        self.assertEqual(provider.state(), 'ready')
        self.assertEqual(provider.complete(CONVERSATION), 'Resposta do processo real.')
        process = supervisor.process
        supervisor.stop()
        self.assertIsNotNone(process.poll())
        self.assertFalse(provider.configured)
        self.assertEqual(provider.state(), 'stopped')

    def test_an_already_healthy_server_on_the_port_is_reused_not_started_again(self):
        server = FakeLlamaServer()
        self.addCleanup(server.close)
        provider = core.local_core_from_environment({
            'IA_LOCAL_CORE_MODEL': str(self.model), 'IA_LOCAL_CORE_PORT': str(server.httpd.server_port)})
        self.assertTrue(provider.configured)
        self.assertIsNone(provider.supervisor.process)


class CoreThroughDialogueApiTests(unittest.TestCase):
    """O núcleo real (HTTP) dentro do contrato agent-dialogue/v1."""

    def test_follow_up_answer_passes_even_when_the_callers_relevance_gate_would_reject_it(self):
        server = FakeLlamaServer(reply='Perfeito! Vou planejar um app para a web com autenticação de usuários.')
        self.addCleanup(server.close)
        api = DialogueAPI(core_provider=provider_for(server.url))
        result = api.turn(
            {'schema': REQUEST_SCHEMA, 'messages': [
                {'role': 'user', 'content': 'Quero criar um app de controle de gastos.'},
                {'role': 'assistant', 'content': 'Boa ideia! Que plataforma você prefere: web ou celular?'},
                {'role': 'user', 'content': 'Web, com login.'},
            ]},
            validate_candidate=lambda answer: (False, 'not-relevant'),  # o portão lexical do checkpoint
        )
        self.assertTrue(result['ok'])
        self.assertEqual(result['backend'], 'dialogue-local-core')
        self.assertEqual(result['generation']['quality_gate_result'], 'accepted')
        sent = server.requests[0]['messages']
        self.assertEqual([item['role'] for item in sent], ['system', 'user', 'assistant', 'user'])
        self.assertEqual(sent[-1]['content'], 'Web, com login.')

    def test_streaming_reaches_the_client_through_the_api(self):
        server = FakeLlamaServer(pieces=['Claro, ', 'vamos lá.'])
        self.addCleanup(server.close)
        api = DialogueAPI(core_provider=provider_for(server.url))
        fragments = []
        result = api.turn({'schema': REQUEST_SCHEMA, 'messages': CONVERSATION}, on_delta=fragments.append)
        self.assertTrue(result['ok'])
        self.assertEqual(fragments, ['Claro, ', 'vamos lá.'])
        self.assertEqual(result['text'], 'Claro, vamos lá.')


class SpeedScriptTests(unittest.TestCase):
    def run_script(self, *arguments):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = speed.main(list(arguments))
        return code, out.getvalue(), err.getvalue()

    def test_prints_the_three_scenarios_and_keeps_the_history_as_a_prefix(self):
        server = FakeLlamaServer(pieces=['Resposta ', 'de teste.'])
        self.addCleanup(server.close)
        code, out, _ = self.run_script('--url', server.url, '--history-chars', '800')
        self.assertEqual(code, 0)
        for scenario in ('1 pergunta curta', '2 conversa de', '3 mesmo histórico + seguimento'):
            self.assertIn(scenario, out)
        self.assertIn('8.5 tok/s', out)
        self.assertIn('Como ler', out)
        self.assertEqual(len(server.requests), 4)  # aquecimento + 3 cenários
        second, third = server.requests[2]['messages'], server.requests[3]['messages']
        self.assertEqual(third[:len(second)], second)  # é o que permite o cache de prefixo funcionar

    def test_reports_a_server_that_is_down_and_refuses_remote_urls(self):
        code, _, err = self.run_script('--url', f'http://127.0.0.1:{free_port()}')
        self.assertEqual(code, 2)
        self.assertIn('não respondeu', err)
        code, _, err = self.run_script('--url', 'http://example.com:8080')
        self.assertEqual(code, 2)
        self.assertIn('loopback', err)


if __name__ == '__main__':
    unittest.main()
