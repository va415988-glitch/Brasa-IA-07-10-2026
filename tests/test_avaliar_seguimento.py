import contextlib
import io
import json
import re
import socket
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import avaliar_seguimento as bench

SUITE_PATH = ROOT / 'tests/data/avaliacao_seguimento.json'
SUITE = json.loads(SUITE_PATH.read_text(encoding='utf-8'))
CASES = {case['id']: case for case in SUITE['cases']}
GENERIC = 'Entendi. Vamos continuar com isso, me conte mais detalhes sobre o pedido.'


def good_answer(case):
    """Resposta de teste que contém o primeiro termo de cada critério e nenhum termo proibido."""
    terms = [check['any'][0].rstrip('*') for check in case['checks']]
    text = 'Resposta de teste: ' + ' '.join(terms)
    while len(text) < int(case.get('min_chars', 20)) + 5:
        text += ' certo'
    return text


class FakeRuntime:
    """Imita as três rotas que o avaliador chama. ``mode`` escolhe o comportamento."""

    def __init__(self, mode='good'):
        self.mode, self.requests = mode, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                outer.requests.append((self.path, body))
                request_id = body.get('request_id', '')
                case_id = request_id.split('-', 3)[3] if request_id.count('-') >= 3 else ''
                case = CASES.get(case_id)
                answer = good_answer(case) if outer.mode in ('good', 'rules', 'tool') and case else GENERIC
                if self.path == '/api/v1/agent/pursue':
                    if outer.mode == 'clarify':
                        payload = {'ok': True, 'awaitingClarification': True, 'understanding': {'question': 'Qual item?'}}
                    else:
                        payload = {'ok': True, 'report': {'finalText': answer, 'status': 'completed'}}
                else:
                    backend = 'local-conversation-rules' if outer.mode == 'rules' else 'dialogue-local-core'
                    payload = {'ok': True, 'text': answer, 'backend': backend}
                    if outer.mode == 'tool':
                        payload['tool_call'] = {'tool': 'search_files', 'arguments': {}}
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f'http://127.0.0.1:{self.httpd.server_port}'

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def run_main(*arguments):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = bench.main(list(arguments))
    return code, out.getvalue(), err.getvalue()


class SuiteQualityTests(unittest.TestCase):
    def test_suite_is_valid_and_covers_every_kind_of_follow_up(self):
        cases = bench.validate_suite(SUITE)
        self.assertEqual(len(cases), 20)
        self.assertEqual({case['kind'] for case in cases},
                         {'referencia', 'refinamento', 'resposta', 'selecao', 'correcao',
                          'continuidade', 'troca_de_assunto', 'ambiguo'})

    def test_repeating_the_last_user_message_never_passes(self):
        # Um papagaio que só devolve o último pedido não pode entender "o seguimento".
        for case in SUITE['cases']:
            with self.subTest(case=case['id']):
                parrot = case['messages'][-1]['content']
                self.assertFalse(bench.evaluate(case, parrot, True)['passed'])

    def test_a_generic_canned_reply_never_passes(self):
        for case in SUITE['cases']:
            with self.subTest(case=case['id']):
                self.assertFalse(bench.evaluate(case, GENERIC, True)['passed'])

    def test_every_case_can_be_passed_by_a_grounded_answer(self):
        for case in SUITE['cases']:
            with self.subTest(case=case['id']):
                self.assertTrue(bench.evaluate(case, good_answer(case), True)['passed'])

    def test_malformed_suites_are_rejected(self):
        broken = json.loads(json.dumps(SUITE))
        broken['cases'][0]['messages'] = broken['cases'][0]['messages'][:-1]  # termina em assistente
        with self.assertRaisesRegex(ValueError, 'terminar em fala do usuário'):
            bench.validate_suite(broken)
        duplicated = json.loads(json.dumps(SUITE))
        duplicated['cases'][1]['id'] = duplicated['cases'][0]['id']
        with self.assertRaisesRegex(ValueError, 'repetido'):
            bench.validate_suite(duplicated)


class MatchingTests(unittest.TestCase):
    def test_matching_ignores_accents_and_case_and_respects_word_boundaries(self):
        text = bench.normalize('Prezados Senhores, venho SOLICITAR o adiamento da reunião.')
        self.assertTrue(bench.term_matches(text, 'reuniao'))
        self.assertTrue(bench.term_matches(text, 'solicitar'))
        self.assertFalse(bench.term_matches(text, 'reuni'))      # sem * é palavra inteira
        self.assertTrue(bench.term_matches(text, 'reuni*'))      # com * é prefixo
        self.assertTrue(bench.term_matches(text, 'prezad*'))
        self.assertFalse(bench.term_matches(text, 'mar'))
        self.assertTrue(bench.term_matches(bench.normalize('sum(lista)'), 'sum('))
        self.assertFalse(bench.term_matches(text, '*'))

    def test_forbidden_terms_and_length_limits_fail_a_case(self):
        case = CASES['troca-capital-franca']
        self.assertTrue(bench.evaluate(case, 'A capital da França é Paris.', True)['passed'])
        dragged = bench.evaluate(case, 'Paris. Mas voltando à recursão, ela precisa de uma condição de parada.', True)
        self.assertFalse(dragged['passed'])
        self.assertEqual(dragged['forbidden_hits'], ['recurs*'])
        short = CASES['ref-resumo-uma-frase']
        self.assertFalse(bench.evaluate(short, 'REST é uma API ' + 'muito longa ' * 40, True)['passed'])
        self.assertFalse(bench.evaluate(CASES['sel-nome-segundo'], 'Bora', True)['passed'])  # abaixo do mínimo


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)
        self.servers = []

    def tearDown(self):
        for server in self.servers:
            server.close()

    def runtime(self, mode='good'):
        server = FakeRuntime(mode)
        self.servers.append(server)
        return server

    def report(self):
        files = sorted(Path(self.output.name).glob('seguimento-*.json'))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text(encoding='utf-8'))

    def test_chat_route_sends_the_history_and_reports_the_origin_of_each_answer(self):
        server = self.runtime('good')
        code, out, _ = run_main('--base-url', server.url, '--route', 'chat', '--output-dir', self.output.name,
                                '--pause-seconds', '0')
        self.assertEqual(code, 0)
        self.assertIn('Acertos: 20/20 (100.0%)', out)
        self.assertEqual(len(server.requests), 20)
        path, body = server.requests[0]
        self.assertEqual(path, '/api/chat')
        self.assertEqual(body['messages'], CASES['ref-funcao-lista']['messages'])
        summary = self.report()['summary']
        self.assertEqual(summary['origins'], {'gerador': 20})
        self.assertEqual(summary['rules_share'], 0.0)

    def test_answers_from_canned_rules_are_flagged_even_when_they_pass(self):
        server = self.runtime('rules')
        code, out, _ = run_main('--base-url', server.url, '--route', 'chat', '--output-dir', self.output.name,
                                '--pause-seconds', '0')
        self.assertEqual(code, 0)
        self.assertIn('AVISO: a maioria das respostas veio de regras prontas', out)
        self.assertEqual(self.report()['summary']['rules_share'], 100.0)

    def test_dialogue_route_can_pick_the_generator(self):
        server = self.runtime('good')
        code, _, _ = run_main('--base-url', server.url, '--route', 'dialogue', '--provider', 'local-core',
                              '--only', 'sel-sqlite', '--output-dir', self.output.name, '--pause-seconds', '0')
        self.assertEqual(code, 0)
        path, body = server.requests[0]
        self.assertEqual(path, '/api/v1/dialogue/turn')
        self.assertEqual(body['schema'], 'agent-dialogue-request/v1')
        self.assertEqual(body['provider'], 'local-core')

    def test_agent_route_mimics_the_interface_request(self):
        server = self.runtime('good')
        code, out, _ = run_main('--base-url', server.url, '--route', 'agent', '--only', 'resp-plataforma-web',
                                '--output-dir', self.output.name, '--pause-seconds', '0')
        self.assertEqual(code, 0)
        self.assertIn('Acertos: 1/1', out)
        path, body = server.requests[0]
        self.assertEqual(path, '/api/v1/agent/pursue')
        self.assertEqual(body['schema'], 'agent-request/v2')
        self.assertEqual(body['objective'], 'auto')
        self.assertEqual(body['prompt'], 'Web, com login.')
        self.assertEqual(body['history'][-1], {'role': 'user', 'content': 'Web, com login.'})
        self.assertRegex(body['operationId'], r'^agent-core-[A-Za-z0-9._-]{1,100}$')
        self.assertRegex(body['conversation_id'], r'^[A-Za-z0-9._:-]{1,160}$')

    def test_a_clarification_request_or_a_tool_request_counts_as_not_understood(self):
        clarify = self.runtime('clarify')
        run_main('--base-url', clarify.url, '--route', 'agent', '--only', 'sel-sqlite',
                 '--output-dir', self.output.name, '--pause-seconds', '0')
        row = self.report()['results'][0]
        self.assertFalse(row['passed'])
        self.assertEqual(row['problem'], 'pediu esclarecimento')
        tool = self.runtime('tool')
        for old in Path(self.output.name).glob('seguimento-*.json'):
            old.unlink()
        run_main('--base-url', tool.url, '--route', 'chat', '--only', 'sel-sqlite',
                 '--output-dir', self.output.name, '--pause-seconds', '0')
        row = self.report()['results'][0]
        self.assertFalse(row['passed'])
        self.assertEqual(row['problem'], 'pediu uma ferramenta em vez de responder')

    def test_generic_answers_fail_and_fail_under_sets_the_exit_code(self):
        server = self.runtime('generic')
        code, out, _ = run_main('--base-url', server.url, '--route', 'chat', '--fail-under', '50',
                                '--output-dir', self.output.name, '--pause-seconds', '0')
        self.assertEqual(code, 1)
        self.assertIn('Acertos: 0/20', out)

    def test_usage_errors_are_reported_with_exit_code_2(self):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            closed = probe.getsockname()[1]
        code, _, err = run_main('--base-url', f'http://127.0.0.1:{closed}', '--only', 'sel-sqlite',
                                '--output-dir', self.output.name)
        self.assertEqual(code, 2)
        self.assertIn('indisponível', err)
        code, _, err = run_main('--only', 'nao-existe', '--output-dir', self.output.name)
        self.assertEqual(code, 2)
        self.assertIn('desconhecidos', err)
        code, out, _ = run_main('--validate-only')
        self.assertEqual(code, 0)
        self.assertRegex(out, r'Bateria válida: 20 casos em 8 tipos')


if __name__ == '__main__':
    unittest.main()
