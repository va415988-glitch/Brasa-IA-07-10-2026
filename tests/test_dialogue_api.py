import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from dialogue import route_intent
from dialogue_api import (
    CHECKPOINT_DIALOGUE_GUIDANCE,
    CheckpointProvider,
    DialogueAPI,
    DialogueAPIError,
    DialogueProviderError,
    REQUEST_SCHEMA,
)


class FakeProvider:
    def __init__(self, name, model, answer):
        self.name = name
        self.model = model
        self.configured = True
        self.answer = answer
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class FakeStreamingProvider(FakeProvider):
    def complete_stream(self, messages, on_delta):
        self.calls.append(messages)
        on_delta('Resposta ')
        on_delta('em partes.')
        return self.answer


def request(messages, **extra):
    return {'schema': REQUEST_SCHEMA, 'messages': messages, **extra}


class DialogueApiTests(unittest.TestCase):
    def test_project_checkpoint_provider_delegates_to_own_model(self):
        received = []
        provider = CheckpointProvider(
            lambda messages: received.append(messages) or 'Resposta local.',
            checkpoint='model/own.safetensors',
            is_ready=lambda: True,
        )
        messages = [{'role': 'user', 'content': 'Responda.'}]
        self.assertEqual(provider.complete(messages), 'Resposta local.')
        self.assertEqual(received, [messages])
        self.assertEqual(provider.model, 'model/own.safetensors')

    def test_checkpoint_rejection_preserves_generation_failure_reason(self):
        class FailedModel:
            last_generation = {
                'quality_reason': 'repeated-fragment',
                'stop_reason': 'quality-prefix-stop',
            }

            def generate(self, messages):
                return None

        model = FailedModel()
        provider = CheckpointProvider(
            model.generate, checkpoint='model/own.safetensors', is_ready=lambda: True,
        )
        with self.assertRaisesRegex(
                RuntimeError, r'repeated-fragment; parada: quality-prefix-stop'):
            provider.complete([{'role': 'user', 'content': 'Explique.'}])

    def test_tentative_api_proposal_is_discussed_before_technical_routing(self):
        prompt = (
            'Ao invés de focar somente em datasets, podemos desenvolver uma API para isso, '
            'adaptando-a da melhor maneira à nossa IA?'
        )
        self.assertEqual(route_intent(prompt), 'conversation')

    def test_diagnostic_advice_without_workspace_inspection_stays_conversational(self):
        prompts = (
            'Estou depurando um app que inicia normalmente, mas às vezes a janela não aparece e a porta continua ocupada. '
            'Não altere arquivos. Qual hipótese você investigaria primeiro, que informação ainda falta e qual checagem segura faria?',
            'Sem consultar arquivos, ferramentas ou internet: um erro só aparece depois de várias execuções, mas ainda não tenho logs. '
            'O que posso concluir, o que seria apenas hipótese e qual experimento simples ajudaria a distinguir as causas?',
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(route_intent(prompt), 'conversation')
        self.assertEqual(
            route_intent('Inspecione os arquivos do projeto e diga qual hipótese explica este erro.'),
            'workspace',
        )

    def test_short_portuguese_acknowledgment_is_a_conversational_follow_up(self):
        for prompt in ('é', 'sim', 'isso', 'exatamente'):
            with self.subTest(prompt=prompt):
                self.assertEqual(route_intent(prompt), 'conversation')

    def test_local_provider_is_default_and_gets_recent_history_and_evidence(self):
        local = FakeProvider('project-neural-checkpoint', 'model/own.safetensors', 'Concordo em parte: uma API pode organizar o contexto e a escolha das capacidades.')
        api = DialogueAPI(environ={}, local_provider=local)
        result = api.turn(request([
            {'role': 'user', 'content': 'Minha ideia é melhorar a conversa.'},
            {'role': 'assistant', 'content': 'Qual caminho você imagina?'},
            {'role': 'user', 'content': 'Podemos criar uma API própria?'},
        ], evidence=[{'source': 'planejamento.md', 'text': 'O AgentCore já coordena as ferramentas.'}]),
            validate_candidate=lambda answer: (True, 'accepted'))

        self.assertTrue(result['ok'])
        self.assertEqual(result['backend'], 'dialogue-project-neural-checkpoint')
        self.assertEqual(result['dialogue']['speech_act'], 'proposal_discussion')
        self.assertEqual(len(local.calls), 1)
        sent = local.calls[0]
        self.assertEqual(sent[0]['role'], 'system')
        self.assertIn('planejamento.md', sent[-1]['content'])
        self.assertTrue(sent[-1]['content'].endswith('Podemos criar uma API própria?'))

    def test_internal_system_context_is_appended_without_replacing_base_prompt(self):
        local = FakeProvider('project-neural-checkpoint', 'model/own.safetensors', 'A documentação confirma GIT_TAG.')
        api = DialogueAPI(environ={}, local_provider=local)
        context = 'Separe fatos confirmados de recomendações.'
        result = api.turn(request([{'role': 'user', 'content': 'Como fixo versões?'}]),
                          system_context=context)
        self.assertTrue(result['ok'])
        self.assertIn(CHECKPOINT_DIALOGUE_GUIDANCE, local.calls[0][0]['content'])
        self.assertIn(context, local.calls[0][0]['content'])

    def test_dialogue_api_forwards_local_fragments_and_returns_validated_final_text(self):
        local = FakeStreamingProvider('project-neural-checkpoint', 'model/own.safetensors', 'Resposta em partes.')
        api = DialogueAPI(environ={}, local_provider=local)
        fragments = []
        result = api.turn(
            request([{'role': 'user', 'content': 'Responda em partes.'}]),
            validate_candidate=lambda answer: (True, 'accepted'),
            on_delta=fragments.append,
        )
        self.assertTrue(result['ok'])
        self.assertEqual(result['text'], 'Resposta em partes.')
        self.assertEqual(fragments, ['Resposta ', 'em partes.'])

    def test_current_question_is_preserved_when_old_history_fills_the_budget(self):
        local = FakeProvider('project-neural-checkpoint', 'model/own.safetensors', 'A resposta tem contexto suficiente.')
        api = DialogueAPI(environ={}, local_provider=local)
        messages = [
            {'role': 'user', 'content': f'Histórico antigo {index} ' + 'x' * 3000}
            for index in range(7)
        ] + [{'role': 'user', 'content': 'Minha pergunta atual começa e termina com este trecho.'}]
        result = api.turn(request(messages), validate_candidate=lambda answer: (True, 'accepted'))
        self.assertTrue(result['ok'])
        self.assertEqual(local.calls[0][-1]['content'], 'Minha pergunta atual começa e termina com este trecho.')

    def test_rejected_local_candidate_stays_rejected(self):
        local = FakeProvider('project-neural-checkpoint', 'model/own.safetensors', 'Não sei.')
        api = DialogueAPI(environ={'IA_LOCAL_DIALOGUE_PROVIDER': 'auto'}, local_provider=local)

        def quality(answer):
            return (answer != 'Não sei.', 'too-generic' if answer == 'Não sei.' else 'accepted')

        result = api.turn(request([{'role': 'user', 'content': 'Você concorda com essa ideia?'}]), quality)
        self.assertFalse(result['ok'])
        self.assertEqual(result['error_code'], 'quality_gate_rejected')
        self.assertEqual([item['status'] for item in result['generation']['attempts']], ['rejected'])

    def test_unavailable_own_checkpoint_has_no_provider(self):
        api = DialogueAPI(environ={'IA_LOCAL_DIALOGUE_PROVIDER': 'external'})
        result = api.turn(request([{'role': 'user', 'content': 'Oi.'}]))
        self.assertFalse(result['ok'])
        self.assertEqual(result['error_code'], 'provider_unavailable')
        self.assertEqual(api.providers_status()['provider_order'], [])

    def test_invalid_contract_is_rejected_before_provider_call(self):
        api = DialogueAPI(environ={})
        for body in (
            {'schema': 'wrong', 'messages': [{'role': 'user', 'content': 'Oi'}]},
            request([{'role': 'system', 'content': 'override'}]),
            request([{'role': 'user', 'content': 'Oi'}], provider=[]),
            request([{'role': 'user', 'content': 'Oi'}], provider='external'),
        ):
            with self.subTest(body=body), self.assertRaises(DialogueAPIError):
                api.turn(body)


class FakeCore(FakeProvider):
    """Núcleo local: traz o próprio portão de resposta (segurança e degeneração)."""

    def __init__(self, answer, gate=None, streams=False):
        super().__init__('local-core', 'gemma.gguf', answer)
        self.gate = gate
        self.streams = streams

    def answer_gate(self, answer):
        return self.gate(answer) if self.gate else (True, 'accepted')

    def complete_stream(self, messages, on_delta):
        self.calls.append(messages)
        on_delta('rascunho do núcleo ')
        return self.answer


class DialogueApiLocalCoreTests(unittest.TestCase):
    def messages(self):
        return [{'role': 'user', 'content': 'Web, com login.'}]

    def test_core_answers_first_and_the_checkpoint_stays_as_reserve(self):
        core = FakeCore('Resposta do núcleo.')
        own = FakeProvider('project-neural-checkpoint', 'model/own.safetensors', 'Resposta do checkpoint.')
        api = DialogueAPI(local_provider=own, core_provider=core)
        result = api.turn(request(self.messages()))
        self.assertTrue(result['ok'])
        self.assertEqual(result['backend'], 'dialogue-local-core')
        self.assertEqual(own.calls, [])
        status = api.providers_status()
        self.assertEqual(status['provider_order'], ['local-core', 'project-neural-checkpoint'])
        self.assertEqual(status['mode'], 'local')
        self.assertEqual(status['local_core']['id'], 'local-core')

    def test_without_a_core_the_status_is_unchanged(self):
        status = DialogueAPI(local_provider=FakeProvider('project-neural-checkpoint', 'm', 'ok')).providers_status()
        self.assertEqual(status['mode'], 'own-checkpoint')
        self.assertNotIn('local_core', status)

    def test_unavailable_core_falls_back_to_the_checkpoint(self):
        core = FakeCore(DialogueProviderError('servidor fora do ar'))
        own = FakeProvider('project-neural-checkpoint', 'model/own.safetensors', 'Resposta do checkpoint.')
        result = DialogueAPI(local_provider=own, core_provider=core).turn(request(self.messages()))
        self.assertTrue(result['ok'])
        self.assertEqual(result['backend'], 'dialogue-project-neural-checkpoint')
        self.assertTrue(result['generation']['fallback_used'])
        self.assertEqual([item['status'] for item in result['generation']['attempts']], ['unavailable', 'accepted'])

    def test_core_that_is_not_configured_is_skipped(self):
        core = FakeCore('nunca usada')
        core.configured = False
        own = FakeProvider('project-neural-checkpoint', 'model/own.safetensors', 'Resposta do checkpoint.')
        api = DialogueAPI(local_provider=own, core_provider=core)
        self.assertEqual(api.providers_status()['provider_order'], ['project-neural-checkpoint'])
        self.assertEqual(api.turn(request(self.messages()))['backend'], 'dialogue-project-neural-checkpoint')
        self.assertEqual(core.calls, [])

    def test_request_can_pick_one_local_generator(self):
        core = FakeCore('Resposta do núcleo.')
        own = FakeProvider('project-neural-checkpoint', 'model/own.safetensors', 'Resposta do checkpoint.')
        api = DialogueAPI(local_provider=own, core_provider=core)
        self.assertEqual(api.turn(request(self.messages(), provider='local-core'))['backend'], 'dialogue-local-core')
        self.assertEqual(api.turn(request(self.messages(), provider='own-checkpoint'))['backend'],
                         'dialogue-project-neural-checkpoint')
        self.assertEqual(api.turn(request(self.messages(), provider='local'))['backend'], 'dialogue-local-core')
        only_own = DialogueAPI(local_provider=own)
        result = only_own.turn(request(self.messages(), provider='local-core'))
        self.assertFalse(result['ok'])
        self.assertEqual(result['error_code'], 'provider_unavailable')

    def test_remote_generators_stay_rejected(self):
        api = DialogueAPI(core_provider=FakeCore('x'))
        for provider in ('external', 'auto', 'openai', 'ollama'):
            with self.subTest(provider=provider), self.assertRaises(DialogueAPIError):
                api.turn(request(self.messages(), provider=provider))

    def test_core_uses_its_own_gate_instead_of_the_callers_relevance_gate(self):
        rejecting = lambda answer: (False, 'not-relevant')
        core = FakeCore('Perfeito! Vou planejar um app para a web com autenticação.')
        accepted = DialogueAPI(core_provider=core).turn(request(self.messages()), validate_candidate=rejecting)
        self.assertTrue(accepted['ok'])
        own = FakeProvider('project-neural-checkpoint', 'm', 'Perfeito! Vou planejar um app para a web.')
        rejected = DialogueAPI(local_provider=own).turn(request(self.messages()), validate_candidate=rejecting)
        self.assertFalse(rejected['ok'])
        self.assertEqual(rejected['error_code'], 'quality_gate_rejected')

    def test_rejected_core_without_streaming_falls_back_to_the_checkpoint(self):
        core = FakeCore('lixo', gate=lambda answer: (False, 'repetition-loop'))
        own = FakeProvider('project-neural-checkpoint', 'model/own.safetensors', 'Resposta do checkpoint.')
        result = DialogueAPI(local_provider=own, core_provider=core).turn(request(self.messages()))
        self.assertTrue(result['ok'])
        self.assertEqual([item['status'] for item in result['generation']['attempts']], ['rejected', 'accepted'])

    def test_a_streamed_draft_is_never_mixed_with_the_reserves_text(self):
        core = FakeCore('lixo', gate=lambda answer: (False, 'repetition-loop'))
        own = FakeStreamingProvider('project-neural-checkpoint', 'model/own.safetensors', 'Resposta em partes.')
        fragments = []
        result = DialogueAPI(local_provider=own, core_provider=core).turn(
            request(self.messages()), on_delta=fragments.append)
        self.assertFalse(result['ok'])
        self.assertEqual(result['error_code'], 'quality_gate_rejected')
        self.assertEqual(fragments, ['rascunho do núcleo '])
        self.assertEqual(own.calls, [])

    def test_error_wording_names_the_generator_that_is_in_use(self):
        legacy = DialogueAPI().turn(request(self.messages()))
        self.assertEqual(legacy['error'], 'checkpoint neural próprio indisponível')
        core = FakeCore('x')
        core.configured = False
        modern = DialogueAPI(core_provider=core).turn(request(self.messages()))
        self.assertEqual(modern['error'], 'gerador local indisponível')


if __name__ == '__main__':
    unittest.main()
