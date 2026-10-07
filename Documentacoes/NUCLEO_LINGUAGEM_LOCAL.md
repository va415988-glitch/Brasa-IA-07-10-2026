# Núcleo de linguagem local (pesos abertos)

Decisão de 07/10/2026: o chat passa a poder usar um **modelo de pesos abertos
rodando nesta máquina** como núcleo de linguagem. O checkpoint próprio continua
no projeto como reserva e como trilha de pesquisa (destilação). A meta de
produto é um assistente que converse com continuidade e, depois, aceite voz e
imagem. Hardware de referência: **somente CPU**.

## Por que existe

O checkpoint ativo (`model/godmode/context-32768-v1`) tem 2 camadas e largura
128, treinado em janelas de 512 tokens (a janela de 32768 é extensão de
posições, sem treino nela). Pelas contas da configuração, são cerca de 0,4
milhão de parâmetros nos blocos do transformer e ~6,7 milhões no total, quase
tudo em tabelas de embedding e de posição. Esse porte não sustenta conversa:

- 0/24 no teste de geração neural com contexto longo
  (`logs/dialogue-evaluations/neural-active.json`) e 0/96 no avaliador de
  programação ([qualificação](QUALIFICACAO_PROGRAMACAO_2026-10-02.md));
- a bancada de 50 perguntas marcava 100% em "continuidade de contexto", mas 47
  das 50 respostas vieram de `local-conversation-rules`, e as de continuidade
  batem com frases escritas à mão em `model_server.py`. Esse 100% mede a
  tabela de regras contra a própria tabela.

As camadas de regras e as respostas prontas existem para compensar o gerador.
Trocar o gerador é o que destrava o resto.

## O que muda e o que não muda

- **Desligado por padrão.** Sem `IA_LOCAL_CORE_MODEL` ou `IA_LOCAL_CORE_URL`, o
  comportamento é o de antes.
- **Mantido:** nenhuma conversa sai da máquina. A URL do núcleo só pode ser
  `http://127.0.0.1`, `localhost` ou `[::1]`, sem credenciais; `external`, `auto`
  e qualquer provedor remoto continuam rejeitados pelo `DialogueAPI`.
- **Alterado:** antes só o checkpoint próprio gerava texto. Agora, com o núcleo
  configurado, a cadeia é `núcleo local → checkpoint próprio (reserva)`.
- Respostas do núcleo **não contam** como evidência de geração do checkpoint
  nas auditorias (`scripts/godmode.py`). Isso é intencional.

## Arquitetura

```text
interface ─► runtime Rust :3000 ─► agent-core :3200 ─► worker Python :3101
                                                          │
                                       DialogueAPI ───────┤
                                          1. LocalCoreProvider ─► llama-server :3102 (GGUF aberto)
                                          2. CheckpointProvider (checkpoint próprio, reserva)
```

O worker supervisiona o `llama-server`: inicia o processo a partir do GGUF,
guarda o log em `.agent-state/local-core.log` e o encerra junto com ele (no
Linux, também se o worker morrer). Se já houver um servidor saudável na porta,
ele é reaproveitado. Enquanto o modelo carrega, o chat usa o checkpoint.

Código: `python/language_core.py`. Integração: `DialogueAPI(core_provider=...)`
e `main()` do `model_server.py`.

## Instalar (CPU)

1. **llama.cpp.** Obtenha o binário `llama-server` (releases oficiais do
   projeto ou compilação local) e deixe-o no `PATH`, ou aponte
   `IA_LOCAL_CORE_BIN`.
2. **Um GGUF.** Candidato principal: **Gemma 4 E2B ou E4B** (versões *instruct*,
   quantização Q4_K_M). A página de multimodalidade do llama.cpp lista os dois
   com entrada de imagem e de áudio, e as fontes que consultei indicam licença
   Apache 2.0; **confirme a licença e o tamanho no cartão do modelo**. Se a CPU
   não aguentar o E4B, use o E2B ou um modelo só de texto de ~4B. Procure o GGUF
   no Hugging Face; o nome do arquivo varia por quem o publica.
3. **Configure** no `.env` da raiz (ou no ambiente):

   ```text
   IA_LOCAL_CORE_MODEL=/caminho/para/o-modelo-Q4_K_M.gguf
   IA_LOCAL_CORE_BIN=/caminho/para/llama-server   # só se não estiver no PATH
   ```

4. **Inicie** com `./start.sh` e confira o estado:

   ```bash
   curl http://127.0.0.1:3000/api/v1/dialogue/providers
   ```

   Em `local_core.state`, `loading` vira `ready` quando o modelo termina de
   carregar. Estados de falha: `start-failed` (GGUF ou binário ausente, com a
   mensagem no log do worker) e `exited` (o processo caiu; veja
   `.agent-state/local-core.log`).

### Variáveis

| Variável | Padrão | Uso |
|---|---|---|
| `IA_LOCAL_CORE_MODEL` | — | GGUF que o worker inicia |
| `IA_LOCAL_CORE_URL` | — | servidor que você já mantém (vence `MODEL`) |
| `IA_LOCAL_CORE_BIN` | `llama-server` no PATH | binário |
| `IA_LOCAL_CORE_MMPROJ` | — | projetor multimodal (visão e áudio), para as próximas etapas |
| `IA_LOCAL_CORE_PORT` | 3102 | porta do servidor iniciado |
| `IA_LOCAL_CORE_CTX` | 4096 | janela de contexto do servidor (`-c`) |
| `IA_LOCAL_CORE_THREADS` | do llama.cpp | threads (`-t`) |
| `IA_LOCAL_CORE_MAX_TOKENS` | 512 | limite de geração por resposta |
| `IA_LOCAL_CORE_HISTORY_CHARS` | 6000 | orçamento de histórico por pedido |
| `IA_LOCAL_CORE_TEMPERATURE` | 0.7 | amostragem |
| `IA_LOCAL_CORE_TIMEOUT` | 180 | segundos sem resposta do servidor |
| `IA_LOCAL_CORE_ARGS` | — | argumentos extras; **só** do ambiente do processo, nunca do `.env` |

## Medir e avaliar

```bash
# velocidade na sua CPU: 1º texto, tokens/s, e se o cache de prefixo funciona
python3 scripts/medir_nucleo_local.py

# o problema original: o 2º pedido que depende do 1º (20 conversas novas)
python3 scripts/avaliar_seguimento.py --route chat --show
python3 scripts/avaliar_seguimento.py --route agent            # caminho da interface
python3 scripts/avaliar_seguimento.py --route dialogue --provider local-core
python3 scripts/avaliar_seguimento.py --route dialogue --provider own-checkpoint   # comparação
```

A suíte (`tests/data/avaliacao_seguimento.json`) não reaproveita a bancada de
50. Cada caso exige um termo que só existe nos turnos anteriores; uma resposta
que apenas repete o último pedido, ou uma resposta genérica pronta, não passa
em nenhum caso (há teste para isso). O relatório informa a **origem** de cada
resposta e avisa quando a maioria vem de regras: um acerto vindo de regra não
prova que o seguimento foi entendido. Os critérios são amplos; leia as
respostas, o automático não mede nuance.

## Decisões que importam em CPU

- **Template do próprio GGUF** (`--jinja`): o formato multi-turno é o em que o
  modelo foi treinado, em vez de uma serialização do projeto.
- **Turnos normalizados.** Templates estritos (como o do Gemma) recusam papéis
  que não alternam; o provedor junta falas seguidas, descarta assistente no
  início e garante que o último turno seja do usuário.
- **Orçamento de histórico.** Em CPU o custo dominante é ler o prompt, não gerar.
  O padrão são 6000 caracteres de histórico (~1,5 mil tokens); a fala atual
  nunca é descartada.
- **Prefixo estável.** O servidor só reaproveita o cache de KV enquanto o início
  do prompt não muda. O guia-base é constante; a orientação por pedido vai para
  o fim da última fala do usuário (`cache_prompt` ligado).
- **Portão de qualidade próprio.** O portão do checkpoint exige que a resposta
  repita uma palavra da pergunta. Em seguimentos isso reprova respostas
  corretas: com a função real do projeto, 3 de 4 respostas certas a seguimentos
  ("Web, com login.", "Gostei do segundo, desenvolva um slogan.", "Agora deixe
  ele mais triste.") foram rejeitadas como `not-relevant`. O núcleo usa um
  portão só de segurança e degeneração (vazamento de marcadores do template,
  caracteres de controle, laços de repetição).
- **Streaming.** Se um rascunho já foi transmitido e é rejeitado, a cadeia para
  em vez de misturar o texto do checkpoint ao rascunho; as contingências do
  `_reply` respondem.
- Blocos `<think>…</think>` são removidos da resposta e do streaming.

## O que foi verificado, e o que não foi

**Verificado** (ambiente de desenvolvimento, sem GPU):

- 65 testes novos ou ampliados passam (`test_language_core.py`,
  `test_dialogue_api.py`, `test_avaliar_seguimento.py`), com um servidor falso
  que imita o protocolo do `llama-server`, inclusive streaming em *chunked*, e
  um teste que inicia um processo real pelo supervisor (um script no lugar do
  `llama-server`). Mutações deliberadas no código (prefixo, normalização,
  portão, filtro de `<think>`) fazem os testes falharem.
- A suíte de diálogo que já existia não regrediu: as mesmas 5 falhas
  pré-existentes antes e depois, 204 testes passando em ambos (com um stub de
  `torch`, porque o ambiente não o tem). Os testes de UI em Node passam.

**Não verificado:**

- **Nenhum modelo real foi executado.** A rede do ambiente de desenvolvimento
  bloqueia o Hugging Face e as releases do GitHub. Velocidade em CPU, qualidade
  em português, comportamento do template e do cache com um GGUF real ficam
  para a primeira execução na sua máquina.
- Os argumentos do `llama-server` (`-m`, `--host`, `--port`, `-c`, `-t`, `-np`,
  `--jinja`, `--mmproj`) seguem a documentação do llama.cpp que conheço e não
  foram exercitados; se uma versão recusar algum, o erro aparece em
  `.agent-state/local-core.log` e o chat continua no checkpoint.
- A rota `agent` do avaliador foi escrita a partir do código do agent-core e
  testada só contra um servidor falso.
- Tamanho e velocidade do Gemma 4 vêm de fontes secundárias (blogs e
  benchmarks de terceiros, em outros processadores). Use `medir_nucleo_local.py`.

## Limites e próximos passos

- Em CPU, modelos de ~1 a 4 bilhões de parâmetros são o teto prático. Eles
  conversam bem melhor que o checkpoint atual, mas ficam abaixo de modelos
  maiores; a troca para uma GPU depois não exige mudar nada além do GGUF.
- O núcleo só entra no caminho `DialogueAPI`. Rotas que não passam por ali
  (planejamento de produto, requisitos, memória curada) seguem com as regras de
  hoje; revisar cada uma é trabalho futuro.
- **Voz:** reconhecimento e síntese locais, com streaming e interrupção, como
  backends dos packs `audio` de `multimodal/packs.json`.
- **Visão:** `IA_LOCAL_CORE_MMPROJ` + o pack `vision`.
- **Trilha própria:** destilar o núcleo para um modelo do projeto e só trocá-lo
  quando passar nas mesmas bancadas.
- O seguimento ainda depende de o agent-core classificar o objetivo pelo texto
  isolado (`classifyObjective`): um seguimento de *build* pode virar
  `conversation`. Com o núcleo, a conversa deixa de depender do tamanho do
  modelo, mas esse roteamento continua a corrigir.
