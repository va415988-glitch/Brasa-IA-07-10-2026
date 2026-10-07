# API de diálogo do checkpoint próprio

## Objetivo

`agent-dialogue/v1` prepara histórico e evidências, chama um gerador local e
valida a resposta. Os geradores locais são o checkpoint neural treinado neste
projeto e, quando configurado, o
[núcleo de linguagem local de pesos abertos](NUCLEO_LINGUAGEM_LOCAL.md), um
modelo que roda nesta máquina (`llama-server` em loopback). O runtime não
encaminha conversas a Ollama, a serviços remotos nem a qualquer endpoint fora
da máquina. A web é uma fonte opcional de evidências; quem escreve a resposta é
sempre um gerador local.

## Rotas

### `GET /api/v1/dialogue/providers`

Retorna `agent-dialogue-providers/v1`, com `mode: "own-checkpoint"` e o estado
do checkpoint carregado. Com o núcleo local configurado, `mode` passa a `"local"`
e a resposta traz um bloco `local_core` (`configured`, `model`, `server`,
`state`, `managed`); `provider_order` lista a cadeia na ordem de uso. Não
retorna chaves ou credenciais.

### `POST /api/v1/dialogue/turn`

Recebe `agent-dialogue-request/v1`:

```json
{
  "schema": "agent-dialogue-request/v1",
  "request_id": "turno-123",
  "messages": [
    {"role": "user", "content": "Tenho uma proposta para melhorar a conversa."},
    {"role": "assistant", "content": "Qual é a proposta?"},
    {"role": "user", "content": "Podemos criar uma API própria?"}
  ],
  "evidence": [
    {"source": "https://exemplo.org/documentacao", "text": "Trecho consultado..."}
  ]
}
```

`provider` pode ser omitido ou definido como `local` (a cadeia: o núcleo local,
se configurado, e depois o checkpoint próprio), `own-checkpoint` (só o
checkpoint) ou `local-core` (só o núcleo local). Valores como `external` e
`auto` são rejeitados. O pedido aceita até 80 mensagens, mas o contexto entregue
ao gerador mantém no máximo oito mensagens recentes e até oito evidências (o
núcleo local aplica ainda um orçamento de histórico em caracteres,
`IA_LOCAL_CORE_HISTORY_CHARS`, porque em CPU ler o prompt é o custo dominante).
Texto de páginas externas é dado não confiável, não uma instrução.

A resposta `agent-dialogue-response/v1` registra o backend (`dialogue-local-core`
ou `dialogue-project-neural-checkpoint`), a identificação do modelo, o gate de
qualidade e as tentativas. Cada gerador tem o seu gate: o checkpoint usa o gate
do chamador; o núcleo local usa um gate de segurança e degeneração, sem exigir
sobreposição lexical com a pergunta (que reprova respostas corretas a
seguimentos). Se nenhum gerador local estiver disponível ou a resposta reprovar,
a rota retorna erro; ela não troca para um gerador remoto. Se um rascunho já foi
transmitido em streaming e reprovou, a cadeia para em vez de misturar textos.
Fallbacks de regras que existam no chat principal são identificados
separadamente e não são apresentados como geração neural.

## Busca web

Quando `IA_LOCAL_BRAVE_SEARCH_API_KEY` está configurada no ambiente do runtime,
`search_web` chama o endpoint LLM Context da Brave Search API. Ele retorna
trechos de páginas já extraídos, títulos, URLs e metadados; `research_web`
reutiliza esses trechos para compor evidências sem baixar as mesmas páginas de
novo. O checkpoint próprio continua responsável por gerar a resposta. O endpoint
Answers da Brave, que gera respostas, não é usado. Sem a chave, a busca usa o
scraper existente do DuckDuckGo. Uma falha da API configurada aparece como erro
e não dispara uma troca silenciosa de provedor.

A requisição envia `Cache-Control: no-cache` (a Brave descreve esse controle
como best effort). Os trechos brutos são removidos dos tool traces e da cópia
persistida do histórico do navegador; a resposta final do assistente continua
salva como parte normal da conversa. Para salvar conteúdo retornado pela Brave
no acervo, é preciso definir `IA_LOCAL_BRAVE_ALLOW_STORAGE=true`, e somente se o
plano contratado conceder direitos explícitos de armazenamento. A chave de busca,
por si só, não concede direitos sobre o conteúdo das páginas. RAG melhora a
informação disponível durante a resposta, mas não atualiza os pesos.

## Verificação manual

```bash
curl http://127.0.0.1:3000/api/v1/dialogue/providers

curl -X POST http://127.0.0.1:3000/api/v1/dialogue/turn \
  -H 'Content-Type: application/json' \
  -d '{"schema":"agent-dialogue-request/v1","messages":[{"role":"user","content":"Você concorda com essa ideia?"}]}'
```
