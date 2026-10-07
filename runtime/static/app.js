'use strict';
const chat = document.getElementById('chat'), input = document.getElementById('message');
let conversation = [], timeline = [], attachments = [], currentConversationId = null;
let agentRunRef = null;
let agentTaskRef = null;
let pendingAgent = null;
let selectedTool = null, sending = false, activityCursor = 0, polling = false;
let openFilePath = null, originalFileContent = '';
let openFileWorkspaceRoot = '';
let dockDirectoryPath = '', dockWorkspaceRoot = '', dockRequestSequence = 0, dockTrashItems = [];
let workspaceEntryDialogState = null;
let previewZoom = 1;
let workflowReviewOffset = 0;
let interfaceStudioDraft = false;
let pendingInterfacePreview = null;
let interfaceEditPathDraft = null;
let projectCatalogBase = localStorage.getItem('ia-local-zero-project-base') || '';
const liveActions = new Map();
const agentStreamDrafts = new Map();
const MAX_AGENT_STEPS = 32;
const workflowLabels = {understand:'Entendendo a solicitação', plan:'Montando o plano', learn:'Consultando conhecimento', retrieve:'Recuperando contexto', act:'Executando a etapa', verify:'Verificando o resultado', answer:'Preparando a resposta', complete:'Fluxo concluído'};
const HISTORY_KEY = 'ia-local-zero-conversations-v1';
const LAST_CONVERSATION_KEY = 'ia-local-zero-last-conversation-v1';
const toolLabels = {terminal_run:'Executando comando aprovado', process_start:'Iniciando servidor local aprovado', process_status:'Acompanhando processo local', process_stop:'Encerrando processo local aprovado', apply_batch:'Aplicando lote revisável', undo_batch:'Desfazendo lote', search_web:'Pesquisando na internet', research_web:'Fazendo pesquisa guiada', open_page:'Lendo página', list_files:'Listando arquivos', search_files:'Buscando no código', inspect_project:'Inspecionando o projeto', project_checks:'Executando testes do projeto', diagnose_project:'Diagnosticando falha', propose_repair:'Validando correção', apply_repair:'Aplicando correção', read_file:'Lendo arquivo', extract_document_text:'Extraindo texto do documento', list_sources:'Consultando fontes', set_workspace:'Selecionando projeto', create_workspace:'Criando projeto', create_file:'Criando arquivo', create_web_page:'Criando página web', edit_file:'Editando arquivo', create_directory:'Criando pasta'};
const backendLabels = {'static-analysis':'Análise estática dos anexos','local-conversation':'Conversa local','diagnostic-reasoning-fallback':'Triagem diagnóstica local · contingência','research-evidence-fallback':'Síntese verificada das fontes · contingência','curated-memory':'Memória curada · fallback','local-knowledge':'Acervo local · fallback','local-neural':'Modelo neural próprio','dialogue-local-core':'Núcleo local · pesos abertos','research-evidence':'Síntese das fontes consultadas','source-assessment':'Verificação de fontes','quality-gate':'Limite de conhecimento'};
function node(tag, className, text) { const el = document.createElement(tag); if (className) el.className = className; if (text != null) el.textContent = text; return el; }
// Laboratório experimental: pedidos isolados, sem despacho de ferramentas.
let experimentalLabState = null, experimentalLabPending = false;
const experimentalCoreLabels = {
  'decision-format':'Formato e contratos', 'evidence-extraction':'Extração e fontes',
  'evidence-abstention':'Abstenção e conflitos', 'numeric-comparison':'Comparação numérica',
  'tool-arguments':'Consultas e argumentos', programming:'Programação', research:'Pesquisa',
  planning:'Planejamento', communication:'Comunicação', mathematics:'Matemática'
};
function setExperimentalLabBusy(pending) {
  experimentalLabPending = pending;
  const submit = document.getElementById('cognitive-lab-submit');
  submit.disabled = pending || experimentalLabState?.enabled !== true;
  submit.textContent = pending ? 'Aguarde…' : 'Analisar com a V4';
  document.getElementById('cognitive-lab-refresh').disabled = pending;
}
function renderExperimentalLabStatus(data) {
  experimentalLabState = data;
  const status = document.getElementById('cognitive-lab-status');
  status.dataset.state = 'failed';
  status.textContent = data.enabled ? 'Disponível para análise experimental · competência reprovada' : 'Laboratório desativado · competência reprovada';
  document.getElementById('cognitive-lab-model').textContent = `${data.label || 'V4 · experimental'} · contexto de ${data.context_tokens} tokens · ${data.loaded ? 'modelo carregado' : 'carregamento na primeira análise'}`;
  document.getElementById('cognitive-lab-model').title = String(data.checkpoint || '');
  document.getElementById('cognitive-lab-limits').textContent = (data.limits || []).map(String).join(' · ');
  const cores = document.getElementById('cognitive-lab-cores');
  cores.replaceChildren();
  for (const [id, state] of Object.entries(data.core_states || {})) {
    const row = node('li');
    const label = experimentalCoreLabels[id] || id;
    const result = state === 'failed' ? 'Reprovado' : state === 'not_evaluated' ? 'Não avaliado' : 'Sem prova válida';
    row.append(node('span', null, label), node('strong', null, result));
    cores.append(row);
  }
  if (!cores.children.length) cores.append(node('li', null, 'Nenhuma prova de competência disponível.'));
}
async function refreshExperimentalLab() {
  if (experimentalLabPending) return;
  experimentalLabState = null;
  setExperimentalLabBusy(true);
  const status = document.getElementById('cognitive-lab-status');
  status.textContent = 'Consultando o modelo…';
  document.getElementById('cognitive-lab-error').textContent = '';
  document.getElementById('cognitive-lab-model').textContent = '';
  document.getElementById('cognitive-lab-limits').textContent = '';
  document.getElementById('cognitive-lab-cores').replaceChildren();
  try {
    const response = await fetch('/api/v1/models/experimental', {cache:'no-store'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || data.text || `Falha ao consultar o modelo (HTTP ${response.status}).`);
    if (data.schema !== 'experimental-model-status/v1' || data.qualified !== false) throw new Error('O estado recebido não corresponde ao laboratório experimental.');
    renderExperimentalLabStatus(data);
  } catch (error) {
    status.textContent = 'Não foi possível consultar o laboratório.';
    document.getElementById('cognitive-lab-error').textContent = error.message;
  } finally { setExperimentalLabBusy(false); }
}
async function openExperimentalLab() {
  const dialog = document.getElementById('cognitive-lab-dialog');
  if (!dialog.open) dialog.showModal();
  await refreshExperimentalLab();
}
function closeExperimentalLab() { document.getElementById('cognitive-lab-dialog').close(); }
function useExperimentalLabExample(kind) {
  document.getElementById('cognitive-lab-question').value = kind === 'compare' ? 'A versão 41 atende ao mínimo exigido por Pipa?' : 'Qual é o limite de Pipa?';
  document.getElementById('cognitive-lab-source').value = kind === 'compare' ? 'Pipa: versão mínima = 42.' : 'Pipa: limite = 42.';
  document.getElementById('cognitive-lab-question').focus();
}
function experimentalLabRequest(question, source) {
  return {
    schema:'experimental-cognitive-request/v1',
    messages:[{role:'user', content:question}, {role:'tool', content:JSON.stringify({tool:'read_file', ok:true, data:{path:'dados.json', content:source}, error:''})}],
    cognition:{schema:'agent-cognition/v1', available_tools:[]}
  };
}
function renderExperimentalLabResult(data, accepted) {
  document.getElementById('cognitive-lab-result').hidden = false;
  document.getElementById('cognitive-lab-result-state').textContent = accepted ? 'Saída estruturada recebida · competência reprovada' : 'Análise bloqueada · competência reprovada';
  document.getElementById('cognitive-lab-result-text').textContent = String(data.text || data.error || 'O modelo não retornou uma resposta.');
  document.getElementById('cognitive-lab-decision').textContent = data.decision == null ? 'Nenhuma decisão aceita.' : JSON.stringify(data.decision, null, 2);
  document.getElementById('cognitive-lab-raw').textContent = String(data.raw_output || 'Nenhuma saída bruta disponível.');
}
async function submitExperimentalLab(event) {
  event.preventDefault();
  if (experimentalLabPending || experimentalLabState?.enabled !== true) return;
  const question = document.getElementById('cognitive-lab-question').value.trim();
  const source = document.getElementById('cognitive-lab-source').value.trim();
  const errorBox = document.getElementById('cognitive-lab-error');
  errorBox.textContent = '';
  document.getElementById('cognitive-lab-result').hidden = true;
  if (!question || !source || question.length > 200 || source.length > 200) {
    errorBox.textContent = 'Preencha a pergunta e o conteúdo observado com até 200 caracteres cada.';
    return;
  }
  setExperimentalLabBusy(true);
  try {
    const response = await fetch('/api/v1/models/experimental/decide', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(experimentalLabRequest(question, source))});
    const data = await response.json();
    const accepted = response.ok && data.ok === true && data.schema === 'experimental-cognitive-response/v1' && data.experimental === true && data.qualified === false && data.tool_executed === false;
    renderExperimentalLabResult(data, accepted);
    if (!accepted) errorBox.textContent = String(data.error || data.text || `A análise não foi aceita (HTTP ${response.status}).`);
    if (experimentalLabState) {
      renderExperimentalLabStatus({...experimentalLabState, loaded:response.ok || Boolean(data.raw_output)});
    }
  } catch (error) {
    errorBox.textContent = `Não foi possível concluir a análise: ${error.message}`;
  } finally { setExperimentalLabBusy(false); }
}
// Fim do laboratório experimental.
let followChatTail = true;
function scrollChat() {
  if (followChatTail) chat.scrollTop = chat.scrollHeight;
  const jump = document.getElementById('chat-jump-latest');
  if (jump) jump.hidden = followChatTail;
}
chat.addEventListener('scroll', () => {
  followChatTail = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 90;
  const jump = document.getElementById('chat-jump-latest');
  if (jump) jump.hidden = followChatTail;
}, {passive:true});
function jumpToLatestMessage() {
  followChatTail = true;
  scrollChat();
}
function setWorkspaceIndicator(path) {
  if ((path || '') !== dockWorkspaceRoot) {
    dockWorkspaceRoot=path || ''; dockDirectoryPath=''; dockTrashItems=[]; dockRequestSequence++;
    const files=document.getElementById('dock-files');
    if(files) files.replaceChildren(node('div','dock-empty',path?'Carregando arquivos…':'Selecione um projeto para gerenciar arquivos.'));
    const status=document.getElementById('dock-status');
    if(status) status.textContent='Nenhuma verificação executada.';
  }
  const value = path || 'Nenhum projeto selecionado';
  if (path) {
    localStorage.setItem('ia-local-zero-workspace', path);
    const parent = path.replace(/\\/g, '/').replace(/\/$/, '').split('/').slice(0, -1).join('/') || '/';
    projectCatalogBase = parent;
    localStorage.setItem('ia-local-zero-project-base', parent);
  }
  const side = document.getElementById('workspace-current');
  const header = document.getElementById('header-workspace');
  const headerLabel = document.getElementById('header-workspace-label');
  const dock = document.getElementById('dock-workspace');
  if (side) { side.textContent = value; side.title = value; }
  if (headerLabel) headerLabel.textContent = value;
  if (header) { header.title = value; header.setAttribute('aria-label', `Projeto ativo: ${value}`); }
  if (dock) { dock.textContent = value; dock.title = value; }
  const homePath=document.getElementById('workspace-home-path');
  if (homePath) homePath.textContent=value;
  const rootLabel=document.getElementById('dock-root-label');
  if (rootLabel && path) rootLabel.textContent=(path.split('/').filter(Boolean).pop() || 'PROJETO').toUpperCase();
  updateDockFileActions();
}
function workspaceRoutingContext() {
  return {workspace_selected:Boolean(localStorage.getItem('ia-local-zero-workspace'))};
}
function projectParent(path) {
  const normalized = String(path || '').replace(/\\/g, '/').replace(/\/$/, '');
  const index = normalized.lastIndexOf('/');
  return index > 0 ? normalized.slice(0, index) : '/';
}
function closeProjectMenu() {
  const menu = document.getElementById('project-menu');
  const button = document.getElementById('header-workspace');
  if (menu) menu.hidden = true;
  if (button) button.setAttribute('aria-expanded', 'false');
}
function projectMenuItem(project, current) {
  const button = node('button', `project-menu-item${project.path === current ? ' active' : ''}`);
  button.type = 'button';
  button.title = project.path;
  const icon = node('span', 'project-menu-icon', project.kind === 'git' ? '◆' : '▦');
  const copy = node('span', 'project-menu-copy');
  copy.append(node('strong', null, project.name), node('small', null, project.path));
  button.append(icon, copy, node('span', 'project-menu-check', project.path === current ? '✓' : ''));
  button.onclick = () => selectHeaderProject(project.path);
  return button;
}
async function loadProjectMenu() {
  const list = document.getElementById('project-list');
  if (!list) return;
  list.replaceChildren(node('div', 'project-menu-loading', 'Carregando projetos locais…'));
  try {
    const response = await fetch('/api/projects', {cache:'no-store'});
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || 'não foi possível listar os projetos');
    projectCatalogBase = data.base || projectCatalogBase;
    if (projectCatalogBase) localStorage.setItem('ia-local-zero-project-base', projectCatalogBase);
    const current = localStorage.getItem('ia-local-zero-workspace') || data.current || '';
    list.replaceChildren();
    for (const project of data.projects || []) list.append(projectMenuItem(project, current));
    if (!list.children.length) list.append(node('div', 'project-menu-loading', 'Nenhuma pasta de projeto encontrada.'));
  } catch (error) {
    list.replaceChildren(node('div', 'project-menu-error', error.message));
  }
}
async function toggleProjectMenu(event) {
  event?.stopPropagation();
  const menu = document.getElementById('project-menu');
  const button = document.getElementById('header-workspace');
  if (!menu) return;
  const opening = menu.hidden;
  menu.hidden = !opening;
  if (button) button.setAttribute('aria-expanded', String(opening));
  if (opening) await loadProjectMenu();
}
async function selectHeaderProject(path) {
  closeProjectMenu();
  const selected = await toolRequest('set_workspace', {path}, false);
  if (!selected.ok) return notice(selected.error || 'Não foi possível selecionar esse projeto.');
  setWorkspaceIndicator(selected.data.workspace);
  refreshProjectDock();
  notice(`Projeto ativo: ${selected.data.workspace}`);
}
async function chooseHeaderProject() {
  closeProjectMenu();
  try {
    const response = await fetch('/api/choose-directory', {cache:'no-store'});
    const data = await response.json();
    if (data.ok) await selectHeaderProject(data.path);
    else if (!data.cancelled) notice(data.error || 'Não foi possível escolher a pasta.');
  } catch (error) { notice(`Não foi possível abrir o gerenciador: ${error.message}`); }
}
async function createHeaderProject() {
  const field = document.getElementById('project-name-input');
  const name = field?.value.trim() || '';
  if (!name || name === '.' || name === '..' || /[\\/]/.test(name)) return notice('Informe apenas o nome da nova pasta, sem barras.');
  const parent = projectCatalogBase || projectParent(localStorage.getItem('ia-local-zero-workspace'));
  if (!parent) return notice('Escolha primeiro uma pasta-base para o projeto.');
  const path = `${parent.replace(/\/$/, '')}/${name}`;
  const data = await toolRequest('create_workspace', {path}, false);
  if (!data.ok) return notice(data.error || 'Não foi possível criar o projeto.');
  setWorkspaceIndicator(data.data.workspace);
  if (field) field.value = '';
  await loadProjectMenu();
  refreshProjectDock();
  notice(`Novo projeto criado: ${data.data.workspace}`);
}
function updateEditorMeta() {
  const editor=document.getElementById('stage-editor');
  const info=document.getElementById('stage-editor-info');
  if(!editor || !info) return;
  const lines=editor.value ? editor.value.split('\n').length : 0;
  info.textContent=`${lines} ${lines===1?'linha':'linhas'}`;
}
function copyEditorCode() {
  const editor=document.getElementById('stage-editor');
  if(!editor || !editor.value) return notice('Nenhum código aberto para copiar.');
  navigator.clipboard.writeText(editor.value).then(()=>{
    notice('Código do arquivo copiado para a área de transferência.');
  }).catch(()=>{
    notice('Não foi possível copiar o código.');
  });
}
function copyCodeBlock(button) {
  const pre = button.closest('.code-block-wrap')?.querySelector('.code-block-pre');
  const codeEl = pre?.querySelector('code');
  if (!codeEl) return;
  const text = pre.dataset.raw || codeEl.textContent || '';
  navigator.clipboard.writeText(text).then(() => {
    const originalText = button.textContent;
    button.textContent = 'Copiado!';
    button.classList.add('copied');
    setTimeout(() => {
      button.textContent = originalText;
      button.classList.remove('copied');
    }, 1800);
  }).catch(() => {
    button.textContent = 'Erro';
    setTimeout(() => { button.textContent = 'Copiar'; }, 1800);
  });
}
function toggleCodeBlock(button) {
  const wrap = button.closest('.code-block-wrap');
  if (!wrap) return;
  const expanded = wrap.classList.toggle('expanded');
  button.textContent = expanded ? 'Recolher' : 'Expandir';
}
function openCodeOverlay(button) {
  const wrap = button.closest('.code-block-wrap');
  const pre = wrap?.querySelector('.code-block-pre');
  const lang = wrap?.querySelector('.code-block-lang')?.textContent || 'código';
  if (!pre) return;
  const raw = pre.dataset.raw || pre.textContent || '';
  const overlay = node('div', 'code-overlay');
  const card = node('section', 'code-overlay-card');
  const head = node('div', 'code-overlay-head');
  const title = node('div');
  title.append(node('div', 'code-overlay-title', `Código · ${lang}`), node('div', 'code-overlay-subtitle', `${raw.split('\n').length} linhas · modo de foco`));
  const actions = node('div', 'code-overlay-actions');
  const copy = node('button', null, 'Copiar');
  copy.onclick = () => navigator.clipboard.writeText(raw).then(() => { copy.textContent = 'Copiado'; setTimeout(() => { copy.textContent = 'Copiar'; }, 1400); });
  const close = node('button', null, 'Fechar');
  close.onclick = () => overlay.remove();
  actions.append(copy, close); head.append(title, actions);
  const body = node('pre', 'code-overlay-pre'); body.textContent = raw;
  card.append(head, body); overlay.append(card); overlay.onclick = event => { if (event.target === overlay) overlay.remove(); };
  document.body.append(overlay); close.focus();
}
function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  if (sidebar) sidebar.classList.toggle('open');
}
function toggleWorkspaceExplorer() {
  const headerToggle = document.getElementById('context-panel-toggle');
  if (document.body.classList.contains('chat-mode') && window.matchMedia('(max-width: 1050px)').matches) {
    const open = document.body.classList.toggle('chat-context-open');
    headerToggle?.setAttribute('aria-expanded', String(open));
    headerToggle?.setAttribute('aria-label', open ? 'Fechar painel de contexto' : 'Abrir painel de contexto');
    return;
  }
  const collapsed = document.body.classList.toggle('workspace-explorer-collapsed');
  localStorage.setItem('ia-local-zero-explorer-collapsed', String(collapsed));
  headerToggle?.setAttribute('aria-expanded', String(!collapsed));
  headerToggle?.setAttribute('aria-label', collapsed ? 'Expandir painel de contexto' : 'Recolher painel de contexto');
  const button = document.querySelector('.project-dock .dock-head-actions .panel-close');
  if (button) {
    button.textContent = collapsed ? '›' : '‹';
    button.setAttribute('aria-label', collapsed ? 'Expandir Explorer' : 'Recolher Explorer');
  }
}
function restoreWorkspaceExplorer() {
  if (localStorage.getItem('ia-local-zero-explorer-collapsed') === 'true') {
    document.body.classList.add('workspace-explorer-collapsed');
    const button = document.querySelector('.project-dock .dock-head-actions .panel-close');
    if (button) { button.textContent = '›'; button.setAttribute('aria-label', 'Expandir Explorer'); }
  }
  const expanded = !document.body.classList.contains('workspace-explorer-collapsed');
  document.getElementById('context-panel-toggle')?.setAttribute('aria-expanded', String(expanded));
}
function setEditorFile(data) {
  const tab=document.getElementById('stage-tab'), editor=document.getElementById('stage-editor');
  if(!tab || !editor) return;
  document.body.classList.add('has-editor');
  document.body.classList.remove('workspace-mode');
  openFilePath=data.path; originalFileContent=data.content || '';
  openFileWorkspaceRoot=localStorage.getItem('ia-local-zero-workspace') || '';
  document.querySelector('.stage-workarea')?.classList.remove('preview-open');
  tab.dataset.path=data.path || 'Arquivo';
  tab.textContent=tab.dataset.path;
  editor.value=data.content || '';
  updateEditorMeta();
  const copyBtn = document.getElementById('stage-copy');
  if(copyBtn) copyBtn.disabled = false;
  ['stage-save','stage-review','stage-discard'].forEach(id=>{ const button=document.getElementById(id); if(button) button.disabled=true; });
  const previewButton=document.getElementById('stage-preview-button'); if(previewButton) { previewButton.disabled=!/\.html?$/i.test(data.path || ''); previewButton.textContent='Prévia'; }
  ['stage-zoom-out','stage-zoom-in'].forEach(id=>{ const button=document.getElementById(id); if(button) button.disabled=!/\.html?$/i.test(data.path || ''); });
  editor.oninput=()=>{
    const dirty=editor.value!==originalFileContent;
    tab.textContent=`${tab.dataset.path || openFilePath || 'Arquivo'}${dirty ? ' •' : ''}`;
    ['stage-save','stage-review','stage-discard'].forEach(id=>{ const button=document.getElementById(id); if(button) button.disabled=!dirty; });
    updateEditorMeta();
  };
  editor.onkeydown=(e)=>{
    if (e.key === 'Tab') {
      e.preventDefault();
      const start = editor.selectionStart, end = editor.selectionEnd;
      editor.value = editor.value.substring(0, start) + '  ' + editor.value.substring(end);
      editor.selectionStart = editor.selectionEnd = start + 2;
      editor.dispatchEvent(new Event('input'));
    } else if ((e.ctrlKey || e.metaKey) && e.key === 's') {
      e.preventDefault();
      if (editor.value !== originalFileContent) {
        saveEditorFile();
      }
    }
  };
}
function closeEditorFile(force=false) {
  if (!openFilePath) return;
  const editor=document.getElementById('stage-editor');
  if (!force && editor && editor.value !== originalFileContent && !confirm('Há alterações não salvas. Fechar mesmo assim?')) return;
  openFilePath = null;
  originalFileContent = '';
  openFileWorkspaceRoot = '';
  document.body.classList.remove('has-editor');
  const tab=document.getElementById('stage-tab'), info=document.getElementById('stage-editor-info');
  if (tab) { tab.textContent='Nenhum arquivo aberto'; tab.dataset.path=''; }
  if (info) info.textContent='';
  if (editor) { editor.value=''; editor.oninput=null; editor.onkeydown=null; }
  document.querySelector('.stage-workarea')?.classList.remove('preview-open');
  ['stage-copy','stage-save','stage-review','stage-discard','stage-preview-button','stage-zoom-out','stage-zoom-in'].forEach(id=>{
    const button=document.getElementById(id); if(button) button.disabled=true;
  });
}
function updateDockFileActions() {
  for(const id of ['dock-new-file','dock-new-folder']) {
    const button=document.getElementById(id); if(button) button.disabled=!dockWorkspaceRoot;
  }
  const restore=document.getElementById('dock-restore');
  if(restore) { restore.disabled=!dockTrashItems.length; restore.title=dockTrashItems.length ? `Restaurar ${dockTrashItems[0].path}` : 'Nenhum item na lixeira local'; }
}
async function workspaceEntryRequest(operation, fields, workspace, visible=true) {
  const id=requestId('entry');
  const payload={schema:'workspace-entry/v1',operation,workspace_root:workspace,request_id:id,...fields};
  if(visible) return requestWithActivity('/api/v1/workspace/entries',payload,`workspace-entry:${id}`,
    operation==='trash'?'Movendo para a lixeira':operation==='restore'?'Restaurando item':operation==='edit_file'?'Salvando arquivo':'Criando no workspace',false);
  try {
    const response=await fetch('/api/v1/workspace/entries',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});
    return await response.json();
  } catch(error) { return {ok:false,error:error.message}; }
}
function openDockFolder(path) { return renderDockDirectory(path); }
function loadDockRoot() { return renderDockDirectory(''); }
async function renderDockDirectory(path) {
  const dockFiles=document.getElementById('dock-files'); if(!dockFiles) return;
  const workspace=localStorage.getItem('ia-local-zero-workspace'), sequence=++dockRequestSequence;
  if(!workspace) { dockFiles.replaceChildren(node('div','dock-empty','Selecione um projeto para gerenciar arquivos.')); return; }
  const data=await toolRequest('list_files',{path,_expected_workspace:workspace},false);
  if(sequence!==dockRequestSequence || workspace!==localStorage.getItem('ia-local-zero-workspace')) return;
  if(!data.ok) { dockFiles.replaceChildren(node('div','dock-empty',data.error)); return; }
  dockDirectoryPath=path; dockFiles.replaceChildren();
  const rootLabel=document.getElementById('dock-root-label');
  if(rootLabel) rootLabel.textContent=path || (workspace.split('/').filter(Boolean).pop() || 'PROJETO').toUpperCase();
  if(path) {
    const back=node('button','dock-action dock-parent','‹ Pasta anterior');
    back.type='button'; back.onclick=()=>openDockFolder(path.split('/').slice(0,-1).join('/')); dockFiles.append(back);
  }
  for(const entry of data.data.entries.slice(0,80)) {
    const relative=(path?path+'/':'')+entry.name;
    const row=node('div','entry'), label=node('div','dock-entry-label',`${entry.kind==='directory'?'▣':'▤'} ${entry.name}`);
    row.dataset.path=relative; label.title=relative;
    const actions=node('div','dock-entry-actions'), open=node('button',null,entry.kind==='directory'?'›':'Abrir');
    open.type='button'; open.setAttribute('aria-label',`Abrir ${entry.name}`);
    open.onclick=()=>entry.kind==='directory'?openDockFolder(relative):runPanelRead(relative);
    label.onclick=open.onclick; label.style.cursor='pointer';
    const remove=node('button','dock-entry-delete','×'); remove.type='button';
    remove.setAttribute('aria-label',`Excluir ${entry.name}`); remove.title=`Excluir ${entry.name}`;
    remove.disabled=!['file','directory'].includes(entry.kind);
    remove.onclick=()=>openWorkspaceEntryDialog('trash',relative,entry.kind);
    actions.append(open,remove); row.append(label,actions); dockFiles.append(row);
  }
  if(!data.data.entries.length) dockFiles.append(node('div','dock-empty','Pasta vazia. Crie um arquivo ou uma pasta pelos botões acima.'));
  if(data.data.entries.length>80 || data.data.truncated) dockFiles.append(node('div','dock-empty','A listagem foi limitada aos primeiros 80 itens.'));
  const trash=await workspaceEntryRequest('list_trash',{},workspace,false);
  if(sequence===dockRequestSequence && workspace===localStorage.getItem('ia-local-zero-workspace')) {
    dockTrashItems=trash.ok && Array.isArray(trash.data?.entries) ? trash.data.entries : [];
    updateDockFileActions();
  }
}
function openWorkspaceEntryDialog(operation,path='',kind='file') {
  const workspace=localStorage.getItem('ia-local-zero-workspace'), dialog=document.getElementById('workspace-entry-dialog');
  if(!workspace || !dialog) return notice('Selecione um projeto primeiro.');
  if(sending || liveActions.size) return notice('Aguarde a operação atual terminar.');
  workspaceEntryDialogState={operation,workspace,directory:dockDirectoryPath,path,kind,pending:false};
  const removing=operation==='trash', folder=operation==='create_directory';
  document.getElementById('workspace-entry-title').textContent=removing?'Excluir item':folder?'Nova pasta':'Novo arquivo';
  document.getElementById('workspace-entry-location').textContent=`Projeto: ${workspace}`;
  const field=document.getElementById('workspace-entry-path');
  field.value=removing?path:dockDirectoryPath?dockDirectoryPath+'/':''; field.readOnly=removing;
  document.getElementById('workspace-entry-content-label').hidden=operation!=='create_file';
  document.getElementById('workspace-entry-content').value='';
  const closingOpenFile=openFileWorkspaceRoot===workspace && openFilePath && (openFilePath===path || openFilePath.startsWith(path+'/'));
  const dirty=closingOpenFile && document.getElementById('stage-editor')?.value!==originalFileContent;
  document.getElementById('workspace-entry-note').textContent=removing
    ? `O ${kind==='directory'?'diretório e seu conteúdo serão movidos':'arquivo será movido'} para a lixeira local. Use Restaurar para recuperá-lo.${dirty?' As alterações não salvas no editor serão descartadas.':''}`
    : 'O item será criado no caminho informado, sem substituir arquivos existentes.';
  document.getElementById('workspace-entry-error').textContent='';
  document.getElementById('workspace-entry-submit').textContent=removing?'Mover para lixeira':folder?'Criar pasta':'Criar arquivo';
  dialog.showModal(); field.focus(); if(!removing) field.setSelectionRange(field.value.length,field.value.length);
}
function closeWorkspaceEntryDialog() {
  if(workspaceEntryDialogState?.pending) return;
  document.getElementById('workspace-entry-dialog')?.close(); workspaceEntryDialogState=null;
}
async function submitWorkspaceEntry(event) {
  event.preventDefault(); const state=workspaceEntryDialogState;
  if(!state || state.pending) return;
  const error=document.getElementById('workspace-entry-error');
  if(localStorage.getItem('ia-local-zero-workspace')!==state.workspace) { error.textContent='O projeto ativo mudou. Feche esta janela e tente novamente.'; return; }
  const path=state.operation==='trash'?state.path:document.getElementById('workspace-entry-path').value.trim();
  if(!path || path.endsWith('/')) { error.textContent='Informe um nome de arquivo ou pasta.'; return; }
  const content=document.getElementById('workspace-entry-content').value;
  state.pending=true; error.textContent='';
  const buttons=[document.getElementById('workspace-entry-submit'),document.getElementById('workspace-entry-cancel')];
  buttons.forEach(button=>button.disabled=true);
  try {
    const result=await workspaceEntryRequest(state.operation,{path,...(state.operation==='create_file'?{content}:{})},state.workspace);
    if(!result.ok) { error.textContent=result.error || 'A operação não foi confirmada.'; return; }
    state.pending=false; closeWorkspaceEntryDialog();
    if(state.operation==='trash' && openFileWorkspaceRoot===state.workspace && openFilePath && (openFilePath===path || openFilePath.startsWith(path+'/'))) closeEditorFile(true);
    if(localStorage.getItem('ia-local-zero-workspace')===state.workspace) {
      const preserveEditor=state.operation==='create_file' && openFilePath && document.getElementById('stage-editor')?.value!==originalFileContent;
      if(state.operation==='create_file' && !preserveEditor) setEditorFile({path:result.data.path,content});
      await renderDockDirectory(state.directory);
      const status=document.getElementById('dock-status');
      if(status) status.textContent=state.operation==='trash'?`${path} movido para a lixeira local.`:`${path} criado.${preserveEditor?' Suas alterações no editor foram preservadas.':''}`;
    }
  } finally { state.pending=false; buttons.forEach(button=>button.disabled=false); }
}
async function restoreDockEntry() {
  if(sending || liveActions.size) return notice('Aguarde a operação atual terminar.');
  const workspace=localStorage.getItem('ia-local-zero-workspace'), item=dockTrashItems[0];
  if(!workspace || !item) return;
  const result=await workspaceEntryRequest('restore',{trash_id:item.trash_id},workspace);
  if(!result.ok) return notice(result.error || 'Não foi possível restaurar.');
  if(localStorage.getItem('ia-local-zero-workspace')===workspace) {
    await renderDockDirectory(dockDirectoryPath);
    const status=document.getElementById('dock-status'); if(status) status.textContent=`${result.data.path} restaurado.`;
  }
}
function toggleEditorPreview() {
  const editor=document.getElementById('stage-editor'), workarea=document.querySelector('.stage-workarea'), frame=document.getElementById('stage-preview');
  if(!editor || !workarea || !frame) return;
  const open=workarea.classList.toggle('preview-open');
  if(open) frame.srcdoc=editor.value;
}
function changePreviewZoom(delta) {
  previewZoom=Math.max(.6,Math.min(1.4,previewZoom+delta));
  const frame=document.getElementById('stage-preview'), label=document.getElementById('stage-zoom');
  if(frame) { const size=Math.round(420*previewZoom); frame.style.width=`${size}px`; frame.style.height=`${size}px`; }
  if(label) label.textContent=`${Math.round(previewZoom*100)}%`;
}
function reviewEditorChanges() {
  const editor=document.getElementById('stage-editor'); if(!editor || !openFilePath) return;
  const current=editor.value, old=originalFileContent;
  const artifact=makeCodeArtifact(openFilePath,old,current,'proposed');
  assistantReply(`Alterações pendentes em ${openFilePath}. Revise o diff estruturado abaixo antes de salvar.`,'Revisão local',[],null,artifact);
}
function discardEditorChanges() {
  const editor=document.getElementById('stage-editor'); if(!editor) return;
  editor.value=originalFileContent; editor.dispatchEvent(new Event('input'));
  assistantReply(`Descartei as alterações pendentes em ${openFilePath}.`,'Alterações descartadas');
}
async function saveEditorFile() {
  const editor=document.getElementById('stage-editor');
  if(!editor || !openFilePath || editor.value===originalFileContent) return notice('Nenhuma alteração pendente neste arquivo.');
  const workspace=openFileWorkspaceRoot, path=openFilePath;
  if(!workspace || workspace!==localStorage.getItem('ia-local-zero-workspace')) return notice('Selecione o projeto deste arquivo antes de salvar.');
  if(!(await confirmMutation('Salvar alterações?',openFilePath))) return;
  if(path!==openFilePath || workspace!==localStorage.getItem('ia-local-zero-workspace')) return notice('O arquivo ou projeto ativo mudou. Revise antes de salvar.');
  const before=originalFileContent, savedContent=editor.value;
  const result=await workspaceEntryRequest('edit_file',{path,old_text:before,new_text:savedContent},workspace);
  if(!result.ok) return assistantReply(result.error,'Arquivo não salvo');
  assistantReply(`Salvei ${path}.`,`Arquivo salvo · ${result.elapsed_ms} ms`,[],null,result.data?.artifact || makeCodeArtifact(path,before,savedContent,'applied'));
  if(openFilePath===path && openFileWorkspaceRoot===workspace) { originalFileContent=savedContent; editor.dispatchEvent(new Event('input')); }
  if(workspace!==localStorage.getItem('ia-local-zero-workspace')) return;
  const checks=await toolRequest('project_checks',{check:'auto',path,_expected_workspace:workspace},false);
  const status=document.getElementById('dock-status');
  if(status) status.textContent=checks.ok && checks.data?.executed===false ? 'Arquivo salvo. Este projeto não possui verificações configuradas.' : checks.ok && checks.data?.passed===true ? 'Arquivo salvo. Verificações passaram.' : 'Arquivo salvo. Verificação pendente ou com falha.';
}
function setMode(mode) {
  // Chat and Training are the only top-level destinations. Project files and
  // tools stay available as contextual work inside Chat.
  const value=mode==='training'?'training':'chat';
  localStorage.setItem('ia-local-zero-mode',value);
  document.body.classList.toggle('chat-mode',value==='chat');
  document.body.classList.toggle('workspace-mode',value==='workspace');
  document.body.classList.toggle('training-mode',value==='training');
  document.body.classList.remove('chat-context-open');
  if(value==='training') closeEditorFile();
  document.getElementById('mode-chat')?.classList.toggle('active',value==='chat');
  document.getElementById('mode-training')?.classList.toggle('active',value==='training');
  const subtitle=document.getElementById('mode-subtitle');
  if(subtitle) subtitle.textContent=value==='training'?'Aprendizado verificável':'Ideias em movimento';
  const dockTitle=document.getElementById('dock-title');
  if(dockTitle) dockTitle.textContent=value==='chat'?'CONTEXTO':'CONHECIMENTO';
  const contextToggle=document.getElementById('context-panel-toggle');
  if(contextToggle) {
    contextToggle.hidden=value!=='chat';
    if(value==='chat') contextToggle.setAttribute('aria-expanded',
      String(!window.matchMedia('(max-width: 1050px)').matches && !document.body.classList.contains('workspace-explorer-collapsed')));
  }
  const homePath=document.getElementById('workspace-home-path');
  if(homePath) homePath.textContent=localStorage.getItem('ia-local-zero-workspace') || 'Nenhum projeto selecionado';
  if(value==='training') loadTraining();
}
async function loadAutonomousLearning() {
  const target=document.getElementById('autonomous-summary-content');
  if(!target) return;
  target.replaceChildren(node('span','autonomous-summary-state','Consultando próximo ciclo…'));
  try {
    const response=await fetch('/api/v1/learning/autonomous',{cache:'no-store'});
    const data=await response.json();
    if(!response.ok || !data.ok) throw new Error(data.error || 'estado indisponível');
    const selected=data.next?.selected||{};
    const active=data.control?.active;
    const history=Array.isArray(data.control?.history)?data.control.history.slice(-3).reverse():[];
    const state=node('span','autonomous-summary-state',active ? 'Ciclo em andamento' : 'Pronto para continuar');
    const title=node('strong',null,(selected.label||selected.topic||'Nenhuma trilha selecionada'));
    const detail=node('small',null,active
      ? 'Executando '+(active.topic||selected.topic||'trilha')+' · ciclo '+(active.cycle_id||'ativo')
      : 'Próximo foco: '+(selected.topic||'a definir')+' · '+Math.round(Number(selected.progress||0)*100)+'% dos critérios do laboratório');
    const reason=node('small',null,active ? 'O worker está pesquisando, praticando e registrando evidências.' : (selected.blocker||selected.reason||'lacunas ainda não fechadas'));
    target.replaceChildren(state,title,detail,reason);
    if(history.length) {
      const historyBox=node('div','autonomous-summary-history');
      historyBox.append(node('small',null,'Últimos ciclos'));
      for(const item of history) {
        const outcome=item.outcome||{};
        const progress=outcome.skill_progress!=null ? ' · laboratório '+Math.round(Number(outcome.skill_progress||0)*100)+'%' : '';
        historyBox.append(node('span',null,(outcome.topic||item.cycle_id||'ciclo')+' · '+(outcome.status||'registrado')+progress));
      }
      target.append(historyBox);
    }
    const audit=Array.isArray(data.audit)?data.audit.slice(-6).reverse():[];
    if(audit.length) {
      const details=document.createElement('details');
      details.className='autonomous-audit';
      details.append(node('summary',null,'Atividade recente · '+audit.length+' eventos'));
      const auditList=node('div','autonomous-audit-list');
      for(const item of audit) {
        const outcome=item.outcome||{};
        const metric=item.passed!=null ? ' · '+item.passed+'/'+(item.total||'?')+' aprovadas' : '';
        const event=node('div','autonomous-audit-event',(item.event||'evento')+' · '+(item.topic||outcome.topic||item.cycle_id||'professor')+metric);
        auditList.append(event);
        for(const log of (outcome.logs||[]).slice(-3)) {
          auditList.append(node('span','autonomous-audit-log',log.message||String(log)));
        }
      }
      details.append(auditList);
      target.append(details);
    }
    if(!active) {
      const run=node('button','autonomous-run','Executar próximo ciclo');
      run.type='button';
      run.onclick=runAutonomousCycle;
      target.append(run);
    }
  } catch(error) {
    target.replaceChildren(node('span','autonomous-summary-state','Estado indisponível: '+error.message));
  }
}
async function runAutonomousCycle() {
  const target=document.getElementById('autonomous-summary-content');
  if(target) target.replaceChildren(node('span','autonomous-summary-state','Roteando ciclo de aprendizado…'));
  try {
    let tickApi={method:'POST',path:'/api/v1/learning/autonomous/tick'};
    let routeMessage='Skill: Ciclo autônomo de aprendizado';
    try {
      const task='Execute o próximo ciclo de aprendizado autônomo';
      const routeResponse=await fetch('/api/v1/skills/route',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({task})});
      const route=await routeResponse.json();
      const skill=(route.selected_skills||[]).find(item=>item.id==='autonomous-learning-cycle');
      const api=(route.api_candidates||[]).find(item=>item.id==='learning.autonomous.tick');
      if(routeResponse.ok && route.ok && skill && api) {
        tickApi={method:api.method,path:api.path};
        routeMessage=`${skill.label} · ${api.method} ${api.path} · ${api.requires_approval?'iniciado pelo seu comando explícito':'sem aprovação extra'}`;
      }
    } catch (_) { /* O botão é explícito; segue pelo endpoint local conhecido se o preview falhar. */ }
    if(target) target.replaceChildren(node('span','autonomous-summary-state',routeMessage));
    const response=await fetch(tickApi.path,{method:tickApi.method});
    const data=await response.json();
    if(!response.ok || !data.ok) throw new Error(data.error || 'não foi possível iniciar o ciclo');
    const messages={
      started:'Ciclo iniciado para '+(data.plan?.selected?.topic||data.job?.topic||'a próxima trilha')+'.',
      running:'Já existe um ciclo em execução.',
      finished:'Ciclo observado e encerrado; o estado foi atualizado.',
      cooldown:'O limite diário foi atingido; a próxima execução ficará para depois.'
    };
    notice(messages[data.status]||('Professor autônomo: '+(data.status||'estado atualizado')));
    await loadAutonomousLearning();
    if(document.body.classList.contains('training-mode')) await loadTraining();
  } catch(error) {
    notice('Professor autônomo: '+error.message);
    await loadAutonomousLearning();
  }
}
async function loadWorkflowReviewSummary() {
  const button=document.getElementById('workflow-review-open');
  if(!button) return;
  try {
    const response=await fetch('/api/workflow-review?offset=0&limit=1&status=pending',{cache:'no-store'});
    const data=await response.json();
    if(!response.ok || !data.ok) throw new Error(data.error||'fila indisponível');
    const counts=data.counts||{};
    button.textContent=`Revisar trajetórias · ${Number(counts.pending||0)} pendentes`;
    button.title=`${Number(counts.total||0)} coletadas · ${Number(counts.approved||0)} aprovadas · ${Number(counts.rejected||0)} rejeitadas`;
    button.disabled=Number(counts.pending||0)===0;
  } catch(error) {
    button.textContent='Revisão de trajetórias indisponível';
    button.title=error.message;
    button.disabled=true;
  }
}
function openWorkflowReview() {
  const overlay=node('div','code-overlay');
  overlay.setAttribute('role','dialog'); overlay.setAttribute('aria-modal','true'); overlay.setAttribute('aria-labelledby','workflow-review-title');
  const card=node('section','code-overlay-card');
  const head=node('div','code-overlay-head'), title=node('div');
  title.append(node('div','code-overlay-title','Curadoria de trajetórias'),node('div','code-overlay-subtitle','Revise o contexto e a próxima ação sugerida pelo planejador.'));
  title.firstChild.id='workflow-review-title';
  const headActions=node('div','code-overlay-actions'), close=node('button',null,'Fechar'); close.type='button';
  headActions.append(close); head.append(title,headActions);
  const countsLine=node('div','workflow-review-counts','Carregando contagens…');
  const body=node('div','workflow-review-body');
  const footer=node('div','workflow-review-footer');
  const reviewer=document.createElement('input'); reviewer.type='text'; reviewer.maxLength=100; reviewer.autocomplete='name'; reviewer.placeholder='Revisor'; reviewer.setAttribute('aria-label','Identificador do revisor');
  const rationale=document.createElement('textarea'); rationale.maxLength=1000; rationale.placeholder='Justifique a decisão (mínimo 12 caracteres)'; rationale.setAttribute('aria-label','Justificativa da decisão');
  const nav=node('div','workflow-review-actions'), navGroup=node('div','workflow-review-actions-group'), decisionGroup=node('div','workflow-review-actions-group');
  const prev=node('button',null,'← Anterior'), next=node('button',null,'Próxima →'); prev.type=next.type='button';
  const reject=node('button','workflow-review-reject','Rejeitar'), approve=node('button','workflow-review-approve','Aprovar'); reject.type=approve.type='button';
  navGroup.append(prev,next); decisionGroup.append(reject,approve); nav.append(countsLine,navGroup,decisionGroup);
  footer.append(reviewer,rationale,nav);
  card.append(head,body,footer); overlay.append(card);
  const remove=()=>{ overlay.remove(); document.getElementById('workflow-review-open')?.focus(); };
  close.onclick=remove; overlay.onclick=event=>{if(event.target===overlay) remove();};
  overlay.addEventListener('keydown',event=>{if(event.key==='Escape') remove();});
  prev.onclick=()=>showCandidate(Math.max(0,workflowReviewOffset-1));
  next.onclick=()=>showCandidate(workflowReviewOffset+1);
  reject.onclick=()=>submitDecision('rejected'); approve.onclick=()=>submitDecision('approved');
  document.body.append(overlay);

  function section(label,value) {
    const box=node('section','workflow-review-section'); box.append(node('h3',null,label));
    const pre=node('pre');
    pre.textContent=typeof value==='string' ? (value || '—') : JSON.stringify(value ?? [],null,2);
    box.append(pre); return box;
  }
  function setDecisionEnabled(enabled) {
    reviewer.disabled=!enabled; rationale.disabled=!enabled;
    approve.disabled=!enabled; reject.disabled=!enabled;
  }
  async function showCandidate(offset=0) {
    workflowReviewOffset=Math.max(0,offset);
    body.replaceChildren(node('div','history-empty','Carregando trajetória…'));
    setDecisionEnabled(false); prev.disabled=true; next.disabled=true;
    try {
      const response=await fetch(`/api/workflow-review?offset=${workflowReviewOffset}&limit=1&status=pending`,{cache:'no-store'});
      const data=await response.json();
      if(!response.ok || !data.ok) throw new Error(data.error||'não foi possível ler a fila');
      const counts=data.counts||{}, item=data.items?.[0];
      countsLine.textContent=`${Number(counts.pending||0)} pendentes · ${Number(counts.approved||0)} aprovadas · ${Number(counts.rejected||0)} rejeitadas · ${Number(counts.total||0)} no total`;
      await loadWorkflowReviewSummary();
      if(!item) {
        body.replaceChildren(node('div','history-empty',Number(counts.pending||0)===0?'Não há trajetórias pendentes de revisão.':'Não há mais itens nesta posição da fila.'));
        setDecisionEnabled(false); prev.disabled=workflowReviewOffset===0; next.disabled=true;
        return;
      }
      const metadata=node('div','workflow-review-counts',`${item.candidate_id||'sem identificador'} · ${item.trajectory_status||'estado não registrado'} · backend ${item.trajectory_backend||'não registrado'}`);
      body.replaceChildren(metadata,node('h2','workflow-review-objective',item.objective||'Objetivo ausente'),section(`Ação proposta · ${item.target?.name||'sem ação'}`,item.target?.arguments||{}));
      body.append(section('Ações anteriores',item.previous_actions||[]),section('Observações e resultados anteriores',item.observations||[]),section('Ciclo completo · auditoria posterior, fora do contexto de treino',item.trajectory_review||{available:false}),section('Ferramentas que estavam disponíveis',item.available_tool_candidates||[]));
      setDecisionEnabled(true); prev.disabled=workflowReviewOffset===0; next.disabled=workflowReviewOffset+1>=Number(counts.pending||0);
    } catch(error) {
      body.replaceChildren(node('div','history-empty',`Erro ao carregar a revisão: ${error.message}`));
      countsLine.textContent='Fila de curadoria indisponível';
      setDecisionEnabled(false); prev.disabled=true; next.disabled=true;
    }
  }
  async function submitDecision(review_status) {
    if(!reviewer.value.trim() || rationale.value.trim().length<12) {
      notice('Informe quem revisou e uma justificativa de pelo menos 12 caracteres.'); return;
    }
    const candidate=body.querySelector('.workflow-review-counts')?.textContent.split(' · ')[0];
    const titleText=body.querySelector('.workflow-review-objective')?.textContent||'esta trajetória';
    if(review_status==='approved' && !confirm(`Aprovar “${titleText.slice(0,180)}” para o dataset curado? Isso não inicia treino.`)) return;
    setDecisionEnabled(false); prev.disabled=true; next.disabled=true;
    try {
      const response=await fetch('/api/workflow-review/decision',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({candidate_id:candidate,review_status,reviewer:reviewer.value.trim(),rationale:rationale.value.trim()})});
      const data=await response.json();
      if(!response.ok || !data.ok) throw new Error(data.error||'decisão não registrada');
      rationale.value='';
      notice(review_status==='approved'?'Trajetória aprovada e incluída no snapshot curado.':'Trajetória rejeitada e mantida fora do treino.');
      await showCandidate(workflowReviewOffset);
    } catch(error) {
      notice('Não foi possível registrar a decisão: '+error.message);
      setDecisionEnabled(true); prev.disabled=workflowReviewOffset===0;
    }
  }
  showCandidate(0); close.focus();
}
async function loadTraining() {
  const grid=document.getElementById('training-grid'); if(!grid) return;
  loadAutonomousLearning();
  loadWorkflowReviewSummary();
  grid.replaceChildren(node('div','training-card','Atualizando competências…'));
  try {
    const data=await fetch('/api/skills',{cache:'no-store'}).then(r=>r.json());
    const canonicalSkillTopic=(topic)=>{
      const value=String(topic||''), lower=value.toLowerCase();
      const aliases=[['doc.rust-lang.org','Rust'],['rustlings.rust-lang.org','Rust'],['tokio-rs/axum','Rust'],['ziglang.org','Zig'],['typescriptlang.org','TypeScript'],['nodejs.org','Node.js'],['learn.microsoft.com/dotnet/csharp','C#'],['dev.java','Java'],['learncpp.com','C++'],['python.org','Python'],['thealgorithms/python','Python'],['sqlbolt.com','SQL'],['cheatsheetseries.owasp.org','OWASP']];
      const match=aliases.find(([needle])=>lower.includes(needle));
      return match ? match[1] : value;
    };
    const grouped=new Map();
    for(const skill of Object.values(data.skills||{})) {
      const key=canonicalSkillTopic(skill.topic);
      const current=grouped.get(key);
      if(!current) { grouped.set(key,{...skill,topic:key}); continue; }
      const currentScore=Number(current.evaluation?.progress||current.confidence||0), incomingScore=Number(skill.evaluation?.progress||skill.confidence||0);
      const primary=incomingScore>currentScore?{...skill,topic:key}:current;
      const sources=[...(current.evidence?.sources||[]),...(skill.evidence?.sources||[])];
      primary.evidence={...(primary.evidence||{}),sources:[...new Map(sources.filter(item=>item?.url).map(item=>[item.url,item])).values()],documents:Math.max(Number(current.evidence?.documents||0),Number(skill.evidence?.documents||0)),independent_hosts:Math.max(Number(current.evidence?.independent_hosts||0),Number(skill.evidence?.independent_hosts||0))};
      primary.technologies=[...new Set([...(current.technologies||[]),...(skill.technologies||[])])];
      primary.source_repositories=[...new Set([...(current.source_repositories||[]),...(skill.source_repositories||[]),...(current.source_repository?[current.source_repository]:[]),...(skill.source_repository?[skill.source_repository]:[])])];
      if(primary.source_repositories.length) primary.source_repository=primary.source_repositories[0];
      const verified=[...(current.practice?.verified||[]),...(skill.practice?.verified||[])];
      if(verified.length) {
        const unique=[...new Map(verified.map(item=>[`${item.level||''}:${item.task||item.name||JSON.stringify(item)}`,item])).values()];
        primary.practice={...(primary.practice||{}),verified:unique,passed:unique.filter(item=>item.passed!==false && item.status!=='failed').length,tasks:Math.max(Number(current.practice?.tasks||0),Number(skill.practice?.tasks||0),unique.length),failed:Math.max(Number(current.practice?.failed||0),Number(skill.practice?.failed||0))};
      }
      const labVerified=[...(current.laboratory_checks?.verified||[]),...(skill.laboratory_checks?.verified||[])];
      if(labVerified.length) {
        const unique=[...new Map(labVerified.map(item=>[`${item.level||''}:${item.task||item.name||JSON.stringify(item)}`,item])).values()];
        primary.laboratory_checks={...(primary.laboratory_checks||{}),verified:unique,passed:unique.length,tasks:Math.max(Number(current.laboratory_checks?.tasks||0),Number(skill.laboratory_checks?.tasks||0),unique.length),failed:Math.max(Number(current.laboratory_checks?.failed||0),Number(skill.laboratory_checks?.failed||0))};
      }
      const covered=[...(current.concepts?.covered||[]),...(skill.concepts?.covered||[])];
      const missing=[...(current.concepts?.gaps||[]),...(skill.concepts?.gaps||[])];
      const legacyCoverage=[...(current.concepts?.legacy_reference_lab_coverage||[]),...(skill.concepts?.legacy_reference_lab_coverage||[])];
      primary.concepts={...(primary.concepts||{}),covered:[...new Set(covered)],gaps:[...new Set(missing)],legacy_reference_lab_coverage:[...new Set(legacyCoverage)]};
      grouped.set(key,primary);
    }
    const skills=[...grouped.values()];
    const summary=document.getElementById('training-summary-list');
    const scoreFor=(skill)=>{ const progress=skill.evaluation?.progress; return progress!==null && progress!==undefined && Number.isFinite(Number(progress)) ? Math.min(100,Math.max(0,Math.round(Number(progress)*100))) : null; };
    if(summary) {
      summary.replaceChildren(...(skills.length ? skills.map(skill=>{ const score=scoreFor(skill); const item=node('div','training-summary-item'); item.title=`${skill.topic||'Competência sem nome'} · ${score===null?'sem avaliação':`${score}% dos critérios do laboratório`}`; item.append(node('span','training-summary-name',skill.topic||'Competência sem nome'),node('span','training-summary-score',score===null?'—':`${score}%`)); const meter=node('div','training-summary-meter'); meter.append(Object.assign(document.createElement('i'),{style:`width:${score??0}%`})); item.append(meter); return item; }) : [node('div','history-empty','Nenhuma competência registrada')]));
    }
    if(!skills.length) { grid.replaceChildren(node('div','training-card','Nenhuma competência foi iniciada. Use “Aprenda <assunto>” no chat.')); return; }
    grid.replaceChildren(...skills.map(skill=>{
      const practice=skill.practice||{}, labChecks=skill.laboratory_checks||{}, evidence=skill.evidence||{}, gaps=skill.concepts?.gaps||[], evaluation=skill.evaluation||{};
      const score=scoreFor(skill);
      const unclassified=Number(practice.unclassified_legacy?.length||0)+Number(practice.legacy_unclassified_counts?.passed||0);
      const statusLabels={mastered:'Critérios do laboratório concluídos',partially_known:'Em desenvolvimento',evidence_gap:'Evidências incompletas',discovered:'Registrada'};
      const card=node('article','training-card'); const head=node('div','training-card-head'); const title=node('div',null); title.append(node('div','training-status',statusLabels[skill.status]||skill.status||'em avaliação'),node('h3',null,skill.topic||'Competência sem nome')); const remove=node('button','training-delete','×'); remove.title='Excluir competência'; remove.onclick=()=>deleteTraining(skill.topic); head.append(title,remove); card.append(head,node('div','training-meter')); card.querySelector('.training-meter').append(Object.assign(document.createElement('i'),{style:`width:${score??0}%`}));
      card.append(node('div','training-meta',`${score===null?'Sem avaliação do laboratório':`Progresso dos critérios do laboratório ${score}%`} · Confiança ${Math.round(Number(skill.confidence||0)*100)}% · ${practice.passed||0}/${practice.tasks||0} tarefas do agente aprovadas · ${labChecks.passed||0}/${labChecks.tasks||0} exercícios fixos aprovados · ${evidence.independent_hosts||0} fontes independentes`));
      card.append(node('div','training-evidence',`Evidências: ${evidence.documents||0} documento(s) · falhas do agente ${practice.failed||0} · falhas do laboratório ${labChecks.failed||0}`));
      if(unclassified) card.append(node('div','training-evidence',`${unclassified} registro(s) antigo(s) aguardam confirmação de origem e não contam como desempenho.`));
      if(skill.source_repository) card.append(node('div','training-evidence',`Repositório: ${skill.source_repository}`));
      if(skill.technologies?.length) card.append(node('div','training-evidence',`Tecnologias: ${skill.technologies.join(', ')}`));
      const levels=skill.curriculum?.levels||[]; const objectives=levels.reduce((total,level)=>total+(level.concepts||[]).length,0);
      if(objectives) card.append(node('div','training-evidence',`Trilha: ${levels.length} níveis · ${objectives} objetivos a comprovar`));
      if(evaluation.coverage) card.append(node('div','training-evidence',`Cobertura: ${evaluation.coverage.covered||0}/${evaluation.coverage.total||0} conceitos comprovados`));
      const missingCriteria={sources:'fontes independentes',levels:'níveis',coverage:'cobertura',practice:'prática',transfer:'transferência',integration:'integração'};
      const missing=Object.entries(evaluation.criteria||{}).filter(([,ok])=>!ok).map(([key])=>missingCriteria[key]||key);
      if(missing.length) card.append(node('div','training-evidence',`Ainda falta: ${missing.join(', ')}`));
      card.append(node('div','training-gaps',`Lacunas: ${gaps.slice(0,5).join(', ')||'nenhuma registrada'}`)); return card;
    }));
  } catch(error) { grid.replaceChildren(node('div','training-card',`Não foi possível carregar o estado: ${error.message}`)); }
}
async function deleteTraining(topic) {
  if(!confirm(`Excluir todo o estado de treinamento de “${topic}”?\n\nIsso remove fontes de competência, práticas, lacunas e evidências registradas para esse tema.`)) return;
  const response=await fetch(`/api/skills/${encodeURIComponent(topic)}`,{method:'DELETE'});
  const data=await response.json();
  if(!data.ok) return notice(data.error || 'Não foi possível excluir a competência.');
  await loadTraining();
  setLiveStatus({message:`Competência “${topic}” excluída.`,phase:'complete',done:true});
}
async function clearTraining() {
  if(!confirm('Excluir todo o estado de treinamento?\n\nIsso remove todas as competências, fontes, práticas, lacunas e evidências persistidas.')) return;
  const response=await fetch('/api/skills',{method:'DELETE'}); const data=await response.json();
  if(!data.ok) return notice(data.error || 'Não foi possível limpar o treinamento.');
  await loadTraining();
  setLiveStatus({message:`Treinamento limpo: ${data.removed||0} competência(s) removida(s).`,phase:'complete',done:true});
}
function isActionTerminal(event) {
  // AgentCore statuses describe individual steps as well as the whole task.
  if (event.task_id || (typeof event.kind === 'string' && event.kind.includes('.'))) {
    return event.phase === 'complete';
  }
  return event.done === true || ['completed','blocked','failed','cancelled'].includes(event.status);
}
function actionState(event, terminal) {
  if (!terminal) return 'busy';
  if (event.status === 'blocked' || event.phase === 'blocked') return 'blocked';
  if (event.status === 'cancelled' || event.phase === 'cancelled') return 'cancelled';
  if (event.status === 'failed' || event.phase === 'error') return 'error';
  return terminal ? 'done' : 'busy';
}
const actionStateLabels = {busy:'Em andamento', done:'Concluído', blocked:'Bloqueado', error:'Falhou', cancelled:'Cancelado'};
const actionStateIcons = {busy:'✦', done:'✓', blocked:'!', error:'!', cancelled:'−'};
function actionStateHint(state) {
  return {busy:'Acompanhe a atividade atual abaixo.', done:'Atividade encerrada. Confira o resultado na resposta.',
    blocked:'A tarefa não foi concluída. Consulte o motivo abaixo.',
    error:'A operação falhou. Consulte o erro abaixo.', cancelled:'A execução foi cancelada.'}[state] || '';
}
function setLiveStatus(event) {
  const status = document.getElementById('live-status');
  const text = document.getElementById('live-status-text');
  if (!status || !text) return;
  status.className = 'live-status ' + actionState(event, event.done);
  status.setAttribute('role','status');
  status.setAttribute('aria-live','polite');
  const confidence = event.confidence != null ? ` · conf. ${ChatCore.confidenceText(event.confidence)}` : '';
  const nextText = event.message + (event.elapsed_ms != null ? ` · ${event.elapsed_ms} ms` : '') + confidence;
  if (text.textContent !== nextText) animateChatChange(text);
  text.textContent = nextText;
}
function notice(message) { setLiveStatus({message, phase:'error', done:true}); }
function requestId(prefix) { return `${prefix}-${crypto.randomUUID()}`; }
function animateChatChange(element) {
  const preference = window.matchMedia?.('(prefers-reduced-motion: reduce)');
  if (!element?.animate || preference?.matches) return;
  element.animate([{opacity:.35, transform:'translateY(2px)'},{opacity:1, transform:'translateY(0)'}], {duration:180,easing:'ease-out'});
}
function refreshControls() {
  const reading = attachments.some(item => item.state === 'reading');
  document.querySelector('.send').disabled = sending || reading || liveActions.size > 0;
  document.querySelector('.send').title = reading ? 'Aguarde a leitura dos anexos'
    : sending || liveActions.size > 0 ? 'Aguarde a tarefa em andamento' : 'Enviar mensagem (Enter)';
  document.querySelector('.new-chat').disabled = sending || liveActions.size > 0;
  input.setAttribute('aria-busy', String(sending || reading));
}
function appendActionSteps(details, steps) {
  if (!details) return;
  if (!details.querySelector('summary')) details.append(node('summary'));
  details.querySelector('summary').textContent = `Detalhes da execução · ${(steps || []).length} registros`;
  const rendered = Number(details.dataset.renderedSteps || 0);
  for (const step of (steps || []).slice(rendered)) details.append(node('div','action-log action-step',step));
  details.dataset.renderedSteps = String((steps || []).length);
}
function actionProgressValue(value) {
  if (value == null || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0,Math.min(100,number)) : null;
}
function renderAction(record, entering = false) {
  const row = node('div', `action-message ${record.state || 'busy'}`);
  if (entering) {
    row.classList.add('entering');
    row.addEventListener('animationend',event=>{
      if (event.target === row && event.animationName === 'action-enter') row.classList.remove('entering');
    },{once:true});
  }
  row.append(node('div','action-icon',actionStateIcons[record.state || 'busy']));
  const body = node('div','action-body');
  body.append(node('span','action-state-label',actionStateLabels[record.state || 'busy']),
    node('div','action-state-hint',actionStateHint(record.state || 'busy')));
  body.append(node('div','action-title',record.title), node('div','action-log',record.log));
  const progress = node('div','action-progress');
  progress.setAttribute('aria-hidden','true');
  progress.append(node('span'));
  const value = actionProgressValue(record.progress);
  progress.dataset.mode = record.state === 'busy' ? (value == null ? 'indeterminate' : 'determinate') : record.state;
  if (value != null) progress.style.setProperty('--action-progress',`${value}%`);
  else if (record.state === 'done') progress.style.setProperty('--action-progress','100%');
  body.append(progress);
  const details = node('details','action-steps'); appendActionSteps(details,record.steps);
  body.append(details); row.append(body); chat.append(row); return row;
}
function startAction(operation, title, showInChat) {
  const record = {kind:'action', title, state:'busy', log:'Enviando requisição…', steps:['Enviando requisição…']};
  let row = null;
  if (showInChat) { timeline.push(record); row = renderAction(record,true); saveConversation(); scrollChat(); }
  liveActions.set(operation, {record, row});
  setLiveStatus({message:title, done:false}); refreshControls(); pollActivity();
}
function updateAction(event, terminal = false) {
  const action = liveActions.get(event.operation); if (!action) return;
  const {record, row} = action;
  const eventTerminal = terminal || isActionTerminal(event);
  const log = event.message + (event.elapsed_ms != null ? ` · ${event.elapsed_ms} ms` : '');
  const previousLog = record.log;
  if (record.log !== log) record.steps.push(log);
  // The HTTP response settles this operation. Delayed activity remains evidence,
  // but cannot turn its final status back into a running task.
  if (record.settled && !terminal) {
    if (row) appendActionSteps(row.querySelector('.action-steps'), record.steps);
    return;
  }
  if (terminal) record.settled = true;
  record.log = log;
  const progress = event.progress && typeof event.progress === 'object' ? event.progress.value : event.progress;
  if (progress != null) record.progress=actionProgressValue(progress);
  record.state = actionState(event, eventTerminal);
  if (row) {
    for (const state of ['busy','done','error','blocked','cancelled']) row.classList.toggle(state,record.state === state);
    row.querySelector('.action-state-label').textContent = actionStateLabels[record.state];
    row.querySelector('.action-state-hint').textContent = actionStateHint(record.state);
    const logElement=row.querySelector('.action-body > .action-log');
    if (logElement) {
      logElement.textContent = log;
      if (previousLog !== log) animateChatChange(logElement);
    }
    const progressBar=row.querySelector('.action-progress');
    if (progressBar) {
      if (eventTerminal) {
        progressBar.dataset.mode = record.state;
        if (record.state === 'done') progressBar.style.setProperty('--action-progress','100%');
        else if (record.progress != null) progressBar.style.setProperty('--action-progress',`${record.progress}%`);
      } else if (record.progress != null) {
        progressBar.dataset.mode = 'determinate';
        progressBar.style.setProperty('--action-progress',`${record.progress}%`);
      }
    }
    row.querySelector('.action-icon').textContent = actionStateIcons[record.state];
    const details = row.querySelector('.action-steps') || row.querySelector('details');
    appendActionSteps(details,record.steps);
  }
  setLiveStatus({...event, done:eventTerminal});
}
function recordWorkflow(operation, workflow) {
  if (!workflow || !Array.isArray(workflow.stages)) return;
  const action = liveActions.get(operation); if (!action) return;
  for (const stage of workflow.stages) {
    const label = workflowLabels[stage] || stage;
    if (!action.record.steps.some(step => step === label)) action.record.steps.push(label);
  }
  if (action.row) {
    const details = action.row.querySelector('.action-steps') || action.row.querySelector('details');
    appendActionSteps(details,action.record.steps);
  }
}
async function pollActivity() {
  if (polling || !liveActions.size) return;
  polling = true;
  try {
    const response = await fetch(`/api/events?after_seq=${activityCursor}`, {cache:'no-store', signal:AbortSignal.timeout(2500)});
    if (response.ok) {
      const data = await response.json();
      if (data.latest_seq < activityCursor) activityCursor = 0;
      for (const event of data.events || []) {
        activityCursor = Math.max(activityCursor, event.seq || event.id || 0);
        if (consumeAgentAnswerEvent(event)) continue;
        const progress = event.progress && typeof event.progress === 'object' ? event.progress.value : event.progress;
        updateAction({
          ...event,
          id: event.seq || event.id,
          operation: event.task_id || event.operation,
          message: event.detail || event.title || event.message,
          progress,
          done: isActionTerminal(event),
          phase: event.phase,
        });
      }
    }
  } catch (_) { /* A resposta HTTP encerra a ação mesmo se o canal de eventos cair. */ }
  finally {
    polling = false;
    if (liveActions.size) {
      const streaming=Array.from(agentStreamDrafts.keys()).some(operation=>liveActions.has(operation));
      setTimeout(pollActivity,streaming?160:450);
    }
  }
}
function consumeAgentAnswerEvent(event) {
  const operation=event.operation || event.task_id;
  const message=String(event.message || event.detail || event.title || '');
  if(!operation) return false;
  if(message==='__ANSWER_RESET__') {
    const current=agentStreamDrafts.get(operation);
    if(current?.draft) {
      if(current.draft._streamFrame) {
        if(window.cancelAnimationFrame) window.cancelAnimationFrame(current.draft._streamFrame);
        else clearTimeout(current.draft._streamFrame);
        current.draft._streamFrame=0;
      }
      current.text=''; current.draft.bubble.replaceChildren();
      current.draft._streamText='';
      current.draft.meta.textContent='Rascunho ao vivo · reavaliando a resposta';
      scrollChat();
    }
    return true;
  }
  const prefix='__ANSWER_DELTA__ · ';
  if(!message.startsWith(prefix)) return false;
  // The event feed is shared by API clients and chats. Only the request that
  // registered a draft in this chat may append text to this conversation.
  const current=agentStreamDrafts.get(operation);
  if(!current || !liveActions.has(operation)) return true;
  let fragment;
  try { fragment=decodeURIComponent(message.slice(prefix.length)); }
  catch(_) { return true; }
  if(!current.draft) current.draft=createStreamingDraft();
  current.text+=fragment;
  updateStreamingDraft(current.draft,current.text);
  return true;
}
async function drainActivityEvents(operation) {
  for(let attempt=0;attempt<80 && polling;attempt++) await new Promise(resolve=>setTimeout(resolve,10));
  if(liveActions.has(operation)) {
    await pollActivity();
    for(let attempt=0;attempt<80 && polling;attempt++) await new Promise(resolve=>setTimeout(resolve,10));
  }
}
async function previewSkillRoute(task) {
  const operation=requestId('skill-route');
  const data=await requestWithActivity('/api/v1/skills/route',{task,context:workspaceRoutingContext()},operation,'Roteando tarefa por regras e contratos locais',true);
  if(!data.ok) return assistantReply(data.error || 'O roteador local não conseguiu analisar a tarefa.','Roteamento de skills indisponível');
  const skills=data.selected_skills || [];
  const apis=data.api_candidates || [];
  const skillText=skills.length
    ? skills.map(skill=>`- ${skill.label} · confiança inicial ${Math.round(skill.score*100)}%`).join('\n')
    : '- Nenhuma skill foi selecionada; esclarecer a tarefa antes de escolher.';
  const apiText=apis.length
    ? apis.slice(0,8).map(api=>`- ${api.id}${api.kind==='service-api'?' · '+api.method+' '+api.path:''} · ${api.network_policy}${api.requires_approval?' · requer aprovação':''}`).join('\n')
    : '- Nenhuma API candidata.';
  const clarification=data.needs_clarification ? '\n\nAinda há ambiguidade; o roteador recomenda pedir esclarecimento.' : '';
  const next=data.next_action || {};
  const nextText=next.type==='dispatch'
    ? `${next.capability_id}${next.tool?' · ferramenta '+next.tool:''}${next.method?' · '+next.method+' '+next.path:''}${next.required_inputs?.length?' · entradas obrigatórias: '+next.required_inputs.join(', '):''}${next.requires_approval?' · exige aprovação':''}`
    : next.type==='compose_response' ? `resposta local pela skill ${next.skill_id}` : 'pedir esclarecimento antes de despachar';
  assistantReply(`**Rota determinística:** ${data.routing_reason || 'seleção por regras locais'}\n\n**Skills selecionadas:**\n${skillText}\n\n**Próximo passo:** ${nextText}\n\n**APIs disponíveis para o fluxo:**\n${apiText}${apis.length>8?'\n- … e '+(apis.length-8)+' capacidades adicionais do contrato selecionado.':''}\n\nRede externa solicitada: ${data.network_requested?'sim':'não'}.${clarification}\n\nO comando /rotear entrega o plano; a execução passa pelo fluxo local e continua sujeita à validação do contrato e às aprovações exigidas.`, 'Rota de skills · plano local');
}
async function requestWithActivity(url, payload, operation, title, showInChat = false, retainActivity = false) {
  startAction(operation, title, showInChat);
  const started = performance.now();
  const controller = new AbortController();
  const timeoutMs = url === '/api/chat' || url === '/api/v1/agent/pursue' ? 900000
    : payload?.tool === 'project_checks' || (payload?.tool === 'terminal_run' && payload?.arguments?.operation === 'project_check') ? 65000
    : payload?.tool === 'research_web' ? 30000 : 10000;
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let data;
  try {
    const response = await fetch(url, {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(payload), signal:controller.signal});
    data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || `Falha HTTP ${response.status}`);
    recordWorkflow(operation, data.workflow);
    // O agente precisa ser observável. Mesmo respostas locais rápidas ganham
    // uma janela mínima para que o usuário veja a etapa e seus registros.
    const minimumVisibleMs = url === '/api/chat' ? 900 : 350;
    const remaining = minimumVisibleMs - (performance.now() - started);
    if (remaining > 0) await new Promise(resolve=>setTimeout(resolve, remaining));
    if (data.agent?.recovery) updateAction({operation, message:'Falha detectada; procurando uma alternativa verificável.', phase:'recovering'}, false);
    const confidence = data && data.confidence != null ? Number(data.confidence) : null;
    const awaitingAgentInput = data.awaitingClarification === true || data.awaitingApproval === true || data.status === 'clarifying';
    const agentBlocked = data.report?.status === 'blocked' || awaitingAgentInput;
    const activityMessage = data.awaitingClarification === true || data.status === 'clarifying'
      ? 'AgentCore aguardando esclarecimento.'
      : data.awaitingApproval === true ? 'AgentCore aguardando aprovação.'
        : agentBlocked ? 'AgentCore pausado; consulte o motivo.' : 'Concluído.';
    updateAction({operation, message:activityMessage, phase:agentBlocked ? 'blocked' : 'done', elapsed_ms:Math.round(performance.now()-started), confidence}, true);
  } catch (error) {
    const message = error.name === 'AbortError' ? `O limite de ${Math.round(timeoutMs/1000)} s foi atingido. Não houve confirmação de conclusão; confira o resultado antes de repetir uma alteração.` : `Não foi possível concluir: ${error.message}`;
    data = {ok:false, error:message};
    updateAction({operation, message, phase:'error', elapsed_ms:Math.round(performance.now()-started), confidence:null}, true);
  } finally {
    clearTimeout(timer);
    if(!retainActivity) { liveActions.delete(operation); refreshControls(); }
    saveConversation();
  }
  return data;
}
async function consumeChatStream(response, onDelta) {
  const reader=response.body?.getReader();
  if(!reader) throw new Error('O navegador não disponibilizou o fluxo da resposta.');
  const decoder=new TextDecoder();
  let buffer='', result=null;
  const dispatch=block=>{
    const data=block.split(/\r?\n/).filter(line=>line.startsWith('data:'))
      .map(line=>line.slice(5).trimStart()).join('\n');
    if(!data) return;
    const event=JSON.parse(data);
    if(event.type==='delta') onDelta(event.text || '');
    else if(event.type==='done') result=event.response;
    else if(event.type==='error') throw new Error(event.error || 'A geração não pôde ser concluída.');
  };
  while(true) {
    const {value,done}=await reader.read();
    buffer+=decoder.decode(value || new Uint8Array(),{stream:!done});
    let boundary;
    while((boundary=buffer.indexOf('\n\n'))>=0) {
      dispatch(buffer.slice(0,boundary)); buffer=buffer.slice(boundary+2);
    }
    if(done) break;
  }
  if(buffer.trim()) dispatch(buffer);
  if(!result) throw new Error('O fluxo terminou sem uma resposta final.');
  return result;
}
async function requestChatStream(payload, operation, title, onDelta) {
  startAction(operation,title,true);
  const started=performance.now();
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),900000);
  let data;
  try {
    const response=await fetch('/api/chat/stream',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({...payload,...workspaceRoutingContext()}),signal:controller.signal});
    if(!response.ok) {
      let detail=`Falha HTTP ${response.status}`;
      try { detail=(await response.json()).error || detail; } catch(_) {}
      throw new Error(detail);
    }
    data=await consumeChatStream(response,onDelta);
    if(!data.ok) throw new Error(data.error || 'A resposta não passou pela validação.');
    recordWorkflow(operation,data.workflow);
    const remaining=900-(performance.now()-started);
    if(remaining>0) await new Promise(resolve=>setTimeout(resolve,remaining));
    updateAction({operation,message:'Resposta concluída.',phase:'done',elapsed_ms:Math.round(performance.now()-started),confidence:data.confidence},true);
  } catch(error) {
    const message=error.name==='AbortError'
      ? 'O limite de 900 s foi atingido. A resposta não foi concluída.'
      : `Não foi possível concluir: ${error.message}`;
    data={ok:false,error:message};
    updateAction({operation,message,phase:'error',elapsed_ms:Math.round(performance.now()-started)},true);
  } finally {
    clearTimeout(timer); liveActions.delete(operation); refreshControls(); saveConversation();
  }
  return data;
}
function readHistory() { try { const items=JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); return Array.isArray(items) ? items : []; } catch (_) { return []; } }
function redactBraveContextForHistory(message) {
  if(message?.role!=='tool' || !['search_web','research_web'].includes(message.tool) || typeof message.content!=='string') return message;
  try {
    const envelope=JSON.parse(message.content), data=envelope?.data;
    if(!data || typeof data!=='object') return message;
    const hasBraveContext=data.source==='brave-llm-context-api'
      || (data.results||[]).some(item=>item?.context_text || item?.source_metadata)
      || (data.search_results||[]).some(item=>item?.context_text || item?.source_metadata)
      || (data.pages||[]).some(page=>page?.pre_extracted===true);
    if(!hasBraveContext) return message;
    const redacted={...data,context_payload_in_history:false};
    for(const key of ['results','search_results','pages','answer','citation_ids','grounded']) delete redacted[key];
    return {...message,content:JSON.stringify({...envelope,data:redacted})};
  } catch(_) { return message; }
}
function saveConversation() {
  const first = conversation.find(m => m.role === 'user'); if (!first) return;
  currentConversationId ||= requestId('conversation');
  const title = (first.content || first.attachments?.[0]?.name || 'Nova conversa').replace(/\s+/g,' ').slice(0,120);
  const persistedMessages=conversation.map(redactBraveContextForHistory);
  const record = {id:currentConversationId,title,updatedAt:Date.now(),messages:persistedMessages,timeline,agentRunRef,agentTaskRef,pendingAgent};
  const items = [record, ...readHistory().filter(item => item.id !== currentConversationId)].slice(0,30);
  while (items.length > 1 && JSON.stringify(items).length > 1800000) items.pop();
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items)); }
  catch (_) { notice('Não foi possível salvar o histórico: o armazenamento local está cheio ou indisponível.'); }
  renderHistory();
  try { localStorage.setItem(LAST_CONVERSATION_KEY, currentConversationId); } catch (_) { /* histórico continua disponível nesta sessão */ }
}
function renderHistory(filter='') {
  const list = document.getElementById('history-list'); if (!list) return;
  list.replaceChildren();
  const items = readHistory().filter(i => i.title.toLowerCase().includes(filter.toLowerCase()));
  for (const item of items) {
    const button = node('button', `history-item ${item.id === currentConversationId ? 'active' : ''}`, item.title);
    button.title = item.title; button.onclick = () => loadConversation(item.id); list.append(button);
  }
  if (!list.children.length) list.append(node('div','history-empty', filter ? 'Nenhuma conversa encontrada' : 'Nenhuma conversa salva'));
}
function filterConversations(val) {
  renderHistory(val || '');
  const clearBtn = document.getElementById('clear-search-btn');
  if (clearBtn) clearBtn.style.display = val ? 'inline-block' : 'none';
}
function clearConversationSearch() {
  const input = document.getElementById('conversation-search-input');
  if (input) { input.value = ''; filterConversations(''); input.focus(); }
}
function focusConversationSearch() {
  const input = document.getElementById('conversation-search-input');
  if (input) input.focus();
}
function clearAttachments() { attachments.forEach(item => item.url && URL.revokeObjectURL(item.url)); attachments=[]; renderAttachments(); }
function loadConversation(id) {
  if (sending || liveActions.size) return notice('Aguarde a operação atual terminar para trocar de conversa.');
  const item = readHistory().find(entry => entry.id === id); if (!item) return;
  saveConversation(); interfaceStudioDraft=false; pendingInterfacePreview=null; interfaceEditPathDraft=null; clearAttachments(); closeTool(); currentConversationId = item.id; conversation = item.messages || [];
  agentRunRef = item.agentRunRef || null;
  agentTaskRef = item.agentTaskRef || null;
  pendingAgent = item.pendingAgent || null;
  timeline = item.timeline || conversation.map(m => ({kind:m.role,text:m.content,meta:'Histórico local'})); chat.replaceChildren();
  for (const record of timeline) {
    if (record.kind === 'action') {
      if (record.state === 'busy') Object.assign(record,{state:'error',log:'Operação interrompida ao sair da página. Confira o resultado antes de repetir.'});
      renderAction(record);
    } else addMessage(record.kind, record.text, record.meta, record.cards, false, record.preview, record.artifact, record.contract, record.studio);
  }
  saveConversation();
  selectedTool=null; renderAttachments(); renderHistory(); scrollChat();
  if(agentRunRef) watchAgentRun();
  if(agentTaskRef) void followAgentWorkflow(agentTaskRef);
  if(agentTaskRef?.status==='clarifying') void restoreAgentClarification(agentTaskRef);
  if(agentTaskRef?.status==='awaiting_approval' && agentTaskRef.approval) renderAgentApprovalCard(agentTaskRef,agentTaskRef.approval);
  if(agentTaskRef?.status==='running') watchPersistedAgentTask();
  if(agentTaskRef?.status==='awaiting_approval' && pendingAgent) notice('O AgentCore aguarda aprovação. Use /agente aprovar para continuar.');
}
function restoreLastConversation() {
  const id = localStorage.getItem(LAST_CONVERSATION_KEY);
  if (!id || !readHistory().some(item => item.id === id)) return;
  loadConversation(id);
}
function toggleAttachmentMenu() { document.getElementById('attach-menu').classList.toggle('open'); }
function selectTool(tool) { selectedTool = tool; document.getElementById('attach-menu').classList.remove('open'); renderAttachments(); input.focus(); }
function clearSelectedTool() { selectedTool=null; renderAttachments(); }
function pickAttachment(kind) {
  if (sending) return;
  const picker = document.getElementById('attachment-input'); picker.dataset.kind=kind;
  picker.multiple=true; picker.webkitdirectory=kind === 'directory';
  picker.accept = kind === 'directory' ? '' : kind === 'document' ? '.txt,.md,.markdown,.rst,.json,.jsonl,.csv,.tsv,.html,.htm,.xml,.yaml,.yml,.toml,.py,.js,.jsx,.ts,.tsx,.rs,.css,.scss,.sql,.sh,.pdf,.docx,.xlsx,.pptx,.odt,.ods,.odp,.epub,.rtf,.eml,.ipynb' : `${kind}/*`;
  picker.click(); document.getElementById('attach-menu').classList.remove('open');
}
function renderAttachments() {
  const target=document.getElementById('attachment-list'); target.replaceChildren();
  if (selectedTool) {
    const chip=node('div','attachment-chip tool',toolLabels[selectedTool] || selectedTool);
    const remove=node('button',null,'×'); remove.onclick=clearSelectedTool; remove.setAttribute('aria-label','Remover ferramenta'); chip.append(remove); target.append(chip);
  }
  attachments.forEach((item,index) => {
    const chip=node('div','attachment-chip');
    if (item.url) { const preview=node('img','attachment-preview'); preview.src=item.url; preview.alt='Prévia do anexo'; chip.append(preview); }
    const label=node('span',null,`${item.name} · ${item.status}`); label.title=label.textContent; chip.append(label);
    const remove=node('button',null,'×'); remove.setAttribute('aria-label',`Remover ${item.name}`); remove.onclick=()=>removeAttachment(index); chip.append(remove); target.append(chip);
  });
  refreshControls();
}
function removeAttachment(index) { if (sending) return; const item=attachments.splice(index,1)[0]; if(item?.url) URL.revokeObjectURL(item.url); renderAttachments(); }
function handleAttachments(event) {
  const files=Array.from(event.target.files || []), kind=event.target.dataset.kind || 'document';
  if (kind === 'directory' && files.length) addDirectoryAttachment(files); else files.forEach(file=>addAttachment(file,kind));
  event.target.value=''; event.target.webkitdirectory=false;
}
function prepareAttachment(files, kind, name) {
  if (sending) return;
  if (attachments.length >= 8) return notice('Use no máximo oito anexos por mensagem.');
  const item={kind,name,state:'reading',status:'Lendo…',url:kind==='image'?URL.createObjectURL(files[0]):null};
  attachments.push(item);
  item.ready=ChatCore.prepareFiles(files,(read,total)=> { item.status=`Lendo ${read}/${total}`; renderAttachments(); },
    kind==='directory' ? {} : {readBinary:ingestLocalAttachment})
    .then(result=> { item.data={kind,name,...result}; item.state=result.failed && !result.files.length ? 'error' : 'ready';
      item.status=result.media?.length ? mediaAttachmentSummary(result.media[0]) : result.files.length ? `${result.files.length} lidos · ${result.omitted} omitidos` : result.warnings?.[0] || 'Formato sem leitor disponível'; })
    .catch(()=> { item.state='error'; item.status='Falha na leitura; remova e tente novamente'; })
    .finally(renderAttachments);
  renderAttachments();
}
async function ingestLocalAttachment(file) {
  if(file.size>ChatCore.LIMITS.mediaFileBytes) throw new Error('Use arquivos de até 16 MiB.');
  const encoded=await new Promise((resolve,reject)=> {
    const reader=new FileReader(); reader.onload=()=>resolve(String(reader.result).split(',',2)[1]);
    reader.onerror=()=>reject(new Error('Não consegui ler o arquivo.')); reader.readAsDataURL(file);
  });
  const response=await fetch('/api/v1/attachments/ingest',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({schema:'local-attachment-upload/v1',name:file.name,data_base64:encoded})});
  const payload=await response.json();
  if(!response.ok || !payload.ok) throw new Error(payload.error || 'Leitor local indisponível.');
  return payload.attachment;
}
function mediaAttachmentSummary(receipt) {
  const analysis=receipt.analysis || {}, seconds=analysis.metadata?.duration_seconds;
  if(receipt.kind==='audio') return `Áudio${Number.isFinite(seconds)?` · ${seconds} s`:''} · sem transcrição`;
  if(receipt.kind==='video') return `Vídeo${Number.isFinite(seconds)?` · ${seconds} s`:''} · metadados`;
  if(receipt.kind==='image') return analysis.text ? 'OCR experimental · confira o original' : 'Imagem recebida · sem texto extraído';
  return analysis.text ? 'Texto do documento extraído' : 'Documento recebido · sem texto extraído';
}
function addDirectoryAttachment(files) { prepareAttachment(files,'directory',`${(files[0].webkitRelativePath || files[0].name).split('/')[0]}/`); }
function addAttachment(file,kind) { prepareAttachment([file],kind,file.name || 'Anexo'); }
input.addEventListener('paste', event=> {
  const files=Array.from(event.clipboardData?.files || []); if (!files.length) return;
  event.preventDefault(); files.forEach(file=>addAttachment(file,file.type.startsWith('image/')?'image':file.type.startsWith('audio/')?'audio':file.type.startsWith('video/')?'video':'document'));
});
document.addEventListener('click',event=> { if (!event.target.closest('.attach-wrap')) document.getElementById('attach-menu').classList.remove('open'); });
document.addEventListener('keydown',event=> { if(event.key==='Escape') document.getElementById('attach-menu').classList.remove('open'); });
function safeUrl(value) { try { const url=new URL(value); return ['http:','https:'].includes(url.protocol) ? url.href : null; } catch (_) { return null; } }
function sourceCard(card) {
  const item=node('div','source'), url=safeUrl(card.url);
  const title=node(url?'a':'span',null,`[${card.source_id || 'fonte'}] ${card.title || 'Fonte'}`);
  if(url) { title.href=url; title.target='_blank'; title.rel='noopener noreferrer'; }
  item.append(title,node('small',null,card.snippet || (card.text || '').slice(0,360)));
  if(url && !card.text) { const analyze=node('button','analyze','Analisar aqui'); analyze.onclick=()=>call('open_page',{url,source_id:card.source_id}); item.append(analyze); }
  if(url) { const external=node('a','external','Abrir no navegador ↗'); external.href=url; external.target='_blank'; external.rel='noopener noreferrer'; item.append(external); }
  return item;
}
function makeCodeArtifact(path, oldText='', newText='', status='observed') {
  const oldLines=String(oldText).split('\n'), newLines=String(newText).split('\n');
  const lines=[], max=Math.max(oldLines.length,newLines.length);
  for(let i=0;i<max;i++) {
    if(oldLines[i]===newLines[i]) lines.push({kind:'context',old_line:i+1,new_line:i+1,text:oldLines[i] ?? ''});
    else {
      if(oldLines[i]!==undefined) lines.push({kind:'remove',old_line:i+1,new_line:null,text:oldLines[i]});
      if(newLines[i]!==undefined) lines.push({kind:'add',old_line:null,new_line:i+1,text:newLines[i]});
    }
  }
  const changed=lines.filter(line=>line.kind!=='context').length;
  return {kind:'code',path,language:(path.match(/\.([a-z0-9]+)$/i)?.[1] || 'text').toLowerCase(),status,bytes_before:String(oldText).length,bytes_after:String(newText).length,lines_after:newLines.length,diff:{format:'line-v1',changed,lines,truncated:false}};
}
function batchArtifactFromData(data={}) {
  const operations=Array.isArray(data.operations)?data.operations:[];
  const files=operations.map(operation=>operation?.result?.artifact).filter(artifact=>artifact?.kind==='code');
  const paths=operations.map(operation=>operation?.result?.path).filter(path=>typeof path==='string');
  return {kind:'batch',status:'applied',transaction_id:data.transaction_id || '',undo_available:data.undo_available===true,paths,files,
    summary:`Lote aplicado em ${files.length} arquivo(s)${data.transaction_id?` · ${data.transaction_id}`:''}.`};
}
function artifactDiffText(artifact) {
  if(artifact?.kind==='batch') return (artifact.files || []).map(file=>`--- ${file.path}\n${artifactDiffText(file)}`).join('\n\n') || 'O lote não contém diffs de arquivo.';
  if(artifact?.kind==='terminal') {
    const status=artifact.exit_code == null ? (artifact.passed===false?'falhou':'concluído') : `código de saída ${artifact.exit_code}`;
    return [`$ ${artifact.command || 'operação aprovada'}`, `${status}${artifact.timed_out?' · tempo esgotado':''}`, artifact.stdout || '', artifact.stderr ? `stderr:\n${artifact.stderr}` : ''].filter(Boolean).join('\n');
  }
  const lines=artifact?.diff?.lines || [];
  if(lines.length) return lines.map(line=>`${line.kind==='add'?'+':line.kind==='remove'?'-':' '} ${line.text ?? ''}`).join('\n');
  if(artifact?.content) return String(artifact.content);
  return 'Nenhum detalhe textual foi retornado por esta ferramenta.';
}
function copyArtifactDiff(button,artifact) {
  navigator.clipboard.writeText(artifactDiffText(artifact)).then(()=>{
    const original=button.textContent; button.textContent='Copiado';
    setTimeout(()=>{ button.textContent=original; },1400);
  }).catch(()=>notice('Não foi possível copiar o artefato.'));
}
function openArtifactWorkspace(artifact) {
  if(!artifact?.path) return;
  setMode('chat'); openTool('workspace'); setTimeout(()=>runPanelRead(artifact.path),0);
}
function openArtifactOverlay(artifact) {
  const overlay=node('div','code-overlay drawer'), card=node('section','code-overlay-card'), head=node('div','code-overlay-head'), title=node('div');
  title.append(node('div','code-overlay-title',`${artifact.kind==='diagnostics'?'Diagnóstico':artifact.kind==='terminal'?'Saída do terminal':artifact.kind==='batch'?'Lote de alterações':'Diff'} · ${artifact.path || artifact.check || artifact.transaction_id || 'operação'}`),node('div','code-overlay-subtitle',artifact.status || 'detalhes estruturados'));
  const actions=node('div','code-overlay-actions'), copy=node('button',null,'Copiar'); copy.onclick=()=>copyArtifactDiff(copy,artifact);
  if (artifact.status === 'proposed' && artifact.path) {
    const accept=node('button','artifact-accept','Aceitar'); accept.onclick=()=>{ overlay.remove(); saveEditorFile(); };
    const reject=node('button','artifact-reject','Rejeitar'); reject.onclick=()=>{ overlay.remove(); discardEditorChanges(); };
    actions.append(accept,reject);
  }
  const close=node('button',null,'Fechar'); close.onclick=()=>overlay.remove(); actions.append(copy,close); head.append(title,actions);
  const context=node('div','code-overlay-context',`Fluxo da conversa · ${artifact.path || 'resultado da ferramenta'}`);
  if (artifact.path) { const workspace=node('button',null,'Editar arquivo no Chat'); workspace.onclick=()=>{ overlay.remove(); openArtifactWorkspace(artifact); }; context.append(workspace); }
  const browser=node('button',null,'Painel do navegador'); browser.onclick=()=>{ overlay.remove(); openTool('url'); }; context.append(browser);
  const body=node('pre','code-overlay-pre'); body.textContent=artifact.kind==='diagnostics' ? `${artifact.summary || ''}\n\n${artifact.stdout || ''}\n${artifact.stderr || ''}`.trim() : artifactDiffText(artifact);
  card.append(head,context,body); overlay.append(card); overlay.onclick=event=>{ if(event.target===overlay) overlay.remove(); };
  document.body.append(overlay); close.focus();
}
function renderArtifact(artifact) {
  if(!artifact) return null;
  if(artifact.kind==='batch') {
    const card=node('section','artifact-card batch-artifact-card'), head=node('div','artifact-head'), title=node('div','artifact-title'), actions=node('div','artifact-actions');
    title.append(node('strong',null,`Lote · ${(artifact.files || []).length} arquivo(s)`),node('small',null,artifact.transaction_id || artifact.status || 'alterações aplicadas'));
    const copy=node('button',null,'Copiar diff'); copy.onclick=()=>copyArtifactDiff(copy,artifact); actions.append(copy); head.append(title,actions); card.append(head);
    if(artifact.undo_available) card.append(node('div','diagnostic-summary',`Para desfazer com segurança: “desfaça o lote ${artifact.transaction_id}”. O runtime compara hashes antes de restaurar.`));
    for(const file of artifact.files || []) { const rendered=renderArtifact(file); if(rendered) card.append(rendered); }
    if(!(artifact.files || []).length) card.append(node('div','artifact-empty','O lote alterou pastas; não houve diff de conteúdo de arquivo.'));
    return card;
  }
  const card=node('section',artifact.kind==='diagnostics'?'artifact-card diagnostic-card':'artifact-card');
  const head=node('div','artifact-head'), title=node('div','artifact-title');
  title.append(node('strong',null,artifact.kind==='diagnostics'?'Diagnóstico de verificação':artifact.kind==='terminal'?'Saída do terminal':`Artefato · ${artifact.path || 'arquivo'}`));
  title.append(node('small',null,artifact.status || artifact.language || 'resultado estruturado'));
  const actions=node('div','artifact-actions');
  if(artifact.path) { const workspace=node('button',null,'Abrir arquivo'); workspace.onclick=()=>openArtifactWorkspace(artifact); actions.append(workspace); }
  const focus=node('button',null,'Foco'); focus.onclick=()=>openArtifactOverlay(artifact); actions.append(focus);
  const copy=node('button',null,'Copiar'); copy.onclick=()=>copyArtifactDiff(copy,artifact); actions.append(copy);
  if(artifact.status==='proposed' && artifact.path) {
    const accept=node('button','artifact-accept','Aceitar'); accept.onclick=()=>saveEditorFile();
    const reject=node('button','artifact-reject','Rejeitar'); reject.onclick=()=>discardEditorChanges();
    actions.append(accept,reject);
  }
  head.append(title,actions); card.append(head);
  if(artifact.kind==='diagnostics' || artifact.kind==='terminal') {
    card.append(node('div','diagnostic-summary',artifact.summary || (artifact.kind==='terminal' ? `${artifact.command || 'Comando aprovado'} · ${artifact.exit_code == null ? (artifact.passed===false?'falhou':'concluído'):`código ${artifact.exit_code}`}` : `${artifact.check || 'verificação'} · ${artifact.passed?'passou':'falhou'}`)));
    const output=[artifact.stdout,artifact.stderr].filter(Boolean).join('\n');
    if(output) card.append(node('pre','diagnostic-output',output));
    return card;
  }
  const diff=artifact.diff || {changed:0,lines:String(artifact.content || '').split('\n').map((text,index)=>({kind:'context',old_line:index+1,new_line:index+1,text})),truncated:false};
  const lines=diff.lines || [];
  const summary=node('div','artifact-summary',`${artifact.language || 'text'} · ${diff?.changed ?? 0} alteração(ões) · ${artifact.lines_after || 0} linha(s) resultantes`); card.append(summary);
  const viewport=node('div','artifact-diff');
  for(const line of lines.slice(0,240)) {
    const row=node('div',`diff-line ${line.kind || 'context'}`);
    const number=line.kind==='remove'?line.old_line:line.new_line;
    row.append(node('span','diff-sign',line.kind==='add'?'+':line.kind==='remove'?'-':' '),node('span','diff-number',number == null ? '' : String(number)),node('code',null,line.text ?? '')); viewport.append(row);
  }
  if(lines.length>240) viewport.append(node('div','artifact-more',`Mais ${lines.length-240} linha(s) no foco expandido.`));
  if(lines.length) card.append(viewport); else card.append(node('div','artifact-empty','Nenhuma alteração textual detectada.'));
  return card;
}
function renderResponseContract(contract) {
  if (!contract || contract.schema !== 'agent-response/v1') return null;
  const card=node('details',`response-contract ${contract.status || 'complete'}`);
  if (contract.status && contract.status !== 'complete') card.open=true;
  const summary=node('summary','response-contract-summary','Detalhes da resposta');
  card.append(summary);
  const head=node('div','response-contract-head');
  head.append(node('strong',null,`Estado: ${contract.status || 'complete'}`));
  if(contract.data?.backend) head.append(node('span',null,contract.data.backend));
  card.append(head);
  for(const warning of contract.warnings || []) card.append(node('div','response-contract-warning',`Aviso: ${warning}`));
  if(contract.data?.context?.schema) card.append(node('div','response-contract-detail',`Contexto: ${contract.data.context.schema} · ${contract.data.context.status || 'ready'}`));
  if(contract.data?.workflow?.strategy) card.append(node('div','response-contract-detail',`Estratégia: ${contract.data.workflow.strategy}`));
  return card;
}
function addMessage(kind,text,meta='',cards=[],record=true,preview=null,artifact=null,contract=null,studio=null) {
  document.getElementById('welcome')?.remove();
  if(record) timeline.push({kind,text,meta,cards,preview,artifact,contract,studio});
  const row=node('div',`message ${kind}`), body=node('div');
  const bubble=node('div','bubble');
  if(kind==='assistant' && typeof ChatCore!=='undefined' && ChatCore.formatMarkdown) {
    bubble.innerHTML=ChatCore.formatMarkdown(text);
  } else {
    bubble.textContent=text;
  }
  body.append(bubble);
  if(cards?.length) { const sources=node('div','sources'); cards.forEach(card=>sources.append(sourceCard(card))); body.append(sources); }
  if(preview?.html) {
    const previewCard=node('div','preview-card');
    const head=node('div','preview-head');
    const caption=node('div','preview-caption',`Prévia local · ${preview.path || 'arquivo HTML'}`);
    const actions=node('div','preview-actions');
    const workspaceButton=node('button','preview-action','Abrir projeto'); workspaceButton.onclick=()=>openTool('workspace');
    const fileButton=node('button','preview-action','Ler arquivo'); fileButton.onclick=()=>{ openTool('workspace'); setTimeout(()=>runPanelRead(preview.path),0); };
    const suggestionsButton=node('button','preview-action','Sugerir melhorias'); suggestionsButton.onclick=()=>preparePreviewFollowup(preview,'suggest');
    const iterationButton=node('button','preview-action','Descrever ajuste'); iterationButton.onclick=()=>preparePreviewFollowup(preview,'iterate');
    actions.append(workspaceButton,fileButton,suggestionsButton,iterationButton); head.append(caption,actions);
    const frame=node('iframe','web-preview'); frame.setAttribute('sandbox','allow-scripts'); frame.setAttribute('title',`Prévia de ${preview.path || 'página HTML'}`); frame.srcdoc=preview.html;
    previewCard.append(head,frame); body.append(previewCard);
  }
  const artifactCard=renderArtifact(artifact); if(artifactCard) body.append(artifactCard);
  const contractCard=renderResponseContract(contract); if(contractCard) body.append(contractCard);
  if(studio) renderStudioActions(body,studio);
  const metaElement=node('div','meta',meta); body.append(metaElement);
  const avatar=node('div','avatar',kind==='user'?'Você':'b'); avatar.setAttribute('aria-hidden','true');
  row.append(avatar,body); chat.append(row); scrollChat();
  return {row,bubble,body,meta:metaElement};
}
function assistantReply(text,meta,cards=[],preview=null,artifact=null,contract=null,studio=null) { conversation.push({role:'assistant',content:text,artifact:artifact || undefined,contract:contract || undefined}); addMessage('assistant',text,meta,cards,true,preview,artifact,contract,studio); saveConversation(); }
function renderStudioActions(container,studio) {
  if(studio?.kind!=='interface-directions/v1') return;
  const card=node('section','studio-directions');
  card.append(node('strong','studio-directions-title','Transformar uma direção em prévia'));
  card.append(node('p','studio-directions-copy','Escolha uma das propostas acima. A página será criada no workspace selecionado e aparecerá numa prévia isolada.'));
  const actions=node('div','studio-direction-actions');
  for(let index=1;index<=3;index++) {
    const button=node('button','studio-direction-button',`Usar direção ${index}`);
    button.type='button';
    button.onclick=()=>startInterfacePreview(studio,index);
    actions.append(button);
  }
  card.append(actions); container.append(card);
}
function studioDirectionText(text,index) {
  const lines=String(text||'').split(/\r?\n/);
  const marker=new RegExp(`^\\s*(?:#{1,4}\\s*)?(?:\\*\\*)?(?:(?:dire[cç][aã]o|op[cç][aã]o|conceito)\\s*)?${index}(?:[.) :—–-]|\\s+-)\\s*(.*?)(?:\\*\\*)?\\s*$`,'i');
  const start=lines.findIndex(line=>marker.test(line));
  if(start<0) return lines.join(' ').replace(/\s+/g,' ').slice(0,700);
  const content=[lines[start].replace(marker,'$1')];
  for(let cursor=start+1;cursor<lines.length;cursor++) {
    if(/^\s*(?:#{1,4}\s*)?(?:\*\*)?(?:(?:dire[cç][aã]o|op[cç][aã]o|conceito)\s*)?[1-3](?:[.) :—–-]|\s+-)/i.test(lines[cursor])) break;
    content.push(lines[cursor]);
  }
  return content.join(' ').replace(/\s+/g,' ').trim().slice(0,700);
}
function startInterfacePreview(studio,index) {
  if(sending || liveActions.size) return notice('Aguarde a operação atual terminar.');
  if(!localStorage.getItem('ia-local-zero-workspace')) {
    notice('Escolha um workspace local antes de criar a prévia.');
    openTool('workspace');
    return;
  }
  const direction=studioDirectionText(studio.directions,index);
  pendingInterfacePreview={
    prompt:`Ideia: ${studio.idea}\nDireção visual ${index}: ${direction}. Criar uma interface responsiva em português, com hierarquia clara e uma interação demonstrável.`,
    title:`Protótipo · direção ${index}`,
    path:'preview/index.html'
  };
  input.value=`Gerar prévia da direção ${index}: ${studio.idea}`;
  input.focus();
}
function preparePreviewFollowup(preview,kind) {
  const path=String(preview?.path||'preview/index.html');
  pendingInterfacePreview=null;
  interfaceEditPathDraft=kind==='iterate'?path:null;
  input.value=kind==='suggest'
    ? `Analise a página ${path} em relação ao objetivo da conversa. Sugira três melhorias visuais ou de conteúdo, explique o benefício de cada uma e aguarde minha escolha sem alterar o arquivo.`
    : `Quero alterar a página ${path}. Ajustes desejados: `;
  input.focus();
  input.setSelectionRange(input.value.length,input.value.length);
}
function withPlannerMeta(meta, confidenceValue, fallbackLabel = 'Plano') {
  if (confidenceValue == null || Number.isNaN(Number(confidenceValue))) return meta || fallbackLabel;
  const info = ChatCore.confidenceLabel(confidenceValue);
  const label = `${fallbackLabel} · conf. ${info.label} (${info.score.toFixed(2)})`;
  return meta ? `${label} · ${meta}` : label;
}
function responseSignal(data) {
  const signals = [];
  if (data?.agent?.recovery) signals.push(data.agent.recovery === 'next-source' ? 'Recuperação ativa' : 'Recuperação');
  if (data?.backend === 'quality-gate') signals.push('Contexto limitado');
  if (data?.evidence?.status === 'provisional') signals.push('Fonte não corroborada');
  if (data?.evidence?.status === 'corroborated') signals.push('Fontes independentes');
  if (data?.evidence?.status === 'unverified') signals.push('Não verificado');
  if (data?.skill?.status) signals.push(`Competência: ${data.skill.status}`);
  if (data?.evidence?.fiction_warnings?.length) signals.push('Sinal de ficção');
  const omitted = Number(data?.data?.coverage?.omitted || 0);
  if (omitted > 0) signals.push(`${omitted} arquivo(s) fora da leitura`);
  const workflowPhase = data?.workflow?.phase;
  if (workflowPhase === 'plan') signals.push('Workflow: planejando');
  if (workflowPhase === 'abstain') signals.push('Workflow: aguardando evidência');
  return signals.length ? ` · ${signals.join(' · ')}` : '';
}
function responseMeta(meta, data, confidenceValue, fallbackLabel) {
  return withPlannerMeta(`${meta}${responseSignal(data)}`, confidenceValue, fallbackLabel);
}
function processArtifact(tool, data={}) {
  const readiness=data.readiness || {};
  const state=data.state || 'unknown';
  const summary=tool==='process_start'
    ? `Processo ${data.process_id || ''} iniciado pelo perfil ${data.profile || 'auto-dev'}; aguardando confirmação de prontidão.`
    : tool==='process_stop'
      ? `Processo ${data.process_id || ''} · estado ${state}.`
      : readiness.ready
        ? `Processo ${data.process_id || ''} pronto em ${readiness.url}.`
        : `Processo ${data.process_id || ''} · estado ${state}${readiness.known ? ' · prontidão ainda não confirmada' : ' · prontidão não confirmada'}.`;
  const streams=[data.stdout?.text, data.stderr?.text].filter(Boolean).join('\n');
  return {kind:'terminal',status:state,command:data.command || data.profile || tool,exit_code:data.exit_code,
    passed:readiness.ready===true,summary,stdout:streams,stderr:'',process_id:data.process_id,
    readiness:readiness.url || readiness.evidence || 'não confirmada'};
}
function actionIntro(tool, args={}) {
  const project = localStorage.getItem('ia-local-zero-workspace');
  const target = project ? ` no projeto ativo (${project.split('/').pop() || project})` : '';
  const messages = {
    search_web: 'Vou pesquisar fontes atuais para responder com base em evidências.',
    research_web: 'Vou fazer uma pesquisa mais completa, abrir as fontes relevantes e sintetizar o resultado.',
    inspect_project: `Vou analisar a estrutura do projeto${target} antes de sugerir qualquer mudança.`,
    project_checks: 'Vou verificar o projeto e explicar claramente o que for encontrado.',
    create_web_page: `Entendi. Vou criar essa página${target} e mostrar uma prévia do resultado.`,
    create_file: `Entendi. Vou criar o arquivo solicitado${target}.`,
    edit_file: `Entendi. Vou editar o arquivo solicitado${target} preservando o restante do projeto.`,
    apply_batch: `Vou reunir as mudanças em um lote revisável${target}, mostrar um diff por arquivo e só aplicar depois da sua aprovação.`,
    undo_batch: 'Vou conferir os hashes e desfazer o lote mais recente somente se os arquivos não tiverem mudado depois.',
    create_directory: `Entendi. Vou criar a pasta solicitada${target}.`,
    create_workspace: 'Vou criar o novo diretório de projeto e selecioná-lo como workspace ativo.',
    read_file: 'Vou ler o arquivo solicitado e destacar o que importa.',
    list_files: `Vou explorar o projeto${target} e mostrar a estrutura relevante.`,
    search_files: `Vou procurar no código${target} e trazer apenas as ocorrências úteis.`,
    terminal_run: `Vou executar um perfil permitido no projeto${target}. A ação pede confirmação e mostra a saída ao terminar.`,
    process_start: `Vou identificar e iniciar o servidor de desenvolvimento do projeto${target}. O perfil pede confirmação e o status será acompanhado pela saída local.`,
    process_status: 'Vou consultar o estado e as novas linhas de saída do processo local.',
    process_stop: 'Vou encerrar o processo local iniciado pelo runtime; a ação pede confirmação.'
  };
  return messages[tool] || `Entendi. Vou executar essa etapa${target} e explicar o resultado.`;
}
function toolNeedsApproval(request) {
  if (!request || typeof request !== 'object') return false;
  if (request.requires_approval === true) return true;
  return ['terminal_run','process_start','process_stop','apply_batch','undo_batch','create_workspace','create_web_page','create_file','edit_file','apply_repair','create_directory'].includes(request.tool);
}
function batchApprovalPreview(operations) {
  return (operations || []).map(operation=>{
    const tool=operation?.tool || 'operação', args=operation?.arguments || {}, path=args.path || '(sem caminho)';
    if(tool==='create_file') return `--- /dev/null\n+++ ${path}\n${String(args.content || '').split('\n').map(line=>`+${line}`).join('\n')}`;
    if(tool==='edit_file') return `--- ${path}\n+++ ${path}\n${String(args.old_text || '').split('\n').map(line=>`-${line}`).join('\n')}\n${String(args.new_text || '').split('\n').map(line=>`+${line}`).join('\n')}`;
    return `--- ${path}\n+++ ${path}\n(pasta a criar)`;
  }).join('\n\n');
}

function agentReportText(report, objective) {
  return ChatCore.formatAgentReport(report || {}, {includeSources: objective === 'research'});
}

function agentUnderstandingText(understanding) {
  const lines = [understanding.interpretation || 'Entendi a solicitação.'];
  const assumptions = Array.isArray(understanding.assumptions) ? understanding.assumptions : [];
  const assumptionText = assumptions.map(item => typeof item === 'string' ? item : item?.value)
    .filter(value => typeof value === 'string' && value.trim());
  if (assumptionText.length) lines.push('**Premissas até aqui:**\n' + assumptionText.map(value => '- ' + value).join('\n'));
  if (understanding.next_step) lines.push('**Próximo passo:** ' + understanding.next_step);
  return lines.join('\n\n');
}

function renderAgentClarificationCard(understanding, ref = agentTaskRef) {
  if (!ref?.taskId || !Array.isArray(understanding?.questions) || !understanding.questions.length) return;
  const existing = [...chat.querySelectorAll('.clarification-card')]
    .find(card => card.dataset.taskId === ref.taskId);
  if (existing) existing.remove();
  const card = node('section', 'approval-card inline-confirm clarification-card');
  card.dataset.taskId = ref.taskId;
  const copy = node('div', 'approval-content');
  const head = node('div', 'approval-head');
  const heading = node('div');
  heading.append(node('div', 'approval-eyebrow', 'AGENTCORE · ESCLARECIMENTO'),
    node('div', 'inline-confirm-title', 'Preciso dessas respostas para montar o plano'));
  head.append(node('span', 'approval-icon', '?'), heading);
  copy.append(head, node('div', 'inline-confirm-note', 'As respostas serão usadas para continuar esta mesma tarefa.'));
  const fields = [];
  for (const [index, question] of understanding.questions.slice(0, 3).entries()) {
    const label = node('label', 'clarification-field');
    label.append(node('span', null, question));
    const textarea = node('textarea', 'clarification-answer');
    textarea.rows = 2;
    textarea.maxLength = 2000;
    textarea.placeholder = 'Sua resposta';
    label.append(textarea);
    copy.append(label);
    fields.push(textarea);
  }
  const actions = node('div', 'inline-confirm-actions');
  const send = node('button', 'confirm-accept', 'Responder e continuar');
  send.type = 'button';
  send.onclick = async () => {
    const answers = fields.map(field => field.value.trim()).filter(Boolean);
    if (!answers.length) { notice('Responda pelo menos uma pergunta para continuar.'); fields[0]?.focus(); return; }
    send.disabled = true;
    send.textContent = 'Continuando…';
    await resumeAgentClarification(ref, answers, card, send);
  };
  actions.append(send);
  card.append(copy, actions);
  chat.append(card);
  scrollChat();
}

async function resumeAgentClarification(ref, answers, card, button) {
  const operation = requestId('agent-resume');
  ref.status = 'running';
  saveConversation();
  try {
    const data = await requestWithActivity(`/api/v1/agent/tasks/${encodeURIComponent(ref.taskId)}/resume`, {
      schema: 'agent-resume/v1', request_id: requestId('resume'), kind: 'clarification', answers,
    }, operation, 'Retomando o AgentCore com suas respostas', true, true);
    await drainActivityEvents(operation);
    if (!data.ok) throw new Error(data.error || 'A retomada não foi concluída.');
    card.remove();
    if (data.status === 'clarifying' && data.understanding) {
      ref.status = 'clarifying';
      ref.clarification = data.understanding;
      assistantReply(agentUnderstandingText(data.understanding), 'AgentCore · ainda precisa de esclarecimento');
      renderAgentClarificationCard(data.understanding, ref);
      void refreshAgentWorkflow(ref);
      saveConversation();
      return;
    }
    const report = data.report;
    if (!report) throw new Error('A retomada terminou sem relatório nem novas perguntas.');
    ref.taskId = report.taskId || ref.taskId;
    ref.status = report.status || 'blocked';
    ref.shown = true;
    assistantReply(agentReportText(report, ref.objective || 'auto'), ChatCore.agentReportMeta(report, data.awaitingApproval));
    if (Array.isArray(report.artifacts) && report.artifacts.some(artifact => typeof artifact?.path === 'string' && artifact.path.trim())) {
      await refreshProjectDock();
    }
    void refreshAgentWorkflow(ref);
    if (data.awaitingApproval === true && report.status === 'blocked') {
      const approval = (report.events || []).findLast(event => event.kind === 'approval.required');
      if (approval) renderAgentApprovalCard(ref, approval);
    }
    saveConversation();
  } catch (error) {
    ref.status = 'clarifying';
    button.disabled = false;
    button.textContent = 'Responder e continuar';
    let message = card.querySelector('.clarification-error');
    if (!message) { message = node('div', 'clarification-error'); card.querySelector('.approval-content')?.append(message); }
    message.textContent = error.message;
    saveConversation();
  } finally {
    liveActions.delete(operation);
    refreshControls();
  }
}

function appendAgentApprovalPreview(copy, approval) {
  const diff = approval.payload?.repairDiff || approval.payload?.fileDiff;
  if (diff && typeof diff.newText === 'string' && (diff.oldText === null || typeof diff.oldText === 'string')) {
    const disclosure = node('details', 'approval-details');
    disclosure.open = true;
    disclosure.append(node('summary', null, 'Revisar diff exato antes de autorizar'));
    const preview = node('pre', 'approval-preview');
    const removed = diff.oldText === null ? '' : diff.oldText.split('\n').map(line => '- ' + line).join('\n') + '\n';
    const added = diff.newText.split('\n').map(line => '+ ' + line).join('\n');
    preview.textContent = `--- ${diff.oldText === null ? '/dev/null' : diff.path || 'arquivo'}\n+++ ${diff.path || 'arquivo'}\n${removed}${added}`;
    disclosure.append(preview);
    copy.append(disclosure);
  }
  const batch = approval.payload?.batchPreview;
  if (approval.payload?.tool === 'apply_batch' && Array.isArray(batch)) {
    const disclosure = node('details', 'approval-details');
    disclosure.open = true;
    disclosure.append(node('summary', null, `Revisar diff de ${batch.length} operação(ões)`));
    const preview = node('pre', 'approval-preview');
    preview.textContent = batchApprovalPreview(batch);
    disclosure.append(preview);
    copy.append(disclosure);
  }
}

function renderAgentApprovalCard(ref, approval) {
  if (!ref?.taskId) return;
  const existing = [...chat.querySelectorAll('.agent-approval-card')]
    .find(card => card.dataset.taskId === ref.taskId);
  if (existing) existing.remove();
  const card = node('section', 'approval-card inline-confirm agent-approval-card');
  card.dataset.taskId = ref.taskId;
  const copy = node('div', 'approval-content');
  const head = node('div', 'approval-head');
  const heading = node('div');
  heading.append(node('div', 'approval-eyebrow', 'AGENTCORE · AUTORIZAÇÃO'),
    node('div', 'inline-confirm-title', 'Aprovação necessária'));
  head.append(node('span', 'approval-icon', '✓'), heading);
  const tool = approval.payload?.tool || 'ação solicitada';
  copy.append(head, node('div', 'inline-confirm-path', `Autorizar ${tool}${approval.detail ? ' em ' + approval.detail : ''}?`),
    node('div', 'inline-confirm-note', approval.payload?.reason || 'O agente continuará após sua decisão e verificará o resultado.'));
  appendAgentApprovalPreview(copy, approval);
  const actions = node('div', 'inline-confirm-actions');
  const reject = node('button', 'confirm-cancel', 'Rejeitar');
  const accept = node('button', 'confirm-accept', 'Aprovar e continuar');
  reject.type = accept.type = 'button';
  reject.onclick = () => resumeAgentDecision(ref, approval, 'reject', reject, accept);
  accept.onclick = () => resumeAgentDecision(ref, approval, 'approve', reject, accept);
  actions.append(reject, accept);
  card.append(copy, actions);
  chat.append(card);
  ref.approval = approval;
  ref.status = 'awaiting_approval';
  saveConversation();
  scrollChat();
}

async function resumeAgentDecision(ref, approval, decision, reject = null, accept = null) {
  const actionId = approval?.payload?.actionId;
  if (typeof actionId !== 'string' || !actionId.trim()) return notice('A ação pendente não trouxe um identificador válido; atualize o acompanhamento da tarefa.');
  if (reject && accept) reject.disabled = accept.disabled = true;
  const operation = requestId('agent-resume');
  try {
    const data = await requestWithActivity(`/api/v1/agent/tasks/${encodeURIComponent(ref.taskId)}/resume`, {
      schema: 'agent-resume/v1', request_id: requestId('resume'), kind: 'approval',
      action_id: actionId, decision,
    }, operation, decision === 'approve' ? 'Executando a ação aprovada' : 'Registrando a rejeição da ação', true, true);
    await drainActivityEvents(operation);
    if (!data.ok) throw new Error(data.error || 'A decisão não foi registrada.');
    chat.querySelectorAll('.agent-approval-card').forEach(card => { if (card.dataset.taskId === ref.taskId) card.remove(); });
    if (decision === 'reject') {
      ref.status = 'blocked';
      delete ref.approval;
      assistantReply('A ação foi rejeitada e não foi executada.', 'AgentCore · ação rejeitada');
    } else if (data.report) {
      const report = data.report;
      ref.status = report.status || 'blocked';
      delete ref.approval;
      assistantReply(agentReportText(report, ref.objective || 'auto'), ChatCore.agentReportMeta(report, data.awaitingApproval));
      if (data.awaitingApproval === true) {
        const nextApproval = (report.events || []).findLast(event => event.kind === 'approval.required');
        if (nextApproval) renderAgentApprovalCard(ref, nextApproval);
      }
      if (Array.isArray(report.artifacts) && report.artifacts.some(artifact => typeof artifact?.path === 'string' && artifact.path.trim())) {
        await refreshProjectDock();
      }
    }
    void refreshAgentWorkflow(ref);
    saveConversation();
  } catch (error) {
    if (reject && accept) reject.disabled = accept.disabled = false;
    notice(error.message);
  } finally {
    liveActions.delete(operation);
    refreshControls();
  }
}

async function pursueAgent(prompt, approved = false, objective = 'auto', history = [], material = []) {
  const workspaceRoot = localStorage.getItem('ia-local-zero-workspace') || undefined;
  const operation = requestId('agent-core');
  agentStreamDrafts.set(operation,{draft:null,text:''});
  agentTaskRef = {operationId:operation,status:'running',shown:false};
  void followAgentWorkflow(agentTaskRef);
  saveConversation();
  const data = await requestWithActivity(
    '/api/v1/agent/pursue',
    {...(approved ? {} : {schema:'agent-request/v2'}),prompt,objective,workspaceRoot,history,approved,operationId:operation,
      conversation_id:currentConversationId,request_id:requestId('req'),attachments:ChatCore.agentAttachments(material)},
    operation,
    approved ? 'Retomando a etapa aprovada do AgentCore' : 'AgentCore selecionando, pesquisando e planejando',
    true,
    true,
  );
  await drainActivityEvents(operation);
  if (!data.ok) {
    agentStreamDrafts.get(operation)?.draft?.row.remove(); agentStreamDrafts.delete(operation);
    liveActions.delete(operation); refreshControls();
    await watchPersistedAgentTask(data.error || 'O AgentCore não conseguiu iniciar.');
    return;
  }
  if (data.awaitingClarification === true && data.understanding) {
    const understanding = data.understanding;
    agentTaskRef = {operationId:operation,taskId:understanding.task_id,status:'clarifying',shown:true,
      clarification:understanding,prompt,objective,history};
    const streamed=agentStreamDrafts.get(operation);
    agentStreamDrafts.delete(operation);
    finishStreamingDraft(streamed?.draft,agentUnderstandingText(understanding),'AgentCore · aguardando esclarecimento');
    void refreshAgentWorkflow(agentTaskRef);
    renderAgentClarificationCard(understanding,agentTaskRef);
    liveActions.delete(operation); refreshControls(); saveConversation();
    return;
  }
  if (!data.report) {
    agentStreamDrafts.get(operation)?.draft?.row.remove(); agentStreamDrafts.delete(operation);
    liveActions.delete(operation); refreshControls();
    await watchPersistedAgentTask('O AgentCore encerrou sem relatório e sem indicar uma pergunta pendente.');
    return;
  }
  const report = data.report || {};
  agentTaskRef = {operationId:operation,taskId:report.taskId,status:report.status,shown:true};
  void refreshAgentWorkflow(agentTaskRef);
  saveConversation();
  const streamed=agentStreamDrafts.get(operation);
  agentStreamDrafts.delete(operation);
  finishStreamingDraft(streamed?.draft,agentReportText(report,objective),ChatCore.agentReportMeta(report, data.awaitingApproval));
  if (Array.isArray(report.artifacts) && report.artifacts.some((artifact) => typeof artifact?.path === 'string' && artifact.path.trim())) {
    await refreshProjectDock();
  }
  liveActions.delete(operation); refreshControls();
  const approval = (report.events || []).findLast(event => event.kind === 'approval.required');
  if (data.awaitingApproval === true && report.status === 'blocked' && approval) {
    pendingAgent = {prompt, objective, history};
    agentTaskRef.status = 'awaiting_approval';
    saveConversation();
    const card = node('section', 'approval-card inline-confirm');
    const copy = node('div', 'approval-content');
    const head = node('div', 'approval-head');
    const heading = node('div');
    heading.append(node('div', 'approval-eyebrow', 'AGENTCORE · AUTORIZAÇÃO'), node('div', 'inline-confirm-title', 'Aprovação necessária'));
    head.append(node('span', 'approval-icon', '✓'), heading);
    const approvalTool = approval.payload?.tool;
    const approvalSubject = ['terminal_run','project_checks','process_start','process_stop'].includes(approvalTool)
      ? 'a execução de ' + approvalTool
      : approvalTool === 'apply_batch'
        ? `o lote com ${(approval.payload?.batchPreview || []).length} operação(ões)`
        : approvalTool === 'undo_batch'
          ? 'desfazer o lote ' + (approval.payload?.transaction_id || 'mais recente')
      : ['create_file','edit_file','apply_repair','create_directory'].includes(approvalTool)
        ? approvalTool + ' no caminho ' + (approval.detail || 'do workspace')
        : 'a ação ' + (approvalTool || 'solicitada');
    const approvalReason = approval.payload?.reason;
    const approvalNote = approvalTool === 'process_start'
      ? 'O perfil reconhecido pode executar código declarado no projeto e deixar o servidor ativo até ser encerrado.'
      : approvalTool === 'process_stop'
        ? 'Esta ação encerra o processo identificado pelo runtime.'
        : approvalTool === 'apply_batch'
          ? 'Revise os diffs por arquivo. A aprovação aplica todas as operações como um lote e cria um identificador de undo.'
          : approvalTool === 'undo_batch'
            ? 'O runtime compara os hashes e recusa qualquer restauração que possa sobrescrever alterações posteriores.'
        : approvalReason
          ? approvalReason + ' A aprovação vale para esta ação e será seguida de verificação.'
          : 'A aprovação vale para esta ação e será seguida de verificação.';
    copy.append(
      node('div', 'inline-confirm-path', 'Autorizar ' + approvalSubject + '?'),
      node('div', 'inline-confirm-note', approvalNote),
    );
    appendAgentApprovalPreview(copy, approval);
    copy.prepend(head);
    const actions = node('div', 'inline-confirm-actions');
    const cancel = node('button', 'confirm-cancel', 'Cancelar');
    const accept = node('button', 'confirm-accept', 'Aprovar e continuar');
    cancel.onclick = () => { pendingAgent = null; agentTaskRef.status = 'blocked'; card.remove(); saveConversation(); };
    accept.onclick = async () => { const next = pendingAgent; pendingAgent = null; card.remove(); if (next) await pursueAgent(next.prompt, true, next.objective, next.history); };
    actions.append(cancel, accept); card.append(copy, actions); chat.append(card); scrollChat();
  }
}

const workflowRequests = new WeakSet();
async function refreshAgentWorkflow(ref = agentTaskRef) {
  if (!ref || ref !== agentTaskRef || workflowRequests.has(ref)) return;
  workflowRequests.add(ref);
  let panel;
  try {
    panel = [...chat.querySelectorAll('.agent-workflow')].find(item => item.dataset.operation === ref.operationId);
    if (!panel) {
      panel = node('section','agent-workflow'); panel.dataset.operation = ref.operationId;
      panel.setAttribute('aria-label','Acompanhamento da tarefa');
      chat.append(panel);
    }
    if (!ref.taskId) {
      const listed = await fetch(`/api/v1/agent/tasks?operation_id=${encodeURIComponent(ref.operationId)}`,
        {cache:'no-store',signal:AbortSignal.timeout(8000)}).then(response=>response.json());
      if (!listed.ok) throw new Error(listed.error || 'Consulta indisponível');
      if (ref !== agentTaskRef) return;
      ref.taskId = listed.tasks?.[0]?.taskId;
      if (!ref.taskId) { panel.textContent='Workflow · aguardando registro da tarefa…'; return; }
    }
    const data = await fetch(`/api/v1/agent/tasks/${encodeURIComponent(ref.taskId)}/workflow`,
      {cache:'no-store',signal:AbortSignal.timeout(8000)}).then(response=>response.json());
    if (!data.ok || !data.workflow) throw new Error(data.error || 'Workflow indisponível');
    if (ref !== agentTaskRef || !panel.isConnected) return;
    const w = data.workflow;
    const statuses = {running:'Em andamento',clarifying:'Aguardando esclarecimento',awaiting_approval:'Aguardando aprovação',blocked:'Bloqueado',
      completed:'Concluído',failed:'Falhou',interrupted:'Interrompido'};
    const checks = {unknown:'Ainda não confirmada',stale:'Nova verificação necessária',passed:'Aprovada',failed:'Falhou'};
    const disclosureOpen = panel.querySelector('details')?.open === true;
    const restoreRefreshFocus = document.activeElement === panel.querySelector('button');
    panel.dataset.restoreFocus = String(restoreRefreshFocus);
    panel.dataset.status = w.status;
    panel.replaceChildren(node('h3',null,'Acompanhamento da tarefa'),
      node('p',null,`Estado: ${statuses[w.status] || w.status}`),
      node('p',null,`Verificação: ${checks[w.verification] || 'Não informada'}`));
    if (w.lastObservation?.title) panel.append(node('p',null,`Última atividade: ${w.lastObservation.title}`));
    if (w.next?.reason) panel.append(node('p','workflow-next','Próximo passo recomendado: '+w.next.reason));
    if (w.pendingCriteria?.length) {
      const details=node('details'); details.open=disclosureOpen; details.append(node('summary',null,'Critérios pendentes'));
      const list=node('ul');
      for (const criterion of w.pendingCriteria) {
        const check=w.pendingChecks?.find(check=>check.id === criterion);
        list.append(node('li',null,check?.detail || criterion));
      }
      details.append(list); panel.append(details);
    }
    panel.append(node('small',null,`${w.eventCount} registros · ${w.recoveryAttempts} recuperações. Recomendações não executam ações.`));
    saveConversation();
  } catch (error) {
    if (ref === agentTaskRef && panel?.isConnected) {
      panel.replaceChildren(node('h3',null,'Acompanhamento temporariamente indisponível'),
        node('p',null,'A tarefa pode continuar no servidor. '+error.message));
    }
  } finally {
    workflowRequests.delete(ref);
    if (ref === agentTaskRef && panel?.isConnected) {
      const button=node('button',null,'Atualizar acompanhamento');
      button.onclick=()=>{ button.disabled=true; button.textContent='Atualizando…'; refreshAgentWorkflow(ref); }; panel.append(button);
      if(panel.dataset.restoreFocus === 'true') { button.focus({preventScroll:true}); panel.dataset.restoreFocus='false'; }
    }
  }
}
async function followAgentWorkflow(ref = agentTaskRef) {
  await refreshAgentWorkflow(ref);
  if (ref === agentTaskRef && ref?.status === 'running') {
    setTimeout(()=>followAgentWorkflow(ref),4000);
  }
}

async function restoreAgentClarification(ref = agentTaskRef) {
  if (!ref || ref !== agentTaskRef || ref.status !== 'clarifying') return;
  if (ref.clarification?.questions?.length) {
    renderAgentClarificationCard(ref.clarification, ref);
    return;
  }
  try {
    const data = await fetch(`/api/v1/agent/tasks/${encodeURIComponent(ref.taskId)}`, {cache:'no-store'}).then(response=>response.json());
    if (!data.ok || !data.clarification || ref !== agentTaskRef) return;
    ref.clarification = data.clarification;
    if (!ref.shown) {
      assistantReply(agentUnderstandingText(data.clarification), 'AgentCore · aguardando esclarecimento');
      ref.shown = true;
    }
    renderAgentClarificationCard(data.clarification, ref);
    saveConversation();
  } catch (_) { /* O cartão volta a aparecer quando a tarefa persistida estiver acessível. */ }
}

async function watchPersistedAgentTask(fallbackError = '') {
  const ref = agentTaskRef;
  if(!ref?.operationId || ref.status !== 'running') return;
  for(let attempt=0; attempt<900 && agentTaskRef===ref; attempt++) {
    try {
      if(!ref.taskId) {
        const listed=await fetch(`/api/v1/agent/tasks?operation_id=${encodeURIComponent(ref.operationId)}`,{cache:'no-store'}).then(response=>response.json());
        if(!listed.ok) throw new Error(listed.error || 'consulta da tarefa indisponível');
        ref.taskId=listed.tasks?.[0]?.taskId;
      }
      if(ref.taskId) {
        const data=await fetch(`/api/v1/agent/tasks/${encodeURIComponent(ref.taskId)}`,{cache:'no-store'}).then(response=>response.json());
        if(!data.ok) throw new Error(data.error || 'tarefa persistida indisponível');
        const status=data.task?.status;
        if(status && status!=='running') {
          ref.status=status;
          void refreshAgentWorkflow(ref);
          if(status==='clarifying') {
            ref.clarification=data.clarification || ref.clarification;
            if(!ref.shown && ref.clarification) {
              assistantReply(agentUnderstandingText(ref.clarification),'AgentCore · aguardando esclarecimento');
              ref.shown=true;
            }
            renderAgentClarificationCard(ref.clarification,ref);
            saveConversation();
            return;
          }
          if(!ref.shown) {
            const report=data.report ? {...data.report,events:data.events || []} : null;
            assistantReply(report ? agentReportText(report) : (data.task?.error || 'A tarefa foi interrompida; confira os efeitos antes de tentar novamente.'),
              ChatCore.agentReportMeta(report || {status}, status === 'awaiting_approval'));
            ref.shown=true;
          }
          if(status==='awaiting_approval') {
            const request=data.task?.request || {};
            pendingAgent={prompt:request.prompt,objective:request.objective,history:request.history || []};
            const approval=(data.events || []).findLast(event=>event.kind==='approval.required');
            const call=approval?.payload || {};
            const target=approval?.detail || call.tool || 'ação no workspace';
            const button=node('button','confirm-accept',`Revisar e aprovar ${call.tool || 'ação'}`);
            button.onclick=async()=>{
              const authorized=await confirmMutation(`Autorizar ${call.tool || 'ação'}?`,target,null,call,approval?.detail || 'A autorização vale somente para esta ação.');
              if(!authorized || !pendingAgent) return;
              const next=pendingAgent;
              pendingAgent=null;
              ref.status='blocked';
              button.remove();
              saveConversation();
              await pursueAgent(next.prompt,true,next.objective,next.history);
            };
            chat.append(button);
          }
          saveConversation();
          return;
        }
      }
    } catch(error) {
      if(attempt>=4 && !ref.taskId) {
        assistantReply(fallbackError || `Não foi possível consultar a tarefa: ${error.message}`, 'AgentCore indisponível');
        ref.status='interrupted'; saveConversation(); return;
      }
    }
    if(attempt===4 && !ref.taskId) {
      assistantReply(fallbackError || 'A tarefa não foi registrada no servidor.', 'AgentCore indisponível');
      ref.status='interrupted'; saveConversation(); return;
    }
    await new Promise(resolve=>setTimeout(resolve,1000));
  }
}

function shouldAutoPursueAgent(value, material = []) {
  if (!value || material.length || value.startsWith('/')) return false;
  if (!localStorage.getItem('ia-local-zero-workspace')) return false;
  if (ChatCore.isProjectFailureReport(value)) return true;
  if (/\b(?:refator\w*|extraia|separe|desacople|modularize|reorganize|simplifique|renomeie)\b/i.test(value)) return true;
  if (/^(?:como|o que|qual|quais|por que|porque|explique|me explique|pesquise|fale sobre|existe)\b/i.test(value)) return false;
  const intent = /\b(?:quero|preciso|vamos|pode|faça|faca|crie|construa|implemente|configure|corrija|edite|adicione|desenvolva|monte|prepare|inicie|ajude(?:-me)?\s+a)\b/i;
  const action = /\b(?:criar|crie|construir|construa|implementar|implemente|integrar|integre|configurar|configure|corrigir|corrija|editar|edite|alterar|altere|adicionar|adicione|desenvolver|desenvolva|montar|monte|planejar|planeje|investigar|investigue|preparar|prepare|iniciar|inicie)\b/i;
  return intent.test(value) && action.test(value);
}

    const panel = document.getElementById('tool-panel');
    const toolNames = {
      search: ['Pesquisar na internet', 'Consulte fontes externas e mantenha os resultados nesta área.'],
      url: ['Analisar uma página', 'Abra uma URL, extraia o texto e registre a fonte para esta sessão.'],
      workspace: ['Explorar workspace', 'Escolha qualquer diretório local existente para navegar e editar.'],
      code: ['Buscar no código', 'Encontre ocorrências em arquivos pequenos do workspace.'],
      sources: ['Fontes desta sessão', 'Revise, abra e acompanhe as fontes já coletadas.'],
      media: ['Multimídia', 'Packs locais para documentos, imagens, áudio e vídeo.']
    };
    function escapeHtml(value) { return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c])); }
    function panelShell(name, form, body='') { const [title, description] = toolNames[name]; panel.innerHTML = `<div class="panel-card"><div class="panel-head"><div><div class="panel-title">${title}</div><div class="panel-description">${description}</div></div><button class="panel-close" onclick="closeTool()">×</button></div>${form}<div class="panel-result" id="panel-result">${body}</div></div>`; panel.classList.add('open'); document.querySelectorAll('.side-item').forEach(item => item.classList.toggle('active', item.dataset.tool === name)); }
    function closeTool() { panel.classList.remove('open'); panel.innerHTML = ''; document.querySelectorAll('.side-item').forEach(item => item.classList.remove('active')); }
    function openTool(name) {
      if (name === 'search') panelShell(name, `<div class="panel-form"><input id="panel-input" placeholder="Ex.: novidades do Rust" autofocus><button onclick="runPanelSearch()">Pesquisar</button></div>`);
      else if (name === 'url') panelShell(name, `<div class="panel-form"><input id="panel-input" placeholder="https://exemplo.com" autofocus><button onclick="runPanelUrl()">Analisar</button></div>`);
      else if (name === 'workspace') panelShell(name, `<div class="panel-form"><button onclick="choosePanelWorkspace()">▣ Escolher pasta do projeto</button><button onclick="createPanelWorkspace()">＋ Criar projeto</button></div><div class="panel-form"><input id="panel-input" placeholder="ou informe um caminho absoluto" autofocus><button onclick="runPanelWorkspace()">Abrir</button></div><div class="panel-form"><button class="panel-action" onclick="showCreateFile()">＋ Novo arquivo</button><button class="panel-action" onclick="showCreateDirectory()">＋ Nova pasta</button></div><div class="panel-description" style="margin-top:10px">Escolha uma pasta do usuário pelo gerenciador de arquivos. O diretório do IA Local do Zero não precisa ser usado como workspace.</div>`);
      else if (name === 'code') panelShell(name, `<div class="panel-form"><input id="panel-input" placeholder="termo ou trecho de código" autofocus><button onclick="runPanelCode()">Buscar</button></div>`);
      else if (name === 'sources') { panelShell(name, '', '<div class="panel-empty">Carregando fontes…</div>'); runPanelSources(); }
      else if (name === 'media') {
        panelShell(name, `<div class="panel-form"><input id="media-path" placeholder="caminho relativo no workspace"><button onclick="inspectPanelMedia()">Inspecionar</button><button onclick="extractPanelDocument()">Extrair texto</button></div>`, '<div class="panel-empty">Consultando leitores locais…</div>');
        void renderMediaCapabilities();
      }
      document.getElementById('panel-input')?.addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); ({search:runPanelSearch,url:runPanelUrl,workspace:runPanelWorkspace,code:runPanelCode}[name] || (()=>{}))(); } });
    }
async function renderMediaCapabilities() {
  const target=document.getElementById('panel-result');
  try {
    const data=await fetch('/api/v1/attachments/capabilities',{cache:'no-store'}).then(response=>response.json());
    if(!document.getElementById('media-path') || target!==document.getElementById('panel-result')) return;
    if(!data.ok) throw new Error(data.error || 'Catálogo local indisponível.');
    const entries=[
      ['Documentos',`Texto e Office${data.documents?.pdf ? ', PDF' : '; leitor PDF indisponível'}`],
      ['Tabelas',data.tables?.csv_statistics ? 'CSV e TSV · estatísticas locais' : 'Sem análise de tabelas'],
      ['Imagens',data.images?.decode ? `Leitura de pixels${data.images?.ocr ? ' · OCR próprio experimental' : ' · OCR indisponível'}` : 'Leitor de imagens indisponível'],
      ['Áudio',data.audio?.wav_inspection ? 'WAV · duração, canais e sinal; transcrição ainda indisponível' : 'Leitor de áudio indisponível'],
      ['Vídeo',data.video?.mp4_container_inspection ? 'MP4 e MOV · metadados do contêiner; cenas ainda sem interpretação' : 'Leitor de vídeo indisponível'],
    ];
    target.replaceChildren();
    for(const [title,description] of entries) {
      const entry=node('div','entry'), label=node('div',null,title); label.append(node('small',null,description)); entry.append(label); target.append(entry);
    }
  } catch(error) { if(target===document.getElementById('panel-result')) target.textContent=error.message; }
}
function renderPanelSources(cards) {
  const target=document.getElementById('panel-result'); if(!target) return;
  target.replaceChildren();
  for(const card of cards || []) {
    const element=sourceCard(card), analyze=element.querySelector('.analyze');
    if(analyze) analyze.onclick=()=>runPanelOpen(card.source_id,card.url);
    target.append(element);
  }
  if(!target.children.length) target.append(node('div','panel-empty','Nenhuma fonte nesta sessão.'));
}
    async function runPanelSearch() { const query = document.getElementById('panel-input')?.value.trim(); if (!query) return; const target = document.getElementById('panel-result'); target.innerHTML = '<div class="panel-empty">Pesquisando…</div>'; const data = await toolRequest('search_web', {query}); if (!data.ok) { target.textContent = data.error; return; } renderPanelSources(data.data.results); }
    async function runPanelUrl() { const url = document.getElementById('panel-input')?.value.trim(); if (!url) return; const target = document.getElementById('panel-result'); target.innerHTML = '<div class="panel-empty">Abrindo e extraindo texto…</div>'; const data = await toolRequest('open_page', {url}); if (!data.ok) { target.textContent = data.error; return; } target.innerHTML = `<div class="source"><a href="${escapeHtml(data.data.url || url)}" target="_blank" rel="noreferrer">${escapeHtml(data.data.title || 'Página analisada')}</a><small>${escapeHtml((data.data.text || '').slice(0, 1800))}</small></div>`; }
async function runPanelWorkspace(path) {
  let value=typeof path==='string'?path:(document.getElementById('panel-input')?.value.trim() || '');
  const target=document.getElementById('panel-result'); if(!target) return;
  target.textContent='Lendo workspace…';
  let workspace=localStorage.getItem('ia-local-zero-workspace');
  if(value.startsWith('/')) {
    const selected=await toolRequest('set_workspace',{path:value});
    if(!selected.ok) { target.textContent=selected.error; return; }
    workspace=selected.data.workspace; setWorkspaceIndicator(workspace); value='';
  }
  const data=await toolRequest('list_files',{path:value,...(workspace?{_expected_workspace:workspace}:{})});
  if(workspace && workspace!==localStorage.getItem('ia-local-zero-workspace')) return;
  if(!data.ok) { target.textContent=data.error; return; }
  setWorkspaceIndicator(data.data.workspace);
  await renderDockDirectory(value);
  target.replaceChildren(node('div','panel-description',`Projeto ativo: ${data.data.workspace}`));
  if(value) { const up=node('button','panel-action','↑ Pasta anterior'); up.onclick=()=>runPanelWorkspace(value.split('/').slice(0,-1).join('/')); target.append(up); }
  for(const entry of data.data.entries) {
    const row=node('div','entry'), label=node('div',null,`${entry.kind==='directory'?'▣':'▤'} ${entry.name}`);
    label.append(node('small',null,`${entry.kind} · ${entry.bytes} bytes`));
    const button=node('button',null,entry.kind==='directory'?'Abrir':'Ler');
    const relative=(value?value+'/':'')+entry.name;
    button.onclick=()=>entry.kind==='directory'?openDockFolder(relative):runPanelRead(relative);
    row.append(label,button); target.append(row);
  }
  if(!data.data.entries.length) target.append(node('div','panel-empty','Diretório vazio.'));
}
async function choosePanelWorkspace() {
  const target=document.getElementById('panel-result'); if(target) target.textContent='Abrindo o gerenciador de arquivos…';
  try {
    const response=await fetch('/api/choose-directory',{cache:'no-store'}), data=await response.json();
    if(data.ok) { const input=document.getElementById('panel-input'); if(input) input.value=data.path; await runPanelWorkspace(data.path); }
    else if(!data.cancelled && target) target.textContent=data.error || 'Não foi possível escolher a pasta.';
    else if(target) target.textContent='Seleção cancelada.';
  } catch(error) { if(target) target.textContent=`Não foi possível abrir o gerenciador: ${error.message}`; }
}
async function createPanelWorkspace() {
  const value=document.getElementById('panel-input')?.value.trim(), target=document.getElementById('panel-result');
  if(!value || !value.startsWith('/')) { if(target) target.textContent='Informe um caminho absoluto para o novo projeto.'; return; }
  const data=await toolRequest('create_workspace',{path:value});
  if(!data.ok) { if(target) target.textContent=data.error; return; }
  setWorkspaceIndicator(data.data.workspace);
  await runPanelWorkspace('');
}
    function showCreateFile() { const target = document.getElementById('panel-result'); target.innerHTML = '<div class="panel-form"><input id="create-file-path" placeholder="caminho, ex.: app/main.py"><textarea id="create-file-content" placeholder="conteúdo inicial"></textarea><button onclick="runCreateFile()">Criar</button></div>'; }
    function showCreateDirectory() { const target = document.getElementById('panel-result'); target.innerHTML = '<div class="panel-form"><input id="create-directory-path" placeholder="caminho, ex.: app/src"><button onclick="runCreateDirectory()">Criar pasta</button></div>'; }
    function confirmMutation(action, path, signal=null, details=null, note='A ação só será executada depois da sua confirmação.') {
      return new Promise(resolve => {
        const card=node('section','approval-card inline-confirm');
        const copy=node('div','approval-content');
        const head=node('div','approval-head');
        const heading=node('div');
        heading.append(node('div','approval-eyebrow','AÇÃO NO WORKSPACE'),node('div','inline-confirm-title',action));
        head.append(node('span','approval-icon','!'),heading);
        copy.append(head,node('div','inline-confirm-path',path || 'Esta ação irá alterar o projeto ativo.'),node('div','inline-confirm-note',note));
        if (details && typeof details === 'object') {
          const disclosure=node('details','approval-details');
          const summary=node('summary',null,typeof details.content === 'string' ? 'Ver conteúdo da alteração' : 'Ver argumentos da ferramenta');
          const raw=typeof details.content === 'string' ? details.content : JSON.stringify(details,null,2);
          const toolbar=node('div','approval-preview-toolbar');
          const label=node('span',null,`${raw.split('\n').length} linhas de prévia`);
          const copyPreview=node('button','approval-copy','Copiar');
          copyPreview.type='button';
          copyPreview.onclick=()=>navigator.clipboard.writeText(raw).then(()=>{copyPreview.textContent='Copiado';setTimeout(()=>{copyPreview.textContent='Copiar';},1400);}).catch(()=>notice('Não foi possível copiar a prévia.'));
          toolbar.append(label,copyPreview);
          const preview=node('pre','approval-preview'); preview.textContent=raw;
          disclosure.append(summary,toolbar,preview);
          copy.append(disclosure);
        }
        const actions=node('div','inline-confirm-actions'); const cancel=node('button','confirm-cancel','Cancelar'); const accept=node('button','confirm-accept','Confirmar');
        const abort=()=>finish(false);
        const finish=value=>{ signal?.removeEventListener('abort',abort); card.remove(); resolve(value); };
        cancel.type='button'; accept.type='button'; cancel.onclick=()=>finish(false); accept.onclick=()=>finish(true); actions.append(cancel,accept); card.append(copy,actions); chat.append(card); scrollChat();
        signal?.addEventListener('abort',abort,{once:true});
        if(signal?.aborted) finish(false);
      });
    }
    async function runCreateFile() { const path = document.getElementById('create-file-path')?.value.trim(); const content = document.getElementById('create-file-content')?.value || ''; if (!path || !(await confirmMutation('Criar este arquivo?', path))) return; const data = await toolRequest('create_file', {path, content}); document.getElementById('panel-result').innerHTML = data.ok ? `<div class="panel-empty">Arquivo criado: ${escapeHtml(data.data.path)}</div>` : `<div class="panel-empty">${escapeHtml(data.error)}</div>`; }
    async function runCreateDirectory() { const path = document.getElementById('create-directory-path')?.value.trim(); if (!path || !(await confirmMutation('Criar esta pasta?', path))) return; const data = await toolRequest('create_directory', {path}); document.getElementById('panel-result').innerHTML = data.ok ? `<div class="panel-empty">Pasta criada: ${escapeHtml(data.data.path)}</div>` : `<div class="panel-empty">${escapeHtml(data.error)}</div>`; }
async function runPanelRead(path, startLine=null, endLine=null) {
  if (!document.body.classList.contains('chat-mode')) setMode('chat');
  const target=document.getElementById('panel-result');
  if(target) target.textContent='Lendo arquivo…';
  const workspace=localStorage.getItem('ia-local-zero-workspace');
  const data=await toolRequest('read_file',{path,...(workspace?{_expected_workspace:workspace}:{}),...(startLine ? {start_line:startLine,end_line:endLine || startLine+24} : {})});
  if(workspace!==localStorage.getItem('ia-local-zero-workspace')) return;
  if(!data.ok) { if(target) target.textContent=data.error; else notice(data.error); return; }
  setEditorFile(data.data);
  if(target) target.replaceChildren(node('strong',null,`${data.data.path}${data.data.start_line ? ` · linhas ${data.data.start_line}-${data.data.end_line} de ${data.data.total_lines}` : ''}`),node('pre',null,data.data.content));
}
async function runPanelCode() {
  const query=document.getElementById('panel-input')?.value.trim(), target=document.getElementById('panel-result'); if(!query || !target) return;
  target.textContent='Buscando no código…'; const data=await toolRequest('search_files',{query});
  if(!data.ok) { target.textContent=data.error; return; }
  target.replaceChildren();
  for(const match of data.data.matches) {
    const row=node('div','entry'), label=node('div',null,match.path); label.append(node('small',null,`linha ${match.line} · ${match.text}`));
    const button=node('button',null,'Abrir trecho'); button.onclick=()=>runPanelRead(match.path,Math.max(1,match.line-12),match.line+12); row.append(label,button); target.append(row);
  }
  if(!data.data.matches.length) target.append(node('div','panel-empty','Nenhuma ocorrência encontrada.'));
  if(data.data.truncated) target.append(node('div','panel-description','Busca parcial: o limite de leitura foi atingido.'));
}
    async function runPanelSources() { const data = await toolRequest('list_sources', {}); if (data.ok) renderPanelSources(data.data.sources); else document.getElementById('panel-result').textContent = data.error; }
    async function runPanelOpen(source_id, url) { const data = await toolRequest('open_page', {source_id, url}); if (data.ok) renderPanelSources([data.data]); }
    async function inspectPanelMedia() { const path = document.getElementById('media-path')?.value.trim(); if (!path) return; const target = document.getElementById('panel-result'); target.innerHTML = '<div class="panel-empty">Inspecionando mídia…</div>'; const data = await toolRequest('inspect_media', {path}); if (!data.ok) { target.textContent = data.error; return; } target.innerHTML = `<div class="entry"><div>${escapeHtml(data.data.path)}<small>${escapeHtml(data.data.media_type)} · ${data.data.extension || 'sem extensão'} · ${data.data.bytes} bytes</small></div></div>`; }
    async function extractPanelDocument() { const path = document.getElementById('media-path')?.value.trim(); if (!path) return; const target = document.getElementById('panel-result'); target.innerHTML = '<div class="panel-empty">Extraindo documento…</div>'; const data = await toolRequest('extract_document_text', {path}); if (!data.ok) { target.textContent = data.error; return; } const d=data.data, note=[d.status, ...(d.warnings||[])].filter(Boolean).join(' · '); target.innerHTML = `<div class="source"><strong>${escapeHtml(d.path)} · ${escapeHtml(note)}</strong><small style="white-space:pre-wrap">${escapeHtml((d.text||'').slice(0, 6000))}</small></div>`; }
async function toolRequest(tool, arguments_, showInChat=false) {
  const id=requestId('tool');
  return requestWithActivity('/api/tool-call',{tool,arguments:arguments_,request_id:id},`tool:${tool}:${id}`,toolLabels[tool] || tool,showInChat);
}
async function htmlPreviewForPath(path) {
  if(!/\.html?$/i.test(String(path||''))) return null;
  const result=await toolRequest('read_file',{path},false);
  return result.ok ? {html:result.data.content,path:result.data.path} : null;
}
async function call(tool,arguments_,identityOnly=false) {
  if(liveActions.size) return notice('Aguarde a operação atual terminar.');
  const data=await toolRequest(tool,arguments_,true);
  data.call_fingerprint=JSON.stringify([tool,arguments_ || {}]);
  if(!data.ok) { assistantReply(data.error, 'Operação não concluída'); return data; }
  const confidence = Number((data && data.tool_call && data.tool_call.planner && data.tool_call.planner.confidence) || data?.confidence || 0);
  const meta = responseMeta(`${toolLabels[tool] || tool} · ${data.elapsed_ms} ms`, data, confidence, toolLabels[tool] || tool);
  if(tool==='search_web') assistantReply(`Encontrei ${data.data.result_count} resultado(s) para “${data.data.query}”.`,meta,data.data.results);
  else if(tool==='research_web') {
    const pages=(data.data.pages || []).filter(page => page.text);
    const saved=data.data.saved_to_corpus ? `\n\nLote salvo em ${data.data.saved_to_corpus}.` : '';
    const savedCount=data.data.saved_count != null ? ` ${data.data.saved_count} registro(s) novo(s) foram gravados.` : '';
    const answer=data.data.answer || 'As fontes foram consultadas, mas não houve síntese disponível.';
    assistantReply(pages.length ? `${answer}\n\nPesquisa: ${pages.length} fonte(s) com conteúdo em ${data.data.elapsed_ms} ms.${saved}${savedCount}` : 'A pesquisa não retornou conteúdo verificável. Esta etapa não comprova a conclusão da tarefa.',meta,pages);
  }
  else if(tool==='inspect_project') {
    const d=data.data, list=(items,empty='nenhum identificado') => (items||[]).join(', ') || empty;
    const workspace=String(d.workspace||''), workspaceName=workspace.replace(/[\\/]+$/,'').split(/[\\/]/).pop()||workspace;
    if(identityOnly) {
      assistantReply(workspace?`O workspace que Brasa está usando nesta sessão é **${workspaceName}**.\n\nCaminho observado: \`${workspace}\`.`:'A inspeção local não devolveu o caminho do workspace, então não consigo confirmar o nome.',meta);
      return data;
    }
    const workspaceDetail=workspace?`Workspace ativo: **${workspaceName}**\nCaminho: \`${workspace}\`\n\n`:'';
    const next=(d.signals||[]).length ? `\n\nPontos para revisar:\n- ${d.signals.join('\n- ')}` : '\n\nA estrutura básica foi reconhecida; o próximo passo é ler os arquivos relevantes e executar os testes.';
    assistantReply(`${workspaceDetail}${d.summary}\n\nManifestos: ${list(d.manifests)}\nEntradas: ${list(d.entrypoints)}\nTestes: ${list(d.test_files)}\nVerificações disponíveis: ${list(d.checks)}${next}`,meta);
  }
  else if(tool==='diagnose_project') {
    const d=data.data, evidence=(d.evidence||[]).map(item=>`- ${item}`).join('\n'), steps=(d.next_steps||[]).map(item=>`- ${item}`).join('\n');
    const diagnostic={kind:'diagnostics',status:d.passed?'passed':'failed',check:d.check,passed:d.passed,summary:d.summary,evidence:d.evidence||[],stdout:d.stdout||'',stderr:d.stderr||''};
    assistantReply(`${d.summary||'Falha classificada.'}${evidence?`\n\nEvidências:\n${evidence}`:''}${steps?`\n\nPróximos passos:\n${steps}`:''}`,meta,[],null,diagnostic);
  }
  else if(tool==='read_file') {
    const d=data.data, artifact={kind:'code',path:d.path,status:'observed',language:(d.path?.match(/\.([a-z0-9]+)$/i)?.[1] || 'text').toLowerCase(),content:(d.content||'').slice(0,65536),lines_after:(d.content||'').split('\n').length,...(d.start_line ? {start_line:d.start_line,end_line:d.end_line,total_lines:d.total_lines} : {})};
    assistantReply(`Li ${d.path}${d.start_line ? `, linhas ${d.start_line}-${d.end_line} de ${d.total_lines}` : ''}. O conteúdo foi anexado como artefato revisável no histórico.`,meta,[],null,artifact);
  }
  else if(tool==='extract_document_text') {
    const d=data.data, artifact={kind:'document',path:d.path,status:d.status || 'extracted',language:d.format || 'text',content:(d.text||'').slice(0,65536),lines_after:(d.text||'').split('\n').length};
    const note=[d.truncated?'O texto foi limitado; peça uma seção específica para continuar.':'',...(d.warnings||[])].filter(Boolean).join('\n');
    const status=d.status==='ok'?'Extraí o texto':d.status==='no_text'?'Não encontrei uma camada de texto':'Não consegui extrair o texto';
    assistantReply(`${status} de ${d.path}.${note?`\n\n${note}`:''}`,meta,[],null,artifact);
  }
  else if(tool==='edit_file') {
    const d=data.data;
    const preview=await htmlPreviewForPath(d.path);
    assistantReply(`Edição aplicada em ${d.path}. O diff e o backup ficaram registrados na timeline.`,meta,[],preview,d.artifact);
  }
  else if(tool==='apply_batch') {
    const artifact=batchArtifactFromData(data.data || {});
    assistantReply(`${artifact.summary}\nUndo disponível: ${artifact.undo_available?'sim':'não'}.`,meta,[],null,artifact);
  }
  else if(tool==='undo_batch') {
    const d=data.data || {};
    const paths=(d.operations || []).map(item=>item.path).filter(Boolean);
    assistantReply(`Lote ${d.transaction_id || ''} desfeito em ${d.count || paths.length} operação(ões).${paths.length?`\n\nItens restaurados:\n- ${paths.join('\n- ')}`:''}`,meta);
  }
  else if(tool==='create_file') {
    const d=data.data;
    assistantReply(`Criei ${d.path}. O conteúdo inicial e o artefato ficaram registrados na timeline.`,meta,[],null,d.artifact);
  }
  else if(tool==='propose_repair') {
    const d=data.data;
    assistantReply(`Proposta validada para ${d.path}; nenhum arquivo foi alterado.\n\nMotivo: ${d.reason}\n\nRevise o diff estruturado e aplique com a ação de correção explícita.`,meta,[],null,d.artifact);
  }
  else if(tool==='apply_repair') {
    const d=data.data, reason=d.repair?.reason || 'não informado';
    assistantReply(`Correção aplicada em ${d.path}.\n\nMotivo: ${reason}\nBackup: ${d.backup || 'registrado pelo runtime'}\n\nA verificação do projeto será executada agora.`,meta,[],null,d.artifact);
  }
  else if(tool==='create_web_page') {
    const d=data.data;
    assistantReply(`Criei a página HTML em ${d.path}.\n\nO arquivo está no workspace e a prévia abaixo está isolada em sandbox.`,meta,[],{html:d.preview_html,path:d.path});
  }
  else if(tool==='terminal_run') {
    const d=data.data || {};
    const artifact={kind:'terminal',status:d.passed===false?'failed':'completed',command:d.command || '',exit_code:d.exit_code,timed_out:d.timed_out,passed:d.passed,summary:d.summary || `${d.command || 'Comando aprovado'} finalizado`,stdout:d.stdout || '',stderr:d.stderr || ''};
    assistantReply(`${artifact.summary}${d.elapsed_ms!=null?`\n\nTempo: ${d.elapsed_ms} ms.`:''}`,meta,[],null,artifact);
  }
  else if(['process_start','process_status','process_stop'].includes(tool)) {
    const artifact=processArtifact(tool,data.data || {});
    assistantReply(artifact.summary,meta,[],null,artifact);
  }
  else if(tool==='project_checks') {
    const d=data.data || {};
    if (d.executed === false && !(d.available||[]).length) assistantReply('Ainda não há testes ou verificações configurados neste projeto. Isso não significa que a alteração esteja quebrada; significa apenas que não existe um comando de validação reconhecido para executar.',meta);
    else if (d.passed === true) assistantReply('As verificações disponíveis foram executadas e passaram.',meta);
    else {
      const diagnostic={kind:'diagnostics',status:'failed',check:d.check || 'auto',passed:false,summary:d.message || 'As verificações foram executadas, mas uma falha foi encontrada.',stdout:d.stdout||'',stderr:d.stderr||'',evidence:d.evidence||[]};
      assistantReply(`As verificações foram executadas, mas encontrei uma falha.${d.message ? `\n\n${d.message}` : ''}`,meta,[],null,diagnostic);
    }
  }
  else if(tool==='list_sources') assistantReply(`${data.data.sources.length} fonte(s) nesta sessão.`,meta,data.data.sources);
  else if(tool==='open_page') assistantReply(`${data.data.title || 'Página lida'}\n\n${(data.data.text || '').slice(0,1800)}`,meta,[data.data]);
  else assistantReply(data.data?.summary || data.data?.message || 'A operação foi concluída. Consulte os detalhes da atividade acima para ver o resultado técnico.',meta);
  return data;
}
function toolResultForAgent(tool, result, traceId=null) {
  const data = result?.data ? JSON.parse(JSON.stringify(result.data)) : null;
  if (data && typeof data.preview_html === 'string') delete data.preview_html;
  return {tool, ok: Boolean(result?.ok), data, error: result?.error || null, call_fingerprint:result?.call_fingerprint || null, trace_id: traceId || result?.trace_id || null};
}
async function continueAgent(tool, result, depth=0, traceId=null) {
  conversation.push({role:'tool', tool, content:JSON.stringify(toolResultForAgent(tool,result,traceId))});
  saveConversation();
  if (depth >= MAX_AGENT_STEPS - 1) {
    assistantReply(`O orçamento de ${MAX_AGENT_STEPS} etapas foi atingido. O resultado das ações executadas permanece registrado acima; a tarefa foi pausada por segurança.`, 'Agente local · limite de segurança');
    return result;
  }
  const id=requestId('agent-step');
  const data=await requestWithActivity('/api/chat',{messages:requestMessages(),request_id:id,...workspaceRoutingContext()},`chat:${id}`,'Decidindo a próxima etapa',true);
  if(!data.ok) { assistantReply(data.error,'Continuação indisponível'); return data; }
  if(data.tool_call?.tool) {
    const request=data.tool_call;
    const currentTurn=conversation.slice(conversation.findLastIndex(item=>item.role==='user')+1);
    const observations=currentTurn.filter(item=>item.role==='tool').map(item=>{try{return JSON.parse(item.content);}catch{return {};}});
    const fingerprint=JSON.stringify([request.tool,request.arguments || {}]);
    if(observations.some(item=>item.call_fingerprint===fingerprint)) {
      assistantReply('O planejador repetiu uma chamada já executada com os mesmos argumentos. A execução foi interrompida para revisar o plano.', 'Agente · bloqueado por repetição');
      setLiveStatus({message:'Tarefa pendente: plano repetido',done:true});
      return {ok:false,agent:{status:'blocked',stop_reason:'repeated_call'}};
    }
    const confidence = Number((request && request.planner && request.planner.confidence) || data?.confidence || 0);
    assistantReply(actionIntro(request.tool, request.arguments || {}), withPlannerMeta('Próxima etapa', confidence, 'Próxima etapa'));
    // Mantém uma janela humana entre etapas encadeadas. O trabalho continua
    // assíncrono, mas o usuário consegue ler o plano e observar a transição.
    await new Promise(resolve=>setTimeout(resolve,900));
    if(toolNeedsApproval(request)) return runChatTool(request.tool,request.arguments || {},toolLabels[request.tool] || 'Executar próxima etapa',true,depth+1,data.trace_id || traceId);
    const next=await call(request.tool,request.arguments || {});
    return continueAgent(request.tool,next,depth+1,data.trace_id || traceId);
  }
  await new Promise(resolve=>setTimeout(resolve,650));
  const confidence = Number((data && data.tool_call && data.tool_call.planner && data.tool_call.planner.confidence) || data?.confidence || 0);
  assistantReply(data.text || 'A execução terminou sem uma conclusão verificável.', responseMeta(`${backendLabels[data.backend] || 'Agente local'} · ${data.elapsed_ms || 0} ms`, data, confidence, 'Agente local'),[],null,null,data.response_contract);
  if(data.agent?.status) setLiveStatus({message:({blocked:'Tarefa pendente',failed:'Execução falhou',completed:'Etapa concluída'})[data.agent.status] || data.agent.status,done:true});
  return data;
}
async function runChatTool(tool,arguments_,label,continueLoop=false,depth=0,traceId=null) {
  const terminalAction=tool==='terminal_run';
  const processAction=['process_start','process_stop'].includes(tool);
  const batchAction=['apply_batch','undo_batch'].includes(tool);
  const actionTarget=tool==='process_start'?'Perfil auto-dev · script de desenvolvimento reconhecido no workspace'
    :tool==='process_stop'?`Processo ${arguments_?.process_id || 'mais recente'} iniciado pelo runtime`
      :tool==='apply_batch'?`${arguments_?.operations?.length || 0} operação(ões) · diff por arquivo abaixo`
        :tool==='undo_batch'?`Lote ${arguments_?.transaction_id || 'mais recente'} · hashes serão conferidos antes de alterar`
      :terminalAction?'Workspace ativo · somente os perfis listados serão executados.':arguments_.path || '';
  const approvalNote=tool==='process_start'
    ? 'O perfil reconhecido pode executar código declarado no projeto e deixar o servidor ativo até ser encerrado.'
    :tool==='process_stop'?'Esta ação encerra o processo identificado pelo runtime.'
      :tool==='apply_batch'?'Revise o diff por arquivo. Todas as operações serão aplicadas juntas após sua aprovação.'
          :tool==='undo_batch'?'Qualquer arquivo alterado depois do lote será preservado; nesse caso a reversão será recusada.'
            :tool==='create_workspace'?'A pasta será criada e passará a ser o workspace ativo.'
          :'A ação só será executada depois da sua confirmação.';
  const approvalDetails=tool==='apply_batch'?{content:batchApprovalPreview(arguments_?.operations || [])}:tool==='create_web_page'?{content:arguments_?.prompt || ''}:terminalAction || processAction || batchAction || tool==='create_workspace'?arguments_:null;
  if(!(await confirmMutation(label+'?',actionTarget,null,approvalDetails,approvalNote))) { assistantReply(terminalAction?'Tudo bem — o comando não foi executado.':processAction?'Tudo bem — o processo não foi iniciado nem encerrado.':batchAction?'Tudo bem — o lote não foi aplicado nem desfeito.':'Tudo bem — não alterei nenhum arquivo.', 'Ação cancelada'); return; }
  const result=await call(tool,arguments_);
  if(result?.ok && !continueLoop && (tool==='create_file' || tool==='create_web_page' || tool==='edit_file' || tool==='apply_repair')) {
    const verification=await call('project_checks',{check:'auto',path:result?.data?.path || arguments_?.path || ''});
    if(verification?.ok && verification.data?.executed!==false && verification.data?.passed===false) await call('diagnose_project',verification.data);
  }
  if(continueLoop && result) return continueAgent(tool,result,depth,traceId);
  return result;
}
function terminalOperationFromInput(value) {
  const command=value.replace(/^\/terminal\s*/i,'').trim().replace(/^`|`$/g,'').replace(/\s+/g,' ');
  if(/^(?:git )?status(?: --short)?$/i.test(command) || /^git status(?: --short)?$/i.test(command)) return {operation:'git_status'};
  if(/^git diff --stat$/i.test(command)) return {operation:'git_diff_stat'};
  if(/^cargo test(?: --all-targets)?$/i.test(command)) return {operation:'project_check',check:'cargo-test'};
  if(/^npm test$/i.test(command)) return {operation:'project_check',check:'npm-test'};
  if(/^npm run check$/i.test(command)) return {operation:'project_check',check:'npm-check'};
  if(/^(?:pytest(?: -q)?|python3? -m pytest)$/i.test(command)) return {operation:'project_check',check:'pytest'};
  if(/^python3? -m unittest$/i.test(command)) return {operation:'project_check',check:'unittest'};
  return null;
}
function parseRepairCommand(value) {
  const match=value.match(/^\/(propor|aplicar)\s+corre[cç][aã]o\s+(?:no\s+|para\s+o\s+|para\s+)?arquivo\s+([^\s:]+)\s*\n\s*motivo\s*:\s*\n([\s\S]*?)\n\s*antigo\s*:\s*\n([\s\S]*?)\n\s*novo\s*:\s*\n([\s\S]+)$/i);
  if(!match) return null;
  return {tool:match[1].toLowerCase()==='propor'?'propose_repair':'apply_repair',arguments:{path:match[2].replace(/`/g,''),reason:match[3].trim(),old_text:match[4],new_text:match[5]}};
}
const MAX_CHAT_HISTORY_BYTES=1536*1024;
const MAX_CHAT_TAIL_BYTES=1024*1024;
function utf8Bytes(value) { return new TextEncoder().encode(value).length; }
function summarizeBrowserHistory(messages) {
  const signal=/(objetivo|meta|decid|combin|prefir|precis|requisit|restri|limite|penden|pr[oó]xim|conclu|resultad|falhou|erro|problem|importante|quero|projeto|checkpoint|contexto|n[aã]o pode|deve|\/)/i;
  const candidates=[],seen=new Set();
  for(let index=0;index<messages.length;index++) {
    const message=messages[index];
    if(!['user','assistant'].includes(message.role)) continue;
    const parts=String(message.content||'').split(/(?<=[!?])\s+|(?<=\.)\s+(?=[A-ZÁÉÍÓÚÀ-Ý0-9])|\n+/);
    for(const raw of parts) {
      const sentence=raw.trim().replace(/^[-•\s]+/,'');
      if(sentence.length<16) continue;
      const normalized=sentence.toLocaleLowerCase('pt-BR').replace(/[^\p{L}\p{N}]+/gu,' ').trim();
      if(!normalized||seen.has(normalized)) continue;
      seen.add(normalized);
      const important=signal.test(sentence);
      const goal=/\b(objetivo|meta|decidimos|requisito|restri[cç][aã]o|limite)\b/i.test(sentence);
      const technical=/\b(api|python|rust|typescript|javascript|mem[oó]ria|modelo|agente|token|teste)\b/i.test(sentence)||/(?:[\w.-]+\/){1,}[\w./-]+/.test(sentence);
      if(!important&&!technical) continue;
      const score=(message.role==='user'?4:2)+(important?4:0)+(goal?4:0)+(technical?2:0)+Math.min(index/100,2);
      candidates.push({index,score,line:`- ${message.role==='user'?'Usuário':'Agente'}: “${sentence.slice(0,420)}”`});
    }
  }
  candidates.sort((a,b)=>b.score-a.score||b.index-a.index);
  const selected=[];let size=0;
  for(const candidate of candidates) {
    const bytes=utf8Bytes(candidate.line+'\n');
    if(size+bytes>24000) continue;
    selected.push(candidate);size+=bytes;
  }
  selected.sort((a,b)=>a.index-b.index);
  return selected.length
    ? 'MEMÓRIA COMPACTADA DA SESSÃO — referência histórica; confirme no pedido atual antes de agir:\n'+selected.map(item=>item.line).join('\n')
    : 'Histórico anterior resumido; preserve o pedido atual e pergunte se faltar um requisito.';
}
function requestMessages() {
  // Envia toda a conversa enquanto cabe no limite do worker. Em sessões longas,
  // resume o prefixo e mantém recentes mensagens exatas sob o teto HTTP.
  const latest=conversation.findLastIndex(m=>m.attachments?.length);
  const full=conversation.map((m,index)=>({role:m.role,content:String(m.content||'').slice(0,index===conversation.length-1?64000:24000),attachments:index===latest ? m.attachments : []}));
  if(utf8Bytes(JSON.stringify(full))<=MAX_CHAT_HISTORY_BYTES) return full;

  let tailStart=full.length,tailBytes=2;
  for(let index=full.length-1;index>=0;index--) {
    const size=utf8Bytes(JSON.stringify(full[index]))+1;
    if(tailStart<full.length&&tailBytes+size>MAX_CHAT_TAIL_BYTES) break;
    tailStart=index;tailBytes+=size;
    if(tailBytes>=MAX_CHAT_TAIL_BYTES) break;
  }
  const older=full.slice(0,tailStart),recent=full.slice(tailStart);
  const summary={role:'user',content:summarizeBrowserHistory(older),attachments:[]};
  return [summary,...recent];
}
function recentAgentHistory() {
  return requestMessages()
    .filter(message=>message.role==='user'||message.role==='assistant')
    .slice(-24)
    .map(message=>({role:message.role,content:String(message.content||'').slice(0,6000)}));
}
function mentionsAttachedProject(value) {
  if(!conversation.some(message=>message.role==='user' && message.attachments?.length)) return false;
  if(/\b(?:workspace|projeto\s+ativo|meu\s+projeto)\b/i.test(value)) return false;
  const projectTopic=/\b(?:projeto|sistema|aplicativo|arquivos?|c[oó]digo|testes?)\b/i.test(value);
  const attachedReference=/\b(?:anexos?|deste|desse|este|esse|daqueles?)\b/i.test(value);
  return projectTopic && (attachedReference || !localStorage.getItem('ia-local-zero-workspace'));
}
function asksToChangeAttachedProject(value) {
  return mentionsAttachedProject(value)
    && /\b(?:crie|criar|edite|editar|corrija|corrigir|implemente|implementar|altere|alterar|apague|apagar|remova|remover|execute|executar|rode|rodar)\b/i.test(value);
}
function refersToAttachedProject(value) {
  return mentionsAttachedProject(value) && !asksToChangeAttachedProject(value);
}
function createStreamingDraft() {
  const draft=addMessage('assistant','','Rascunho ao vivo · validando a resposta',[],false);
  draft.row.classList.add('streaming');
  return draft;
}
function updateStreamingDraft(draft, text) {
  if(!draft || !text) return;
  draft._streamText=text;
  if(draft._streamFrame) return;
  const paint=()=>{
    draft._streamFrame=0;
    if(!draft.row.isConnected) return;
    const value=draft._streamText || '';
    draft.bubble.innerHTML=ChatCore.formatMarkdown(value);
    draft.meta.textContent='Rascunho ao vivo · validando a resposta';
    scrollChat();
  };
  // The model can emit several tiny fragments per frame. Coalescing them into
  // one paint keeps markdown parsing and layout work below the display rate.
  draft._streamFrame=window.requestAnimationFrame ? window.requestAnimationFrame(paint) : setTimeout(paint,16);
}
function finishStreamingDraft(draft, text, meta, contract=null,studio=null) {
  if(!draft) return assistantReply(text,meta,[],null,null,contract,studio);
  if(draft._streamFrame) {
    if(window.cancelAnimationFrame) window.cancelAnimationFrame(draft._streamFrame);
    else clearTimeout(draft._streamFrame);
    draft._streamFrame=0;
  }
  const entry={kind:'assistant',text,meta,cards:[],preview:null,artifact:null,contract:contract || undefined,studio:studio || undefined};
  conversation.push({role:'assistant',content:text,contract:contract || undefined});
  timeline.push(entry);
  draft.bubble.innerHTML=ChatCore.formatMarkdown(text);
  draft.meta.textContent=meta;
  draft.row.classList.remove('streaming');
  draft.row.classList.add('stream-complete');
  const contractCard=renderResponseContract(contract);
  if(contractCard) draft.body.append(contractCard);
  if(studio) renderStudioActions(draft.body,studio);
  saveConversation(); scrollChat();
}
async function chatModel(options={}) {
  const id=requestId('chat');
  let draft=null, streamed='';
  const messages=requestMessages();
  if(options.interfaceIdeas && messages.length) {
    const last=messages.at(-1);
    if(last?.role==='user') last.content+=`\n\n[Orientação do Estúdio de Interfaces: organize um briefing curto; apresente exatamente três direções visuais numeradas, cada uma com nome, paleta, composição e sensação. Use títulos no formato exato “1. Nome da direção”, “2. Nome da direção” e “3. Nome da direção”. Recomende uma. Não gere código, não chame ferramentas e não altere arquivos. Termine pedindo que a pessoa escolha uma direção.]`;
  }
  if(options.interfaceEditPath && messages.length) {
    const last=messages.at(-1);
    if(last?.role==='user') last.content+=`\n\n[Modo de iteração do Estúdio: a pessoa quer ajustar o arquivo HTML ${options.interfaceEditPath}. Aplique somente as mudanças que ela descreveu, preservando o objetivo e o restante da página. Leia o arquivo e proponha uma edição exata; se os ajustes não estiverem claros, pergunte antes de alterar. Não crie outro arquivo.]`;
  }
  const data=await requestChatStream({messages,request_id:id},`chat:${id}`,'Gerando resposta em tempo real',fragment=>{
    streamed+=fragment;
    if(!draft) draft=createStreamingDraft();
    updateStreamingDraft(draft,streamed);
  });
  if(!data.ok) { draft?.row.remove(); return assistantReply(data.error,'Resposta indisponível'); }
  if(data.tool_call?.tool) {
    draft?.row.remove();
    const request=data.tool_call;
    const confidence = Number((request && request.planner && request.planner.confidence) || data?.confidence || 0);
    assistantReply(actionIntro(request.tool, request.arguments || {}), withPlannerMeta('Plano da tarefa', confidence, 'Plano da tarefa'));
    if(toolNeedsApproval(request)) return runChatTool(request.tool,request.arguments,toolLabels[request.tool] || 'Executar ação',true,0,data.trace_id);
    const result=await call(request.tool,request.arguments || {});
    return continueAgent(request.tool,result,0,data.trace_id);
  }
  const confidence = Number((data && data.tool_call && data.tool_call.planner && data.tool_call.planner.confidence) || data?.confidence || 0);
  const answer=data.text || 'Nenhum texto retornado.';
  const meta=responseMeta(`${backendLabels[data.backend] || 'Assistente local'} · ${data.elapsed_ms} ms`, data, confidence, 'Resposta');
  const studio=options.interfaceIdeas ? {kind:'interface-directions/v1',idea:options.idea,directions:answer} : null;
  finishStreamingDraft(draft,answer,meta,data.response_contract,studio);
}
async function watchAgentRun() {
  if(!agentRunRef || agentRunRef.terminal || liveActions.has(agentRunRef.id)) return;
  const ref=agentRunRef, operation=ref.id;
  startAction(operation,'Tarefa no servidor',true);
  const row=liveActions.get(operation)?.row;
        const approvalAbort=new AbortController();
  const cancel=node('button',null,'Cancelar tarefa');
  const control=async (action,call_id)=>{
    const result=await fetch(`/api/runs/${operation}`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({action,call_id})}).then(r=>r.json());
    if(!result.ok) throw new Error(result.error);
    return result;
  };
  cancel.onclick=()=>control('cancel').then(()=>approvalAbort.abort()).catch(error=>notice(error.message));
  row?.append(cancel);
  try {
    while(agentRunRef===ref) {
      let data;
      for (let attempt=0; attempt<4; attempt++) {
        try {
          const response=await fetch(`/api/runs/${operation}`,{cache:'no-store',signal:AbortSignal.timeout(15000)});
          data=await response.json();
          break;
        } catch(error) {
          if (attempt===3) throw error;
          await new Promise(resolve=>setTimeout(resolve,500*(attempt+1)));
        }
      }
      if(!data.ok) throw new Error(data.error);
      const run=data.run;
      renderTaskContract(operation, run.contract);
      for(const event of run.events.filter(event=>event.seq>ref.cursor)) {
        updateAction({operation,phase:'running',message:event.text});
        if(event.plan) {
          const routing=event.skill_routing;
          const routingText=routing?`Skills: ${(routing.skills || []).map(skill=>skill.label).join(', ') || 'nenhuma'} · API: ${routing.selected_api?.id || 'sem API selecionada'}\n`:'';
          assistantReply(routingText+event.text+'\n\n'+event.plan.map((call,index)=>`${index+1}. ${toolLabels[call.tool] || call.tool}${call.path?' · '+call.path:''}`).join('\n'),'Plano da tarefa');
        }
        if(event.result) {
          const result=event.result, d=result.data || {};
          const processTool=['process_start','process_status','process_stop'].includes(result.tool);
          const artifact=d.artifact || (result.tool==='read_file'?{kind:'code',path:d.path,content:d.content,status:'observed'}:result.tool==='extract_document_text'?{kind:'document',path:d.path,content:d.text || '',language:d.format || 'text',status:d.status || 'extracted',lines_after:(d.text || '').split('\n').length}:result.tool==='project_checks'?{kind:'diagnostics',status:d.passed && d.executed?'passed':'failed',check:d.check,passed:d.passed,summary:d.message || d.command,stdout:d.stdout || '',stderr:d.stderr || ''}:result.tool==='terminal_run'?{kind:'terminal',status:d.passed===false?'failed':'completed',command:d.command || '',exit_code:d.exit_code,timed_out:d.timed_out,passed:d.passed,summary:d.summary || d.command,stdout:d.stdout || '',stderr:d.stderr || ''}:result.tool==='apply_batch'?batchArtifactFromData(d):processTool?processArtifact(result.tool,d):null);
          const cards=result.tool==='research_web'?(d.pages || []):result.tool==='search_web'?(d.results || []):[];
          const inspectedWorkspace=result.tool==='inspect_project' && d.workspace ? `Workspace ativo: ${String(d.workspace).replace(/[\\/]+$/,'').split(/[\\/]/).pop()} · ${d.workspace}. ${d.summary || ''}` : null;
          const summary=processTool || result.tool==='apply_batch' ? artifact?.summary : result.tool==='undo_batch' ? `Lote ${d.transaction_id || ''} desfeito.` : inspectedWorkspace || d.summary || d.message || d.path || 'resultado registrado';
          assistantReply(`${toolLabels[result.tool] || result.tool}: ${result.ok ? summary : (result.error || 'falha')}`,'Resultado da ferramenta',cards,null,artifact);
        }
        if(event.response || ['failed','blocked','interrupted','cancelled'].includes(event.status)) assistantReply(event.text,`Tarefa · ${event.status}`);
        ref.cursor=event.seq; saveConversation();
      }
      if(['completed','blocked','failed','cancelled','interrupted'].includes(run.status)) {
        ref.terminal=true;
        saveConversation();
        updateAction({operation,phase:run.status==='completed'?'done':'error',message:run.events.at(-1)?.text || run.status},true);
        if(run.status==='interrupted' && !run.uncertain_call) {
          const resume=node('button',null,'Retomar tarefa');
          resume.onclick=async()=>{try{await control('resume');resume.remove();await watchAgentRun();}catch(error){notice(error.message);}};
          row?.append(resume);
        }
        break;
      }
      if(run.status==='awaiting_approval') {
        const call=run.pending;
        const arguments_=call.arguments || {};
        const target=call.tool==='process_start'?'Perfil auto-dev · scripts reconhecidos em package.json/Cargo.toml'
          :call.tool==='process_stop'?`Processo ${arguments_.process_id || 'mais recente'} iniciado pelo runtime`
            :arguments_.path || arguments_.workspace || `Ferramenta ${call.tool}`;
        const note=call.tool==='process_start'
          ? 'O perfil pode executar código declarado pelo projeto e manter o servidor ativo até ser encerrado.'
          :call.tool==='process_stop'?'O runtime encerrará somente o processo que iniciou e seu grupo de filhos.':'A ação só será executada depois da sua confirmação.';
        const approved=await confirmMutation(`Autorizar ${toolLabels[call.tool] || call.tool}?`,target,approvalAbort.signal,arguments_,note);
        await control(approved?'approve':'cancel',call.id);
      } else await new Promise(resolve=>setTimeout(resolve,700));
    }
  } catch(error) {
    updateAction({operation,phase:'error',message:`Acompanhamento indisponível: ${error.message}. A tarefa pode continuar no servidor; reabra esta conversa para consultar.`},true);
  } finally {cancel.remove();liveActions.delete(operation);refreshControls();saveConversation();}
}
function renderTaskContract(operation, contract) {
  if(!contract) return;
  const action=liveActions.get(operation), row=action?.row;
  if(!row) return;
  const body=row.querySelector('.action-body'); if(!body) return;
  let details=body.querySelector('[data-task-contract]');
  if(!details) {
    details=node('details','task-contract');
    details.dataset.taskContract='true';
    body.insertBefore(details, body.querySelector('.action-steps') || null);
  }
  const criteria=(contract.acceptance_criteria || []).map(item=>node('li','',item.text || item.id));
  const status=contract.status || 'draft';
  details.replaceChildren(node('summary',null,`Contrato · ${status}`));
  const grid=node('div','task-contract-grid');
  grid.append(node('span',null,`Intenção: ${contract.intent || 'indefinida'}`));
  grid.append(node('span',null,`Risco: ${contract.risk || 'indefinido'}`));
  grid.append(node('span',null,`Efeito: ${contract.side_effects || 'indefinido'}`));
  grid.append(node('span',null,`Aprovação: ${contract.requires_approval ? 'necessária' : 'não necessária'}`));
  details.append(grid);
  if(criteria.length) { const list=node('ul','task-contract-criteria'); list.append(...criteria); details.append(list); }
}
async function learnTopic(topic, routeTask=`aprenda ${topic}`) {
  const operation=requestId('learn');
  startAction(operation,`Aprendendo: ${topic}`,true);
  const action=liveActions.get(operation);
  action.record.progress=0;
  const bar=node('progress','learning-progress'); bar.max=100; bar.value=0;
  action.row?.querySelector('.action-body')?.insertBefore(bar,action.row.querySelector('details'));
  try {
    let routeMessage='Skill: Pesquisa e aprendizado verificável';
    let learningStartApi={method:'POST',path:'/api/learn'};
    let learningStatusPath='/api/learn/{job_id}';
    try {
      const routeResponse=await fetch('/api/v1/skills/route',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({task:routeTask})});
      const route=await routeResponse.json();
      if(routeResponse.ok && route.ok) {
        const skill=(route.selected_skills||[]).find(item=>item.id==='knowledge-learning');
        const api=(route.api_candidates||[]).find(item=>item.id==='learning.topic.start');
        const statusApi=(route.api_candidates||[]).find(item=>item.id==='learning.topic.status');
        if(skill && api) {
          learningStartApi={method:api.method,path:api.path};
          if(statusApi) learningStatusPath=statusApi.path;
          routeMessage=`Skill: ${skill.label} · API: ${api.method} ${api.path} · rede: ${api.network_policy}`;
        }
        else routeMessage='Skill de aprendizado explícita · rota do catálogo indisponível; usando o endpoint dedicado.';
      } else routeMessage='Skill de aprendizado explícita · roteador indisponível; usando o endpoint dedicado.';
    } catch (_) { routeMessage='Skill de aprendizado explícita · roteador indisponível; usando o endpoint dedicado.'; }
    updateAction({operation,phase:'learning',message:routeMessage,progress:0});
    const started=await fetch(learningStartApi.path,{method:learningStartApi.method,headers:{'content-type':'application/json'},body:JSON.stringify({topic})}).then(response=>response.json());
    if(!started.ok) throw new Error(started.error || 'Não foi possível iniciar.');
    let job=started.job, seen=0;
    while(job.status==='running') {
      for(const entry of job.logs.slice(seen)) updateAction({operation,phase:'learning',message:entry.message,progress:job.progress});
      seen=job.logs.length;
      await new Promise(resolve=>setTimeout(resolve,850));
      const statusUrl=learningStatusPath.replace('{job_id}',encodeURIComponent(job.id));
      const response=await fetch(statusUrl,{cache:'no-store'});
      const data=await response.json();
      if(!data.ok) throw new Error(data.error || 'Não foi possível consultar o progresso.');
      job=data.job;
      // O cartão de atividade é atualizado em tempo real. Evitamos remontar
      // o painel inteiro de competências a cada polling para preservar o chat.
    }
    for(const entry of job.logs.slice(seen)) updateAction({operation,phase:job.status,message:entry.message,progress:job.progress});
    updateAction({operation,phase:job.status==='completed'?'done':'error',message:job.logs.at(-1)?.message || 'Aprendizado encerrado.',progress:100},true);
    if(job.status==='completed') {
      const links=(job.sources || []).filter(source=>source.url).map(source=>`- ${source.title || source.url}: ${source.url}`).join('\n');
      const verification=job.evidence?.status==='corroborated' ? 'Fontes de domínios distintos corroboram o tema; isso não é prova absoluta de todas as afirmações.' : 'Evidência provisória: as fontes ainda não foram corroboradas de forma independente.';
      const skillStatus=job.skill?.status || 'partially_known';
      const lab=job.laboratory?.status==='verified' ? 'Laboratório: exercícios executados e verificados.' : job.laboratory?.status==='partial' ? 'Laboratório: algumas competências foram praticadas; outras ainda não têm executor local seguro.' : job.laboratory?.status==='recovered' ? 'Laboratório: houve falha, mas uma etapa de recuperação foi aprovada.' : job.laboratory?.status==='practice_unavailable' ? 'Laboratório: ainda não há executor local seguro para este domínio.' : 'Laboratório: exercício falhou mesmo após recuperação; a competência permanece parcial.';
      assistantReply(`Pesquisa de “${topic}” concluída e competência registrada como “${skillStatus}”. Foram adicionados ${job.documents_added} documento(s); o chat já pode consultar esse material, mas ainda não considera o tema dominado.\n\nVerificação: ${verification}\n${lab}\n\nFontes consultadas:\n${links || 'Nenhuma fonte registrada.'}\n\nLacunas: ${(job.gaps || job.skill?.concepts?.gaps || []).join(', ') || 'avaliação ainda incompleta'}.\n\nIsso atualiza o índice e o estado do agente; não treina os pesos do modelo.`, 'Aprendizado parcial');
      if(document.body.classList.contains('training-mode')) await loadTraining();
    } else assistantReply(`Não consegui concluir o aprendizado de “${topic}”: ${job.error || 'nenhuma fonte válida'}.`, 'Aprendizado incompleto');
  } catch(error) {
    updateAction({operation,phase:'error',message:`Aprendizado interrompido: ${error.message}`},true);
    assistantReply(`Não consegui acompanhar o aprendizado: ${error.message}`, 'Aprendizado interrompido');
  } finally { liveActions.delete(operation); refreshControls(); saveConversation(); if(document.body.classList.contains('training-mode')) loadTraining(); }
}
async function learnMaterials(urls) {
  const unique=[...new Set(urls.map(url=>url.replace(/[),.;]+$/g,'')))].slice(0,8);
  const operation=requestId('material-plan');
  startAction(operation,`Preparando trilha com ${unique.length} fontes`,true);
  const action=liveActions.get(operation);
  updateAction({operation,phase:'learning',message:`Foram identificadas ${unique.length} fontes. Vou processá-las uma por vez.`,progress:0});
  updateAction({operation,phase:'done',message:'Trilha de fontes preparada; iniciando estudo sequencial.',progress:0},true);
  liveActions.delete(operation); refreshControls();
  for (let index=0; index<unique.length; index++) {
    assistantReply(`Fonte ${index+1}/${unique.length}: vou estudar e validar esta referência antes de avançar.\n${unique[index]}`,'Trilha de treinamento');
    await learnTopic(unique[index]);
    await new Promise(resolve=>setTimeout(resolve,900));
  }
}
async function send() {
  if(sending || liveActions.size) return;
  const value=input.value.trim(); if(!value && !attachments.length) return;
  if(value.length>24000) return notice('A mensagem excede 24 mil caracteres; envie o conteúdo longo como arquivo.');
  const interfaceIdeas=interfaceStudioDraft;
  interfaceStudioDraft=false;
  const interfacePreview=pendingInterfacePreview;
  pendingInterfacePreview=null;
  const interfaceEditPath=interfaceEditPathDraft;
  interfaceEditPathDraft=null;
  sending=true; refreshControls();
  try {
    await Promise.all(attachments.map(item=>item.ready));
    if(attachments.some(item=>item.state!=='ready')) { notice('Um anexo não terminou de carregar. Remova-o e tente novamente.'); return; }
    const material=attachments.map(item=>item.data);
    if(material.reduce((sum,item)=>sum+item.bytes,0)>ChatCore.LIMITS.totalBytes) { notice('Os anexos ultrapassam 256 KiB de texto nesta mensagem. Envie-os em mensagens separadas.'); return; }
    if(material.reduce((sum,item)=>sum+(item.raw_bytes || 0),0)>ChatCore.LIMITS.mediaTotalBytes) { notice('As mídias ultrapassam 64 MiB nesta mensagem. Envie-as em mensagens separadas.'); return; }
    const tool=ChatCore.route(value,material.length>0,selectedTool);
    conversation.push({role:'user',content:value,attachments:material});
    const visible=[value,...material.map(item=>`📎 ${item.name} · ${item.files.length} arquivo(s) lido(s) · ${item.omitted} omitido(s)`) ].filter(Boolean).join('\n');
    addMessage('user',visible); saveConversation(); input.value=''; input.style.height='auto'; clearAttachments(); selectedTool=null; renderAttachments();
    // Aceita o tema e a URL em linhas separadas. Sem [\s\S], esse comando
    // caía no chat comum e terminava em milissegundos sem iniciar o estudo.
    const learning=value.match(/^\/?(?:aprenda|estude|estudar)\s+["“]?([\s\S]+?)["”]?\s*$/i);
    const routedLearning=learning ? null : value.match(/\b(?:aprenda|estude|estudar|quero aprender|me ensine|ensine-me)\s+(?:sobre\s+)?([\s\S]+)$/i);
    const learningTopic=(learning?.[1] || routedLearning?.[1] || '').trim();
    const compoundLearning=learningTopic && /\s+e\s+(?:crie|criar|implemente|implementar|construa|desenvolva|corrija|use|utilize)\b/i.test(learningTopic);
    const materialUrls=[...value.matchAll(/https?:\/\/[^\s<>"'\]]+/g)].map(match=>match[0]);
    const agentCommand=value.match(/^\/agente(?:\s+([\s\S]*))?$/i);
    const routeCommand=value.match(/^\/rotear(?:\s+([\s\S]*))?$/i);
    if(routeCommand) {
      const task=routeCommand[1]?.trim() || '';
      if(task) await previewSkillRoute(task);
      else assistantReply('Use `/rotear <tarefa>` para resolver a skill, ver o próximo passo e as APIs locais candidatas. O comando mostra o plano sem executar ferramentas.','Roteamento local');
    }
    else if(/^\/terminal(?:\s|$)/i.test(value)) {
      const args=terminalOperationFromInput(value);
      if(args) await runChatTool('terminal_run',args,toolLabels.terminal_run);
      else assistantReply('O terminal aceita apenas estes perfis: `git status --short`, `git diff --stat`, `cargo test`, `npm test`, `pytest` ou `python -m unittest`. Comandos livres e scripts arbitrários não são executados.','Terminal local');
    }
    else if (agentCommand) {
      const command=agentCommand[1]?.trim() || '';
      if (/^aprovar$/i.test(command)) {
        if(agentTaskRef?.status==='awaiting_approval' && agentTaskRef.approval)
          await resumeAgentDecision(agentTaskRef,agentTaskRef.approval,'approve');
        else {
          const next=pendingAgent; pendingAgent=null;
          if (next) await pursueAgent(next.prompt, true, next.objective, next.history);
          else assistantReply('Não há uma execução do AgentCore aguardando aprovação.', 'AgentCore');
        }
      } else if (!command) {
        assistantReply('Use `/agente <objetivo>` depois de selecionar um workspace. A aprovação aparecerá dentro do chat.', 'AgentCore');
      } else await pursueAgent(command, false);
    }
    else if(tool===null && !material.length && asksToChangeAttachedProject(value))
      assistantReply('Para alterar ou executar o projeto anexado, selecione a pasta dele como workspace e mencione “meu projeto ativo”. O anexo permite análise, mas não identifica um workspace seguro para escrita.', 'Workspace necessário');
    else if(tool===null && !material.length && refersToAttachedProject(value)) await chatModel();
    else if(interfacePreview && tool===null && !material.length) await runChatTool('create_web_page',interfacePreview,'Criar prévia');
    else if(interfaceEditPath && tool===null && !material.length) await pursueAgent(
      `${value}\n\n[Modo de iteração do Estúdio: ajuste o arquivo HTML ${interfaceEditPath} conforme o pedido, preserve o restante da página, leia o arquivo antes de propor a edição e peça esclarecimento se faltar informação.]`,
      false, 'auto', recentAgentHistory());
    else if(interfaceIdeas && tool===null && !material.length) await pursueAgent(
      `${value}\n\n[Orientação do Estúdio de Interfaces: apresente exatamente três direções visuais numeradas, com nome, paleta, composição e sensação; recomende uma; não gere código nem altere arquivos.]`,
      false, 'conversation', recentAgentHistory());
    else if(tool===null && !material.length && ChatCore.isProjectFailureReport(value)) {
      if(localStorage.getItem('ia-local-zero-workspace')) await pursueAgent(value, false, 'auto', recentAgentHistory());
      else assistantReply('Para investigar esse erro no código, selecione a pasta do projeto como workspace e reenvie o relato. Sem a pasta, não consigo ler os arquivos nem executar os testes.', 'Workspace necessário');
    }
    else if(tool===null && !material.length && ChatCore.isProjectUnderstandingRequest(value)) await pursueAgent(value, false, 'analyze', recentAgentHistory());
    else if(tool==='conversation') {
      if(material.length) await pursueAgent(value || 'Analise os anexos recebidos.', false, 'auto', recentAgentHistory(), material);
      else await pursueAgent(value, false, 'auto', recentAgentHistory());
    }
    else if(shouldAutoPursueAgent(value, material)) await pursueAgent(value, false, 'auto', recentAgentHistory());
    else if(document.body.classList.contains('training-mode') && materialUrls.length>=2 && !material.length) await learnMaterials(materialUrls);
    else if(learningTopic && !compoundLearning && !material.length) await learnTopic(learningTopic,value);
    else if(tool==='search_web') await call(tool,{query:value.replace(/^\/(pesquisar|pesquisa)\s+/i,'').trim()});
    else if(tool==='research_web' && ChatCore.isResearchSynthesisRequest(value))
      await pursueAgent(value, false, 'research', recentAgentHistory());
    else if(tool==='research_web') {
      const save=/\s--salvar\b/i.test(value);
      const query=ChatCore.researchQuery(value);
      await call(tool,{query,max_results:2,save_to_corpus:save});
    }
    else if(tool==='inspect_project') {
      const identityOnly=ChatCore.isWorkspaceIdentityQuestion(value);
      if(identityOnly) await call(tool,{max_depth:1},true);
      else await pursueAgent(value, false, 'analyze', recentAgentHistory());
    }
    else if(tool==='open_page') await call(tool,{url:value.replace(/^\/abrir\s+/i,'').trim()});
    else if(tool==='list_files') await call(tool,{path:value.replace(/^\/arquivos\s*/i,'').trim()});
    else if(tool==='search_files') await call(tool,{query:value.replace(/^\/buscar\s+/i,'').trim()});
    else if(value === '/ajuda' || value === '/help') {
      assistantReply(
        "### Guia rápido de comandos e ferramentas locais:\n\n" +
        "- `/analisar`: Lê arquivos centrais do projeto ativo, resume o que fazem e aponta os caminhos usados como evidência.\n" +
        "- `/buscar <termo>`: Busca ocorrências de texto no código do projeto.\n" +
        "- `/arquivos [pasta]`: Lista arquivos do workspace.\n" +
        "- `/ler <arquivo>`: Lê e exibe o conteúdo do arquivo.\n" +
        "- `/pesquisa <termo>`: Pesquisa guiada na web com fontes e síntese.\n" +
        "- `/pesquisar <termo>`: Busca rápida na internet.\n" +
        "- `/aprenda <tema>`: Pesquisa e indexa conhecimento sobre um framework ou linguagem.\n" +
        "- `/testes`: Executa os testes automatizados do projeto.\n" +
        "- `/terminal <perfil>`: Confirma e executa um perfil permitido, como `git status --short` ou `pytest`.\n" +
        "- `/verificacoes`: Lista os verificadores disponíveis no projeto.\n" +
        "- `/fontes`: Exibe as páginas consultadas nesta sessão.\n\n" +
        "- `/rotear <tarefa>`: Resolve a skill e o próximo passo pelos contratos locais; mostra o plano sem executar a ferramenta.\n" +
        "- `/agente <objetivo>`: Executa o núcleo TypeScript em etapas, com aprovação inline e verificação.\n" +
        "- Pedidos como ‘quero criar’ ou ‘preciso integrar’ são encaminhados automaticamente ao AgentCore quando há um workspace selecionado.\n\n" +
        "**Dicas do Editor Workspace**:\n" +
        "- Pressione `Tab` no editor para identar com 2 espaços.\n" +
        "- Pressione `Ctrl+S` (ou `Cmd+S`) para salvar alterações pendentes.",
        "Guia de uso local"
      );
    }
    else if(value === '/analisar' || value === '/inspect') await pursueAgent(value, false, 'analyze');
    else if(value === '/testes' || value === '/test') await call('project_checks',{check:'auto'});
    else if(value === '/verificacoes') await call('project_checks',{check:'list'});
    else if(value.startsWith('/ler ')) await call('read_file',{path:value.slice(5).trim()});
    else if(value.startsWith('/criar pasta ')) await runChatTool('create_directory',{path:value.slice(13).trim()},'Criar pasta');
    else if(value.startsWith('/criar arquivo ')) { const parts=value.slice(15).split('\n'); await runChatTool('create_file',{path:parts.shift().trim(),content:parts.join('\n')},'Criar arquivo'); }
    else if(value.startsWith('/editar arquivo ')) { const parts=value.slice(16).split('\n---\n'), head=parts.shift().split('\n'); await runChatTool('edit_file',{path:head.shift().trim(),old_text:head.join('\n'),new_text:parts.join('\n---\n')},'Editar arquivo'); }
    else if(parseRepairCommand(value)) { const repair=parseRepairCommand(value); if(repair.tool==='apply_repair') await runChatTool(repair.tool,repair.arguments,'Aplicar correção'); else await call(repair.tool,repair.arguments); }
    else if(value==='/fontes') await call('list_sources',{});
    else if(material.length) await pursueAgent(value || 'Analise os anexos recebidos.', false, 'auto', recentAgentHistory(), material);
    else await pursueAgent(value, false, 'auto', recentAgentHistory());
  } catch(error) { notice(`Falha ao preparar a mensagem: ${error.message}`); }
  finally { sending=false; refreshControls(); saveConversation(); input.focus(); }
}
function welcomeTemplate() {
  // A tela inicial vive no index.html; reutiliza a marcação guardada antes do app.js rodar.
  if (window.BRASA_WELCOME) return window.BRASA_WELCOME;
  return `<div class="welcome" id="welcome">
    <div class="welcome-badge">ESTÚDIO BRASA · 01 / 04</div>
    <h1>Acenda uma ideia.<br><span>Construa o próximo passo.</span></h1>
    <p>Explore ideias, planeje interfaces, crie páginas com prévia e continue o trabalho no seu workspace.</p>
    <div class="welcome-chips">
      <button class="welcome-chip" onclick="quickAction('inspect_project')">
        <span class="chip-icon">⌘</span>
        <span class="chip-text"><strong>Analisar projeto ativo</strong><small>Ler arquivos centrais e mostrar evidências</small></span>
      </button>
      <button class="welcome-chip" onclick="quickAction('research_doc')">
        <span class="chip-icon">✦</span>
        <span class="chip-text"><strong>Pesquisar documentação</strong><small>Consultar fontes na web e acervo</small></span>
      </button>
      <button class="welcome-chip" onclick="quickAction('search_code')">
        <span class="chip-icon">⌕</span>
        <span class="chip-text"><strong>Buscar no código</strong><small>Encontrar funções e classes</small></span>
      </button>
      <button class="welcome-chip" onclick="quickAction('create_page')">
        <span class="chip-icon">🌐</span>
        <span class="chip-text"><strong>Explorar uma interface</strong><small>Compare três direções antes da prévia</small></span>
      </button>
    </div>
    <p class="welcome-note">Feito para criar com critério. Memória, artefatos e histórico ficam no seu computador.<br>Pesquisas externas consultam fontes e enriquecem o índice local.</p>
  </div>`;
}
function quickAction(action) {
  if (action === 'inspect_project') {
    if (sending || liveActions.size) return notice('Aguarde a operação atual terminar.');
    conversation.push({role:'user',content:'Analise meu projeto ativo',attachments:[]});
    addMessage('user','Analise meu projeto ativo');
    saveConversation();
    pursueAgent('Analise meu projeto ativo', false, 'analyze', recentAgentHistory());
  } else if (action === 'research_doc') {
    input.value = '/pesquisa ';
    input.focus();
  } else if (action === 'search_code') {
    selectTool('search_files');
    input.placeholder = 'Digite o termo para buscar no código...';
    input.focus();
  } else if (action === 'create_page') {
    interfaceStudioDraft = true;
    pendingInterfacePreview=null;
    interfaceEditPathDraft=null;
    input.value = 'Quero explorar possibilidades para uma interface: ';
    input.focus();
  }
}
function newChat() {
  if(sending || liveActions.size) return notice('Aguarde a operação atual terminar.');
  saveConversation(); currentConversationId=null; agentRunRef=null; agentTaskRef=null; pendingAgent=null; interfaceStudioDraft=false; pendingInterfacePreview=null; interfaceEditPathDraft=null; conversation=[]; timeline=[]; selectedTool=null; clearAttachments(); closeTool();
  chat.innerHTML=welcomeTemplate();
  renderHistory(); input.focus();
}
function quick(prefix) { input.value=prefix; input.focus(); }
function listSources() { openTool('sources'); }
function refreshProjectDock() { return renderDockDirectory(dockDirectoryPath); }
input.addEventListener('keydown',event=> { if(event.key==='Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); send(); } });
input.addEventListener('input',()=> { input.style.height='auto'; input.style.height=Math.min(input.scrollHeight,140)+'px'; });
document.addEventListener('click', event => { if (!event.target.closest('.project-selector-wrap')) closeProjectMenu(); });
document.getElementById('project-name-input')?.addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); createHeaderProject(); } });
renderHistory(); refreshControls();
restoreLastConversation();
restoreWorkspaceExplorer();
restoreWorkspace();
async function restoreWorkspace() {
  const saved=localStorage.getItem('ia-local-zero-workspace');
  if(!saved) {
    try {
      const catalog=await fetch('/api/projects',{cache:'no-store'}).then(response=>response.json());
      if(catalog.ok && catalog.current) { setWorkspaceIndicator(catalog.current); await loadDockRoot(); }
    } catch (_) { /* O seletor continua disponível mesmo sem sincronização inicial. */ }
    return;
  }
  const selected=await toolRequest('set_workspace',{path:saved},false);
  if(!selected.ok) {
    localStorage.removeItem('ia-local-zero-workspace');
    setWorkspaceIndicator(null);
    return;
  }
  setWorkspaceIndicator(selected.data.workspace);
  await loadDockRoot();
}
document.getElementById('workspace-entry-dialog')?.addEventListener('cancel',event=>{
  if(workspaceEntryDialogState?.pending) event.preventDefault(); else workspaceEntryDialogState=null;
});
setMode(localStorage.getItem('ia-local-zero-mode') || 'chat');
