import { COMMUNITY_GROUPS as GROUPS } from './research-taxonomy.js';
import { createTaxonEditor } from './research-taxon-editor.js';
const SVG = 'http://www.w3.org/2000/svg';
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const groupName = value => GROUPS.find(group => group[0] === value)?.[1] ?? 'Unknown / just a mushroom';
const validGroup = value => GROUPS.some(group => group[0] === value) ? value : 'unknown';

/** All review coordinates refer to native image pixels, regardless of zoom. */
export function normalizeBox(box, width, height, minimum = 2) {
  if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite) || !Number.isFinite(width) || !Number.isFinite(height) || width < minimum || height < minimum) return null;
  const x1 = clamp(Math.round(Math.min(box[0], box[2])), 0, width);
  const y1 = clamp(Math.round(Math.min(box[1], box[3])), 0, height);
  const x2 = clamp(Math.round(Math.max(box[0], box[2])), 0, width);
  const y2 = clamp(Math.round(Math.max(box[1], box[3])), 0, height);
  return x2 - x1 >= minimum && y2 - y1 >= minimum ? [x1, y1, x2, y2] : null;
}

export function pointInImage(clientX, clientY, rect, width, height) {
  if (![clientX, clientY, rect.left, rect.top, rect.width, rect.height, width, height].every(Number.isFinite) || rect.width <= 0 || rect.height <= 0) return null;
  return [clamp((clientX - rect.left) * width / rect.width, 0, width), clamp((clientY - rect.top) * height / rect.height, 0, height)];
}

export function translateBox(box, dx, dy, width, height) {
  const x = clamp(Math.round(dx), -box[0], width - box[2]);
  const y = clamp(Math.round(dy), -box[1], height - box[3]);
  return [box[0] + x, box[1] + y, box[2] + x, box[3] + y];
}

function timeLabel(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return 'Time unavailable';
  const whole = Math.floor(seconds), hours = Math.floor(whole / 3600), minutes = Math.floor(whole / 60) % 60;
  return `${hours ? `${hours}:` : ''}${String(minutes).padStart(2, '0')}:${String(whole % 60).padStart(2, '0')}${seconds % 1 ? `.${Math.floor(seconds % 1 * 10)}` : ''}`;
}

export function priorityFrameIndex(frames,currentId=null,reviewedIds=new Set()) {
  return frames.map((item,index)=>({item,index})).filter(({item})=>item.id!==currentId&&Number.isInteger(item.reviewPriority)&&item.reviewPriority>0)
    .sort((a,b)=>Number(reviewedIds.has(a.item.id))-Number(reviewedIds.has(b.item.id))||b.item.reviewPriority-a.item.reviewPriority||a.index-b.index)[0]?.index??-1;
}

if (typeof document !== 'undefined') {
  const $ = id => document.getElementById(id);
  const state = { collection: null, frames: [], index: -1, source: '', csrf: null, drafts: new Map(), selectedBox: null,
    imageReady: false, reviewReady: false, loading: false, saving: false, sequence: 0, abort: null, retry: null,
    zoom: 1, zoomMode: 'fit', tool: 'draw', gesture: null, aiRefresh: false, terms:null };
  const frame = () => state.frames[state.index] ?? null;
  const draft = () => frame() ? state.drafts.get(frame().id) : null;
  const editable = () => !state.loading && !state.saving && state.imageReady && state.reviewReady && Boolean(draft());
  const canDraw = () => editable() && draft().status === 'visible';
  const sourceFor = item => state.collection?.sources.find(source => source.id === item?.sourceId);
  const create = (tag, className, text) => { const element = document.createElement(tag); if (className) element.className = className; if (text != null) element.textContent = text; return element; };
  const svg = (tag, attributes) => { const element = document.createElementNS(SVG, tag); for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value)); return element; };
  const announce = message => { $('page-status').textContent = message; };
  const safeUrl = value => { try { const url = new URL(value, document.baseURI); return ['http:', 'https:'].includes(url.protocol) ? url.href : null; } catch { return null; } };

  async function api(path, { method = 'GET', body, signal } = {}) {
    const headers = { Accept: 'application/json' };
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    if (method !== 'GET' && state.csrf) headers['X-CSRF-Token'] = state.csrf;
    let response;
    try { response = await fetch(new URL(`api/${path}`, document.baseURI), { method, headers, body: body === undefined ? undefined : JSON.stringify(body), credentials: 'same-origin', cache: 'no-store', signal }); }
    catch (error) { if (error.name === 'AbortError') throw error; throw new Error('The connection was interrupted. Your draft is still here; try again.'); }
    let result;
    try { result = await response.json(); } catch { throw new Error('The lab returned an unreadable response. Please try again.'); }
    if (!response.ok) { const error = new Error(result.error?.message ?? result.error ?? result.message ?? `Request failed (${response.status}).`); error.status = response.status; throw error; }
    return result;
  }

  function reviewError(message = '', retry = null, retryLabel = 'Try again') {
    $('review-error').hidden = !message; $('review-error-text').textContent = message;
    $('review-retry').hidden = !retry; $('review-retry').textContent = retryLabel; state.retry = retry;
  }

  function markDirty(message = 'Unsaved draft. You can move between frames; this draft stays here until you leave the page.') {
    if (!draft()) return; draft().dirty = true; $('save-status').textContent = message;
    $('review-state').textContent = 'Draft · not submitted'; reviewError(); updateControls();
  }

  function renderCounts(summary = state.collection?.summary) {
    if (!summary) return;
    for (const [id, key] of [['frame-count', 'frames'], ['video-count', 'videos'], ['review-count', 'reviews']]) {
      $(id).textContent = Number.isFinite(summary[key]) ? summary[key].toLocaleString() : '—';
    }
    const run = state.collection?.aiRun;
    const prepared = Number.isFinite(run?.reviewedFrames) ? run.reviewedFrames : Number.isFinite(run?.processedFrames) ? run.processedFrames : 0;
    const total = Number.isFinite(run?.totalFrames) ? run.totalFrames : summary.frames;
    $('ai-run-status').textContent = `AI suggestions prepared for ${prepared.toLocaleString()} of ${Number.isFinite(total) ? total.toLocaleString() : '—'} frames${run?.status ? ` · ${String(run.status).replaceAll('-', ' ')}` : ''}. Processing progress, not confirmed mushroom labels.`;
    $('ai-run-status').title = run?.updatedAt ? `Last updated: ${run.updatedAt}` : '';
  }

  function setTerms(terms) {
    state.terms=terms&&typeof terms.version==='string'&&/^[a-f0-9]{64}$/.test(terms.noticeSha256)&&typeof terms.notice==='string'?terms:null;
    if(state.terms)$('participation-notice').textContent=state.terms.notice;
  }

  const workerLabels={disabled:'Training is disabled',paused:'Training is paused',waiting_data:'Waiting for more approved data',waiting_daily_limit:'Waiting for the next training window',waiting_resource:'Waiting for the training computer',waiting_sync:'Waiting for current review permissions',exporting:'Preparing an eligible dataset',training:'Training a candidate',comparing:'Comparing candidate results',retry_wait:'Waiting before a retry',candidate_ready:'Candidate results are ready',failed:'Latest attempt needs attention',offline:'Training computer offline or report unavailable'};
  async function refreshLearning() {
    try {
      const data=await api('training-status');
      $('learning-worker').textContent=workerLabels[data.worker?.state]??workerLabels.offline;
      for(const [id,key] of [['learning-approved','approvedFrames'],['learning-new','newFrames'],['learning-sessions','newSessions']])$(id).textContent=Number.isSafeInteger(data[key])?data[key].toLocaleString():'—';
      $('learning-thresholds').textContent=data.thresholds?`A new run needs at least ${data.thresholds.newFrames} new approved frames across ${data.thresholds.newSessions} new source sessions. At most ${data.thresholds.maxRunsPerDay} run${data.thresholds.maxRunsPerDay===1?'':'s'} per day.`:'Only reviewed, eligible labels can enter training.';
      const candidate=data.latestCandidate,labels={candidate_better:'The latest candidate passed the comparison gate.',candidate_not_better:'The latest candidate did not improve enough to pass.',insufficient_evidence:'The latest candidate has insufficient evaluation evidence.',comparison_failed:'The latest candidate comparison did not finish.'};
      let result=candidate?labels[candidate.status]:'No candidate result has been reported.';
      if(candidate?.comparison){const c=candidate.comparison,pct=v=>Number.isFinite(v)?`${Math.round(v*1000)/10}%`:'not measured',rate=v=>Number.isFinite(v)?`${Math.round(v*100)/100}`:'not measured';result+=` Recall: existing ${pct(c.baselineRecall)}, candidate ${pct(c.candidateRecall)}. Extra detections per evaluated frame: existing ${rate(c.baselineFalsePositivesPerFrame)}, candidate ${rate(c.candidateFalsePositivesPerFrame)}. These are saved test results, not field accuracy.`;}
      $('learning-candidate').textContent=result;
      $('learning-updated').textContent=`${data.updatedAt?`${data.availability==='stale'?'Last report is stale':'Report updated'}: ${new Date(data.updatedAt).toLocaleString()}.`:'No current report from the training computer.'} Candidates remain inactive until separately approved.`;
    }catch{$('learning-worker').textContent=workerLabels.offline;$('learning-updated').textContent='Progress could not be refreshed. Candidates are never activated automatically.';}
  }

  function updateControls() {
    const current = frame(), currentDraft = draft(), usable = editable(), drawAllowed = canDraw();
    $('review-workspace').setAttribute('aria-busy', String(state.loading || state.saving));
    $('visibility-fieldset').disabled = !usable; $('submission-options').disabled = !usable;
    $('draw-tool').disabled = !drawAllowed; $('pan-tool').disabled = !state.imageReady || state.saving;
    for (const id of ['fit-image', 'original-image', 'zoom-in', 'zoom-out']) $(id).disabled = !state.imageReady || state.saving;
    $('add-box').disabled = !drawAllowed || (currentDraft?.boxes.length ?? 0) >= 64;
    $('show-ai').disabled = !usable || !currentDraft?.status;
    $('refresh-ai').disabled = !usable || state.aiRefresh;
    const canSave = usable && Boolean(currentDraft?.status) && Boolean(state.terms);
    $('save-next').disabled = !canSave; $('save-stay').disabled = !canSave;
    $('source-filter').disabled = !state.collection || state.saving;
    $('previous-frame').disabled = state.index <= 0 || state.saving;
    $('next-frame').disabled = state.index < 0 || state.index >= state.frames.length - 1 || state.saving;
    $('next-priority').disabled=state.saving||!state.collection||priorityFrameIndex(state.frames,current?.id)<0;
    $('withdraw-permission').hidden=!currentDraft?.revision||currentDraft.trainingConsent!==true;
    $('withdraw-permission').disabled=!usable||!currentDraft?.revision||currentDraft.trainingConsent!==true;
    $('frame-progress').textContent = current ? `Frame ${state.index + 1} of ${state.frames.length}` : 'No frames in this collection';
    $('draw-tool').classList.toggle('active', state.tool === 'draw'); $('draw-tool').setAttribute('aria-pressed', String(state.tool === 'draw'));
    $('pan-tool').classList.toggle('active', state.tool === 'pan'); $('pan-tool').setAttribute('aria-pressed', String(state.tool === 'pan'));
    $('image-stage').classList.toggle('panning', state.tool === 'pan' || !drawAllowed);
    $('image-viewport').classList.toggle('pan-mode', state.tool === 'pan');
    $('box-help').textContent = currentDraft?.status === 'absent' ? 'No boxes needed. If you spot one, change your observation to “Yes”.' : currentDraft?.status === 'uncertain' ? 'An uncertain review has no confirmed boxes. Choose “Yes” to mark a mushroom you can see, even if its group is unknown.' : 'Choose “Yes” to draw around each mushroom. Unknown is a useful group; you can refine it later.';
    $('review-complete').disabled = !usable || currentDraft?.status === 'uncertain';
    for (const element of $('box-list').querySelectorAll('button,select,input,textarea')) element.disabled = !usable || element.dataset.emptyTaxon === 'true';
    for (const button of $('source-timeline').querySelectorAll('button')) button.disabled = state.saving;
    for (const button of $('ai-list').querySelectorAll('button')) button.disabled = !drawAllowed || button.dataset.accepted === 'true';
  }

  function renderSource() {
    const current = frame(), source = sourceFor(current);
    $('source-title').textContent = source?.title ?? 'Video source'; $('frame-time').textContent = timeLabel(current?.timestampSeconds);
    $('native-size').textContent = current ? `${current.width} × ${current.height} px` : 'Original pixels';
    $('frame-image').alt = `${source?.title ?? 'Mushroom video'} at ${timeLabel(current?.timestampSeconds)}. Review this image for visible mushrooms.`;
    $('source-attribution').textContent = [source?.attribution, source?.license ? `License: ${source.license}.` : '', source?.changes, 'Frame extracted from video and displayed as a JPEG. Review labels are separate from the source image license.'].filter(Boolean).join(' ');
    $('source-links').replaceChildren();
    for (const [label, url] of [['View original source ↗', source?.sourceUrl], ['Read license ↗', source?.licenseUrl]]) {
      const href = url && safeUrl(url); if (!href) continue;
      const link = create('a', '', label); link.href = href; link.target = '_blank'; link.rel = 'noopener noreferrer'; $('source-links').append(link);
    }
    renderTimeline();
  }

  function renderTimeline() {
    const current = frame(), container = $('source-timeline'); container.replaceChildren(); if (!current) return;
    const siblings = state.collection.frames.filter(item => item.sourceId === current.sourceId).sort((a, b) => a.timestampSeconds - b.timestampSeconds);
    const selected = siblings.findIndex(item => item.id === current.id), start = Math.max(0, Math.min(selected - 9, siblings.length - 21)), end = Math.min(start + 21, siblings.length);
    if (start > 0) container.append(create('span', 'timeline-ellipsis', '…'));
    for (const item of siblings.slice(start, end)) {
      const button = create('button', 'timeline-frame', timeLabel(item.timestampSeconds)); button.type = 'button';
      button.setAttribute('aria-current', String(item.id === current.id)); button.setAttribute('aria-label', `Open frame at ${timeLabel(item.timestampSeconds)} from ${sourceFor(item)?.title ?? 'this video'}`);
      button.classList.toggle('reviewed', Boolean(state.drafts.get(item.id)?.revision));
      button.addEventListener('click', () => { if (!state.saving) openById(item.id); }); container.append(button);
    }
    if (end < siblings.length) container.append(create('span', 'timeline-ellipsis', '…'));
  }

  function renderDraft() {
    const value = draft(); if (!value) return;
    for (const input of document.querySelectorAll('input[name="visibility"]')) input.checked = input.value === value.status;
    $('review-complete').checked = value.complete;
    $('permission-state').textContent=value.revision?(value.trainingConsent?'Your saved revision permits training use. You can withdraw that permission below.':'Training permission for your earlier review is withheld. Submitting a new revision accepts the terms above for that new revision.'):'Submitting accepts these terms for your annotations. Your observation still awaits review.';
    $('review-state').textContent = value.dirty ? 'Draft · not submitted' : value.revision ? 'Your saved review' : 'Your observation';
    $('save-status').textContent = value.dirty ? 'Unsaved draft restored for this frame.' : value.revision ? `Your review is saved (revision ${value.revision}). It awaits curator review.` : 'Nothing submitted yet.';
    renderBoxes(); renderAI(); updateControls();
  }

  function renderBoxes() {
    const value = draft(); $('box-list').replaceChildren(); $('box-count').textContent = value?.boxes.length ?? 0;
    $('boxes-empty').hidden = Boolean(value?.boxes.length);
    if (!value) return;
    value.boxes.forEach((box, index) => {
      const card = create('div', `box-editor${state.selectedBox === index ? ' selected' : ''}`), heading = create('div', 'box-editor-heading');
      const choose = create('button', 'select-box'); choose.type = 'button'; choose.append(create('span', 'box-number', index + 1), document.createTextNode(`Mushroom ${index + 1}`));
      choose.setAttribute('aria-pressed', String(state.selectedBox === index)); choose.addEventListener('click', () => { state.selectedBox = index; renderBoxes(); renderOverlay(); $('image-viewport').focus({ preventScroll: true }); });
      const remove = create('button', 'delete-box', 'Remove'); remove.type = 'button'; remove.setAttribute('aria-label', `Remove mushroom box ${index + 1}`); remove.addEventListener('click', () => removeBox(index));
      heading.append(choose, remove); card.append(heading);
      const label = create('label', 'group-label', 'Broad group, if you have a guess'); label.htmlFor = `box-group-${index}`;
      const select = create('select'); select.id = label.htmlFor; select.setAttribute('aria-label', `Broad mushroom group for box ${index + 1}`);
      for (const [id, name] of GROUPS) { const option = create('option', '', name); option.value = id; select.append(option); } select.value = box.group;
      select.addEventListener('change', () => { box.group = select.value; delete box.taxonCandidate; markDirty(); renderBoxes(); }); card.append(label, select);
      card.append(createTaxonEditor({candidate:box.taxonCandidate,groupId:box.group,label:`Identification for mushroom ${index+1}`,onChange:candidate=>{if(candidate)box.taxonCandidate=candidate;else delete box.taxonCandidate;markDirty();}}));
      const details = create('details', 'coordinate-editor'), summary = create('summary', '', 'Adjust original pixel coordinates'), fields = create('div', 'coordinate-fields');
      for (const [key, title, initial] of [['x', 'Left (x)', box.bbox[0]], ['y', 'Top (y)', box.bbox[1]], ['width', 'Width', box.bbox[2] - box.bbox[0]], ['height', 'Height', box.bbox[3] - box.bbox[1]]]) {
        const fieldLabel = create('label', '', title), input = create('input'); input.type = 'number'; input.step = '1'; input.min = key === 'width' || key === 'height' ? '2' : '0'; input.max = String(key === 'x' || key === 'width' ? frame().width : frame().height); input.value = initial; input.dataset.coordinate = key; input.setAttribute('aria-label', `${title} for box ${index + 1}`);
        input.addEventListener('change', () => {
          const values = Object.fromEntries([...fields.querySelectorAll('input')].map(item => [item.dataset.coordinate, Number(item.value)]));
          const candidate = [values.x, values.y, values.x + values.width, values.y + values.height];
          if (Object.values(values).some(number => !Number.isFinite(number)) || values.x < 0 || values.y < 0 || values.width < 2 || values.height < 2 || candidate[2] > frame().width || candidate[3] > frame().height) { reviewError('Keep the box inside the image, with width and height of at least 2 pixels.'); input.value = key === 'x' ? box.bbox[0] : key === 'y' ? box.bbox[1] : key === 'width' ? box.bbox[2] - box.bbox[0] : box.bbox[3] - box.bbox[1]; return; }
          box.bbox = normalizeBox(candidate, frame().width, frame().height); state.selectedBox = index; markDirty(); renderOverlay();
        });
        fieldLabel.append(input); fields.append(fieldLabel);
      }
      details.append(summary, fields); card.append(details); $('box-list').append(card);
    });
    updateControls();
  }

  function proposals() {
    const current = frame(); if (!current || !Array.isArray(current.ai?.boxes)) return [];
    return current.ai.boxes.slice(0, 100).map((proposal, index) => ({ ...proposal, key: `${current.id}:${index}:${proposal.bbox?.join(',')}`, bbox: normalizeBox(proposal.bbox, current.width, current.height) })).filter(proposal => proposal.bbox);
  }

  function renderAI() {
    const value = draft(), shown = Boolean(value?.showAI); $('ai-results').hidden = !shown; $('ai-assisted').hidden = !value?.aiAssisted;
    $('show-ai').textContent = shown ? 'Hide AI suggestions' : 'Show AI suggestions'; $('ai-list').replaceChildren();
    if (shown) {
      const found = proposals(), info = frame()?.ai, model = typeof info?.model === 'string' ? info.model : info?.model?.name;
      const status = String(info?.status ?? 'not run');
      $('ai-summary').textContent = found.length ? `${found.length} suggestion${found.length === 1 ? '' : 's'}${model ? ` from ${model}` : ''}. Blue dashed boxes are unaccepted AI guesses. Accept only what you can see.` : `No suggestions available for this frame. AI status: ${status}. No suggestion does not mean no mushroom.`;
      for (const [index, proposal] of found.entries()) {
        const row = create('div', 'ai-proposal'), description = create('div', '', `Suggestion ${index + 1}`);
        const score = Number.isFinite(proposal.score) ? ` · ${Math.round(proposal.score * 100)}% model score` : '';
        description.append(create('span', '', `${String(proposal.label ?? 'Possible mushroom')}${score}`));
        const accepted = value.accepted.includes(proposal.key), button = create('button', '', accepted ? 'Added to draft' : 'Use this box'); button.type = 'button'; button.dataset.accepted = String(accepted);
        button.addEventListener('click', () => {
          if (!canDraw()) { reviewError('Choose “Yes” before adding a suggested box, and only if you can see the mushroom yourself.'); return; }
          if (value.boxes.length >= 64) { reviewError('This frame already has the maximum 64 boxes.'); return; }
          value.boxes.push({ bbox: [...proposal.bbox], group: validGroup(proposal.label) }); value.accepted.push(proposal.key); value.aiAssisted = true;
          state.selectedBox = value.boxes.length - 1; markDirty('AI box added to your draft. Adjust its edges and group before saving.'); renderBoxes(); renderAI(); renderOverlay();
        }); row.append(description, button); $('ai-list').append(row);
      }
    }
    updateControls(); renderOverlay();
  }

  function drawRect(box, index, { proposal = false, preview = false } = {}) {
    const overlay = $('box-overlay'), [x1, y1, x2, y2] = box;
    const rect = svg('rect', { x: x1, y: y1, width: x2 - x1, height: y2 - y1, class: `box${proposal ? ' proposal' : ''}${state.selectedBox === index && !proposal ? ' selected' : ''}` });
    if (!proposal) rect.dataset.box = index; overlay.append(rect);
    if (proposal || preview) return;
    const tagWidth = 23 / state.zoom, tagHeight = 20 / state.zoom, fontSize = 11 / state.zoom;
    const tagY = Math.max(0, y1 - tagHeight);
    overlay.append(svg('rect', { x: x1, y: tagY, width: tagWidth, height: tagHeight, class: 'box-tag-bg' }));
    const text = svg('text', { x: x1 + 6 / state.zoom, y: tagY + 14 / state.zoom, 'font-size': fontSize, class: 'box-tag' }); text.textContent = String(index + 1); overlay.append(text);
    if (state.selectedBox === index) for (const [corner, x, y] of [['nw', x1, y1], ['ne', x2, y1], ['se', x2, y2], ['sw', x1, y2]]) {
      const size = 11 / state.zoom, handle = svg('rect', { x: x - size / 2, y: y - size / 2, width: size, height: size, class: 'handle' }); handle.dataset.box = index; handle.dataset.corner = corner; overlay.append(handle);
    }
  }

  function renderOverlay() {
    $('box-overlay').replaceChildren(); const current = frame(), value = draft(); if (!current || !value || !state.imageReady) return;
    $('box-overlay').setAttribute('viewBox', `0 0 ${current.width} ${current.height}`);
    if (value.showAI) for (const proposal of proposals()) if (!value.accepted.includes(proposal.key)) drawRect(proposal.bbox, -1, { proposal: true });
    value.boxes.forEach((box, index) => drawRect(box.bbox, index));
    if (state.gesture?.type === 'draw' && state.gesture.preview) drawRect(state.gesture.preview, -1, { preview: true });
  }

  function setZoom(mode, scale = state.zoom) {
    if (!state.imageReady || !frame()) return;
    const viewport = $('image-viewport'), current = frame(), old = state.zoom;
    const center = [(viewport.scrollLeft + viewport.clientWidth / 2) / old, (viewport.scrollTop + viewport.clientHeight / 2) / old];
    state.zoomMode = mode; state.zoom = mode === 'fit' ? Math.min(viewport.clientWidth / current.width, viewport.clientHeight / current.height) : clamp(scale, .05, 8);
    $('image-stage').style.width = `${current.width * state.zoom}px`; $('image-stage').style.height = `${current.height * state.zoom}px`;
    $('zoom-label').value = `${Math.round(state.zoom * 100)}%`; $('zoom-label').textContent = `${Math.round(state.zoom * 100)}%`;
    if (mode === 'fit') { viewport.scrollLeft = 0; viewport.scrollTop = 0; } else { viewport.scrollLeft = center[0] * state.zoom - viewport.clientWidth / 2; viewport.scrollTop = center[1] * state.zoom - viewport.clientHeight / 2; }
    renderOverlay();
  }

  function removeBox(index) {
    if (!editable()) return; draft().boxes.splice(index, 1); state.selectedBox = null; markDirty(); renderBoxes(); renderOverlay(); announce('Box removed from your draft.');
  }

  function startImage(current, token) {
    state.imageReady = false; $('image-stage').hidden = true; $('image-message').hidden = false; $('image-retry').hidden = true;
    $('image-message').querySelector('.loading-ring').hidden = false; $('image-message-text').textContent = 'Loading the original image…'; updateControls();
    const image = new Image(), url = safeUrl(current.imageUrl);
    const fail = message => { if (token !== state.sequence) return; $('image-message-text').textContent = message; $('image-message').querySelector('.loading-ring').hidden = true; $('image-retry').hidden = false; state.imageReady = false; updateControls(); };
    image.onload = () => {
      if (token !== state.sequence) return;
      if (image.naturalWidth !== current.width || image.naturalHeight !== current.height) { fail('This image’s size does not match its frame record. Please reload; drawing is paused to keep coordinates accurate.'); return; }
      $('frame-image').src = url; $('image-stage').hidden = false; $('image-message').hidden = true; state.imageReady = true; setZoom('fit'); updateControls(); renderOverlay();
    };
    image.onerror = () => fail('The original image could not be loaded. Your review is still here.');
    if (url) image.src = url; else fail('This frame has no usable image address.');
  }

  async function openFrame(index, { reloadReview = false } = {}) {
    if (state.saving || index < 0 || index >= state.frames.length) return;
    if (state.gesture) cancelGesture();
    state.abort?.abort(); state.abort = new AbortController(); const token = ++state.sequence;
    state.index = index; state.selectedBox = null; state.loading = true; state.reviewReady = false; reviewError();
    const current = frame(); renderSource(); startImage(current, token);
    if (!state.drafts.has(current.id) || !state.drafts.get(current.id).loaded || reloadReview) {
      const retained = state.drafts.get(current.id);
      if (!retained || !reloadReview) state.drafts.set(current.id, { status: null, boxes: [], complete: false, trainingConsent: false, aiAssisted: false, revision: 0, dirty: false, loaded: false, showAI: false, accepted: [] });
      renderDraft();
      try {
        const result = await api(`reviews/${encodeURIComponent(current.id)}`, { signal: state.abort.signal }); if (token !== state.sequence) return;
        const review = result.review;
        state.drafts.set(current.id, { status: ['visible', 'absent', 'uncertain'].includes(review?.status) ? review.status : null,
          boxes: (review?.boxes ?? []).map(box => ({ bbox: normalizeBox(box.bbox, current.width, current.height), group: validGroup(box.group),...(box.taxonCandidate?{taxonCandidate:structuredClone(box.taxonCandidate)}:{}) })).filter(box => box.bbox),
          complete: review?.complete === true, trainingConsent: review?.trainingConsent === true, aiAssisted: review?.aiAssisted === true,
          revision: Number.isInteger(review?.revision) ? review.revision : 0, dirty: false, loaded: true, showAI: false, accepted: [] });
        state.reviewReady = true;
      } catch (error) {
        if (token !== state.sequence || error.name === 'AbortError') return;
        reviewError(`Your previous review could not be loaded. ${error.message}`, () => openFrame(state.index, { reloadReview: true }), 'Retry loading review');
      }
    } else state.reviewReady = state.drafts.get(current.id).loaded;
    if (token !== state.sequence) return;
    state.loading = false; renderDraft(); announce(`Frame ${index + 1} of ${state.frames.length}, ${sourceFor(current)?.title ?? 'video'}, ${timeLabel(current.timestampSeconds)}.`);
  }

  function openById(id) {
    const index = state.frames.findIndex(item => item.id === id); if (index >= 0) openFrame(index);
  }

  async function saveReview(next) {
    if (!editable()) return; const current = frame(), value = draft();
    if (!value.status) { reviewError('Choose whether mushrooms are visible before saving.'); return; }
    if (value.status === 'visible' && !value.boxes.length) { reviewError('Mark at least one mushroom with a box, or choose “Not sure”.'); $('image-viewport').focus(); return; }
    if (value.status !== 'visible' && value.boxes.length) { reviewError('Remove the boxes before saving “No mushrooms visible” or “Not sure”, or keep “Yes” if you can see mushrooms.'); return; }
    state.saving = true; reviewError(); $('save-status').textContent = 'Saving your observation…'; updateControls();
    try {
      const result = await api(`reviews/${encodeURIComponent(current.id)}`, { method: 'POST', body: { status: value.status, boxes: value.boxes.map(box => ({ bbox: [...box.bbox], group: box.group,...(box.taxonCandidate?{taxonCandidate:box.taxonCandidate}:{}) })), complete: value.complete, participationTerms:{version:state.terms.version,noticeSha256:state.terms.noticeSha256}, aiAssisted: value.aiAssisted, baseRevision: value.revision } });
      value.revision = result.review.revision; value.trainingConsent=result.review.trainingConsent; value.dirty = false; value.loaded = true;
      if (result.summary) { state.collection.summary = result.summary; renderCounts(); }
      $('save-status').textContent = 'Saved. Thank you—your observation is ready for curator review.'; $('review-state').textContent = 'Your saved review'; announce('Review saved for curator approval.'); renderTimeline();
      state.saving = false; renderDraft(); updateControls();
      if (next && state.index < state.frames.length - 1) await openFrame(state.index + 1);
      else if (next) $('save-status').textContent = 'Saved. You reached the last frame in this view. Choose another video or revisit earlier frames.';
    } catch (error) {
      state.saving = false; $('save-status').textContent = 'Your draft has not been saved. It is still on this page.'; updateControls();
      if (error.status === 409) reviewError('This review changed in another tab. Reload your saved review before editing again. Your current draft will be replaced only when you choose Reload.', () => openFrame(state.index, { reloadReview: true }), 'Reload saved review');
      else if(error.status===428)reviewError(error.message,async()=>{try{const fresh=await api('collection');setTerms(fresh.participationTerms);updateControls();reviewError('Terms refreshed. Read them beside Submit, then submit your draft again.');}catch(retryError){reviewError(retryError.message);}},'Refresh participation terms');
      else if (error.status === 403 || error.status === 401) reviewError('Your session needs to reconnect. Your draft is still here.', async () => { try { const session = await api('session', { method: 'POST', body: {} }); state.csrf = session.csrfToken; setTerms(session.participationTerms); updateControls();reviewError('Session reconnected. Try submitting your draft again.'); } catch (retryError) { reviewError(retryError.message); } }, 'Reconnect session');
      else reviewError(error.message, () => saveReview(next), 'Retry save');
    }
  }

  async function withdrawPermission() {
    if(!editable()||!draft()?.revision||draft().trainingConsent!==true)return;
    const value=draft();state.saving=true;reviewError();updateControls();
    try{const result=await api(`reviews/${encodeURIComponent(frame().id)}/withdraw`,{method:'POST',body:{baseRevision:value.revision}});
      value.revision=result.review.revision;value.trainingConsent=false;state.saving=false;renderDraft();
      $('save-status').textContent='Permission withdrawn for future training and evaluation. Earlier history is preserved; completed training cannot be undone.';announce('Training permission withdrawn.');
    }catch(error){state.saving=false;updateControls();reviewError(error.message,()=>openFrame(state.index,{reloadReview:true}),'Reload saved review');}
  }

  async function refreshAI() {
    if (!editable() || state.aiRefresh) return; state.aiRefresh = true; updateControls(); const token = state.sequence;
    try {
      const fresh = await api('collection'), byId = new Map(fresh.frames.map(item => [item.id, item]));
      for (const item of state.collection.frames) if (byId.has(item.id)) item.ai = byId.get(item.id).ai;
      state.collection.aiRun = fresh.aiRun; state.collection.summary = fresh.summary; renderCounts();
      if (token === state.sequence) { renderAI(); announce('AI suggestions refreshed. Your own boxes have not changed.'); }
    } catch (error) { if (token === state.sequence) reviewError(error.message, refreshAI, 'Retry suggestions'); }
    finally { state.aiRefresh = false; updateControls(); }
  }

  function cancelGesture() {
    if (!state.gesture) return;
    if (state.gesture.type === 'move' || state.gesture.type === 'resize') draft().boxes[state.gesture.index].bbox = state.gesture.original;
    state.gesture = null; renderOverlay();
  }

  const overlay = $('box-overlay');
  overlay.addEventListener('pointerdown', event => {
    if (!canDraw() || state.tool !== 'draw' || event.button !== 0) return; event.preventDefault();
    const point = pointInImage(event.clientX, event.clientY, overlay.getBoundingClientRect(), frame().width, frame().height); if (!point) return;
    const index = event.target.dataset.box === undefined ? null : Number(event.target.dataset.box);
    if (index != null && draft().boxes[index]) {
      state.selectedBox = index; state.gesture = { type: event.target.dataset.corner ? 'resize' : 'move', index, corner: event.target.dataset.corner, start: point, original: [...draft().boxes[index].bbox] };
    } else {
      if (draft().boxes.length >= 64) { reviewError('This frame already has the maximum 64 boxes.'); return; }
      state.selectedBox = null; state.gesture = { type: 'draw', start: point, preview: null };
    }
    state.gesture.pointerId = event.pointerId; overlay.setPointerCapture(event.pointerId); $('image-viewport').focus({ preventScroll: true }); renderOverlay();
  });
  overlay.addEventListener('pointermove', event => {
    const gesture = state.gesture; if (!gesture || gesture.pointerId !== event.pointerId || !frame()) return;
    const point = pointInImage(event.clientX, event.clientY, overlay.getBoundingClientRect(), frame().width, frame().height); if (!point) return;
    if (gesture.type === 'draw') gesture.preview = normalizeBox([...gesture.start, ...point], frame().width, frame().height);
    else if (gesture.type === 'move') draft().boxes[gesture.index].bbox = translateBox(gesture.original, point[0] - gesture.start[0], point[1] - gesture.start[1], frame().width, frame().height);
    else {
      const candidate = [...gesture.original]; candidate[gesture.corner.includes('w') ? 0 : 2] = point[0]; candidate[gesture.corner.includes('n') ? 1 : 3] = point[1];
      const normalized = normalizeBox(candidate, frame().width, frame().height); if (normalized) draft().boxes[gesture.index].bbox = normalized;
    }
    renderOverlay();
  });
  overlay.addEventListener('pointerup', event => {
    const gesture = state.gesture; if (!gesture || gesture.pointerId !== event.pointerId) return;
    if (gesture.type === 'draw' && gesture.preview) { draft().boxes.push({ bbox: gesture.preview, group: 'unknown' }); state.selectedBox = draft().boxes.length - 1; }
    if (gesture.type !== 'draw' || gesture.preview) markDirty();
    state.gesture = null; if (overlay.hasPointerCapture(event.pointerId)) overlay.releasePointerCapture(event.pointerId); renderBoxes(); renderOverlay();
  });
  overlay.addEventListener('pointercancel', cancelGesture);
  $('image-viewport').addEventListener('keydown', event => {
    if (event.key === 'Escape') { cancelGesture(); state.selectedBox = null; renderBoxes(); renderOverlay(); return; }
    if (!canDraw() || state.selectedBox == null || event.target !== $('image-viewport')) return;
    if (['Delete', 'Backspace'].includes(event.key)) { event.preventDefault(); removeBox(state.selectedBox); return; }
    const directions = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }, direction = directions[event.key];
    if (direction) { event.preventDefault(); const box = draft().boxes[state.selectedBox], step = event.shiftKey ? 10 : 1; box.bbox = translateBox(box.bbox, direction[0] * step, direction[1] * step, frame().width, frame().height); markDirty(); renderBoxes(); renderOverlay(); }
  });

  let pan = null;
  $('image-viewport').addEventListener('pointerdown', event => {
    if (state.tool !== 'pan' || event.pointerType === 'touch' || !state.imageReady) return;
    pan = { x: event.clientX, y: event.clientY, left: $('image-viewport').scrollLeft, top: $('image-viewport').scrollTop, pointerId: event.pointerId };
    $('image-viewport').setPointerCapture(event.pointerId); $('image-viewport').classList.add('dragging');
  });
  $('image-viewport').addEventListener('pointermove', event => { if (pan?.pointerId !== event.pointerId) return; $('image-viewport').scrollLeft = pan.left + pan.x - event.clientX; $('image-viewport').scrollTop = pan.top + pan.y - event.clientY; });
  for (const eventName of ['pointerup', 'pointercancel']) $('image-viewport').addEventListener(eventName, () => { pan = null; $('image-viewport').classList.remove('dragging'); });
  for (const input of document.querySelectorAll('input[name="visibility"]')) input.addEventListener('change', () => {
    if (!editable()) return;
    if (input.value !== 'visible' && draft().boxes.length) { renderDraft(); reviewError('Your boxes are still in your draft. Remove them to choose “No mushrooms visible” or “Not sure”, or keep “Yes” if you can see mushrooms.'); return; }
    draft().status = input.value; if (input.value === 'visible') state.tool = 'draw'; if (input.value === 'uncertain') { draft().complete = false; $('review-complete').checked = false; } markDirty(); renderAI(); updateControls();
  });
  $('add-box').addEventListener('click', () => { if (!canDraw()) return; const current = frame(), w = Math.max(2, Math.round(current.width * .2)), h = Math.max(2, Math.round(current.height * .2)), x = Math.round((current.width - w) / 2), y = Math.round((current.height - h) / 2); draft().boxes.push({ bbox: [x, y, x + w, y + h], group: 'unknown' }); state.selectedBox = draft().boxes.length - 1; markDirty('A box was added at the center. Move it to the mushroom, or open its coordinate fields.'); renderBoxes(); renderOverlay(); });
  $('draw-tool').addEventListener('click', () => { state.tool = 'draw'; updateControls(); });
  $('pan-tool').addEventListener('click', () => { cancelGesture(); state.tool = 'pan'; updateControls(); });
  $('fit-image').addEventListener('click', () => setZoom('fit'));
  $('original-image').addEventListener('click', () => setZoom('manual', 1));
  $('zoom-in').addEventListener('click', () => setZoom('manual', state.zoom * 1.4));
  $('zoom-out').addEventListener('click', () => setZoom('manual', state.zoom / 1.4));
  $('image-retry').addEventListener('click', () => { if (frame()) startImage(frame(), state.sequence); });
  $('review-complete').addEventListener('change', () => { draft().complete = $('review-complete').checked; markDirty(); });
  $('withdraw-permission').addEventListener('click',withdrawPermission);
  $('show-ai').addEventListener('click', () => { if (!editable() || !draft().status) return; const value = draft(); value.showAI = !value.showAI; if (value.showAI) { value.aiAssisted = true; markDirty('You viewed AI suggestions. This review will be marked AI-assisted.'); } renderAI(); });
  $('refresh-ai').addEventListener('click', refreshAI);
  $('review-form').addEventListener('submit', event => { event.preventDefault(); saveReview(true); });
  $('save-stay').addEventListener('click', () => saveReview(false));
  $('previous-frame').addEventListener('click', () => openFrame(state.index - 1));
  $('next-frame').addEventListener('click', () => openFrame(state.index + 1));
  $('next-priority').addEventListener('click',()=>{const seen=new Set([...state.drafts].filter(([,value])=>value.revision).map(([id])=>id)),index=priorityFrameIndex(state.frames,frame()?.id,seen);if(index>=0)openFrame(index);});
  $('review-retry').addEventListener('click', () => state.retry?.());
  $('source-filter').addEventListener('change', () => { state.source = $('source-filter').value; state.frames = state.collection.frames.filter(item => !state.source || item.sourceId === state.source); openFrame(0); });
  window.addEventListener('beforeunload', event => { if ([...state.drafts.values()].some(value => value.dirty)) { event.preventDefault(); event.returnValue = ''; } });
  new ResizeObserver(() => { if (state.zoomMode === 'fit' && state.imageReady) setZoom('fit'); }).observe($('image-viewport'));

  async function initialize() {
    state.loading = true; $('page-error').hidden = true; $('page-status').textContent = 'Loading the field collection.'; updateControls();
    try {
      const results = await Promise.all([api('session', { method: 'POST', body: {} }), api('collection')]);
      state.csrf = results[0].csrfToken; state.collection = results[1];setTerms(results[0].participationTerms);
      if (!state.csrf || !Array.isArray(state.collection.sources) || !Array.isArray(state.collection.frames)) throw new Error('The collection could not be opened. Please try again.');
      state.frames = state.collection.frames; renderCounts(); $('source-filter').replaceChildren();
      const all = create('option', '', 'All videos'); all.value = ''; $('source-filter').append(all);
      for (const source of state.collection.sources) { const option = create('option', '', source.title); option.value = source.id; $('source-filter').append(option); }
      if (!state.frames.length) { $('image-message-text').textContent = 'The first field collection is being prepared. Please check back soon.'; $('image-message').querySelector('.loading-ring').hidden = true; state.loading = false; updateControls(); announce('The collection has no frames yet.'); return; }
      const requestedFrame = new URLSearchParams(location.search).get('frame');
      const requestedIndex = requestedFrame ? state.frames.findIndex(item => item.id === requestedFrame) : -1;
      state.loading = false; await openFrame(requestedIndex >= 0 ? requestedIndex : 0);
    } catch (error) {
      state.loading = false; $('page-error-text').textContent = error.message; $('page-error').hidden = false; $('image-message-text').textContent = 'The collection could not be opened.'; $('image-message').querySelector('.loading-ring').hidden = true; updateControls();
    }
  }
  $('page-retry').addEventListener('click', initialize);
  initialize();
  refreshLearning();setInterval(()=>{if(!document.hidden)refreshLearning();},60000);
}
