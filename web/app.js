/* AetherStore dashboard.
 *
 * Talks to its own origin by default, where a Cloudflare Pages Function
 * forwards to Cloud Run. That indirection is not decoration: a browser cannot
 * hold a secret, so anything this file knows is public. Keeping the API key
 * in the Function means a visitor can use the service without being handed
 * the credential that authorises it.
 *
 * `?api=https://...` overrides the base, which is how this page is developed
 * against the deployed service without a proxy in front of it.
 */

const API = new URLSearchParams(location.search).get('api') || '/api';

const $ = (id) => document.getElementById(id);

function bytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1048576).toFixed(2)} MB`;
}
const commas = (n) => n.toLocaleString('en-US');
const money = (n) => `$${n.toFixed(2)}`;

async function api(path, options) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      // FastAPI reports validation errors as a list and everything else as a
      // string, and a visitor should see the sentence either way.
      if (typeof body.detail === 'string') detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((e) => e.msg).join('; ');
    } catch { /* keep the status line */ }
    throw new Error(detail);
  }
  return response.json();
}

/* ── tabs ──────────────────────────────────────────────────────────── */

document.querySelectorAll('.tabs button').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tabs button').forEach((other) => {
      const on = other === tab;
      other.setAttribute('aria-selected', String(on));
      $(`panel-${other.dataset.panel}`).hidden = !on;
    });
  });
});

/* ── header facts ──────────────────────────────────────────────────── */

async function loadFacts() {
  try {
    // /api/index rather than /health, so every call this page makes sits
    // under one prefix and the proxy in front of it needs one rule.
    const index = await api('/index');
    $('fact-docs').textContent = commas(index.documents ?? 0);
    $('fact-segments').textContent = index.segments ?? '·';
    $('fact-bytes').textContent = bytes(index.bytes ?? 0);
  } catch {
    $('fact-docs').textContent = '-';
  }
  try {
    const model = await api('/model');
    const pr = model.metrics?.model?.pr_auc;
    $('fact-model').textContent = pr ? pr.toFixed(3) : (model.loaded ? 'loaded' : '-');
    $('fact-model').title = pr ? 'PR-AUC on a held-out future' : '';
  } catch {
    $('fact-model').textContent = '-';
  }
}

/* ── search ────────────────────────────────────────────────────────── */

const form = $('search-form');

/* Page state lives here rather than in the URL: the page is one screen and a
 * visitor paging through results is not navigating. */
let page = { q: '', offset: 0, limit: 10, total: 0, max: 500 };

async function runSearch(event, offset = 0) {
  event?.preventDefault();
  const q = $('q').value.trim();
  if (!q) return;
  page.offset = offset;

  const button = form.querySelector('button');
  button.disabled = true;
  $('search-status').className = 'status';
  $('search-status').textContent = 'searching…';

  try {
    const params = new URLSearchParams({
      q, k: String(page.limit), offset: String(page.offset),
      mode: $('mode').value, explain: 'true',
    });
    const data = await api(`/search?${params}`);

    $('cost').hidden = false;
    $('cost-took').textContent = `${Math.round(data.took_ms)} ms`;
    $('cost-requests').textContent = commas(data.cost.requests);
    $('cost-bytes').textContent = bytes(data.cost.bytes);
    $('cost-segments').textContent = `${data.cost.segments_searched}/${data.cost.segments_total}`;
    $('cost-pruned').textContent = data.cost.segments_pruned;

    page = { ...page, q, total: data.total, max: data.max_offset ?? 500 };
    const from = data.total ? page.offset + 1 : 0;
    const to = page.offset + data.hits.length;
    $('search-status').textContent =
      `${commas(data.total)} matching event${data.total === 1 ? '' : 's'}` +
      (data.total ? `, showing ${commas(from)}\u2013${commas(to)}` : '');
    renderPager();

    const best = data.hits.length ? data.hits[0].score : 1;
    $('results').replaceChildren(
      ...data.hits.map((hit, i) => renderHit(hit, i, best)),
    );

    if (!data.hits.length && !page.offset) {
      $('search-status').textContent =
        'nothing matched. Try "any term", or index it yourself under Live index.';
    }
    rememberProducts(data.hits);
  } catch (error) {
    $('search-status').className = 'status error';
    $('search-status').textContent = error.message;
    $('results').replaceChildren();
  } finally {
    button.disabled = false;
  }
}

function renderHit(hit, i, best) {
  const d = hit.document;
  const li = document.createElement('li');

  const row = document.createElement('div');
  row.className = 'row';
  const left = document.createElement('div');
  const rank = document.createElement('span');
  rank.className = 'rank';
  rank.textContent = `${page.offset + i + 1}.`;
  const title = document.createElement('span');
  title.className = 'title';
  title.textContent = d.title || d.product_id || '(untitled)';
  left.append(rank, title);
  const price = document.createElement('div');
  price.className = 'price';
  price.textContent = d.price != null ? money(d.price) : '\u2014';
  row.append(left, price);

  const meta = document.createElement('div');
  meta.className = 'meta';
  // The index holds events, not products, so one product legitimately appears
  // many times. The timestamp is what stops those rows looking like a bug.
  const when = new Date(d.ts * 1000).toISOString().slice(0, 19).replace('T', ' ');
  const tags = [
    ['tag kind', d.event_type],
    ['tag', d.brand || 'no brand'],
    ['tag', d.category || 'no category'],
  ];
  if (justIndexed.has(d.product_id)) tags.push(['tag fresh', 'you indexed this']);
  for (const [cls, text] of tags) {
    const span = document.createElement('span');
    span.className = cls;
    span.textContent = text;
    meta.append(span);
  }
  const stamp = document.createElement('span');
  stamp.textContent = `${when}  ·  BM25 ${hit.score.toFixed(3)}`;
  meta.append(stamp);

  // Relative to the best hit on this page, so the ranking is visible without
  // anyone having to know what a BM25 score means.
  const bar = document.createElement('div');
  bar.className = 'bar';
  const fill = document.createElement('i');
  fill.style.width = `${Math.max(4, (hit.score / best) * 100)}%`;
  bar.append(fill);

  li.append(row, meta, bar);
  return li;
}

function renderPager() {
  const pages = Math.ceil(page.total / page.limit);
  const here = Math.floor(page.offset / page.limit) + 1;
  const capped = page.offset + page.limit > page.max;

  $('pager').hidden = page.total <= page.limit;
  $('page-label').textContent = `page ${commas(here)} of ${commas(pages)}`;
  $('prev').disabled = page.offset === 0;
  $('next').disabled = here >= pages || capped;
  $('depth-note').hidden = !capped;
}

$('prev').addEventListener('click', () =>
  runSearch(null, Math.max(0, page.offset - page.limit)));
$('next').addEventListener('click', () =>
  runSearch(null, page.offset + page.limit));

form.addEventListener('submit', (e) => runSearch(e, 0));

/* ── prediction ────────────────────────────────────────────────────── */

/* Products the visitor has actually seen in a search, so the prediction panel
 * scores things that exist in the index rather than invented ones. */
const catalogue = new Map();

function rememberProducts(hits) {
  for (const hit of hits) {
    const d = hit.document;
    if (!d.product_id || catalogue.has(d.product_id)) continue;
    catalogue.set(d.product_id, {
      product_id: d.product_id,
      title: d.title || d.product_id,
      price: d.price ?? 0,
      brand: d.brand,
      category: d.category,
    });
  }
  renderCatalogue();
}

function renderCatalogue() {
  const select = $('product');
  const chosen = select.value;
  select.replaceChildren(...[...catalogue.values()].slice(0, 40).map((p) => {
    const option = document.createElement('option');
    option.value = p.product_id;
    option.textContent = `${p.title} · ${money(p.price)}`;
    return option;
  }));
  if (chosen && catalogue.has(chosen)) select.value = chosen;
}

/* The session under construction. Timestamps are synthetic and advance only
 * when the visitor asks them to, because idle time is one of the things the
 * model reads and letting wall clock drive it would make the demo
 * unreproducible. */
let session = [];
let clock = 1570000000;

function addEvent(kind) {
  const product = catalogue.get($('product').value);
  if (!product && kind !== 'purchase') return;
  clock += 20;
  session.push({
    ts: clock,
    event_type: kind,
    product_id: product?.product_id,
    price: product?.price,
    brand: product?.brand,
    category: product?.category,
  });
  score();
}

async function score() {
  if (!session.length) return render(null);
  try {
    const data = await api('/predict', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ events: session }),
    });
    render(data);
  } catch (error) {
    $('gauge-label').textContent = error.message;
  }
}

function render(data) {
  const scores = data?.scores ?? [];
  const last = scores[scores.length - 1];
  // Only the most recent event, never the most recent *scored* event. Falling
  // back through history left the gauge reading "30% chance of abandoning
  // this cart" immediately after the cart had been bought, which is a
  // confident answer to a question that no longer exists.
  const latest = last?.scored ? last : null;

  const value = $('gauge-value');
  const fill = $('gauge-fill');
  if (latest) {
    const pct = latest.probability;
    value.textContent = `${(pct * 100).toFixed(0)}%`;
    const level = pct >= 0.7 ? 'danger' : pct >= 0.5 ? 'warn' : 'live';
    value.className = `gauge-value ${level}`;
    fill.className = `gauge-fill ${level === 'live' ? '' : level}`;
    fill.style.width = `${pct * 100}%`;
    $('gauge-label').textContent = pct >= 0.7 ? 'likely to abandon this cart' : 'chance of abandoning this cart';
  } else {
    value.textContent = '-';
    value.className = 'gauge-value';
    fill.style.width = '0';
    fill.className = 'gauge-fill';
    $('gauge-label').textContent = last ? (last.reason ?? 'not scored yet') : 'no cart yet';
  }

  $('g-events').textContent = last?.events ?? 0;
  $('g-cart').textContent = last?.cart_size ?? 0;
  $('g-value').textContent = money(last?.cart_value ?? 0);

  $('timeline').replaceChildren(...scores.map((s, i) => {
    const li = document.createElement('li');
    if (!s.scored) li.classList.add('unscored');
    const kind = document.createElement('span');
    kind.className = 'kind';
    kind.textContent = s.event_type;
    const what = document.createElement('span');
    what.className = 'what';
    const gap = i > 0 ? s.ts - scores[i - 1].ts : 0;
    what.textContent = gap > 60 ? `after ${Math.round(gap / 60)} min idle` : (session[i]?.product_id ? catalogue.get(session[i].product_id)?.title ?? '' : '');
    const p = document.createElement('span');
    p.className = s.scored ? 'p' : 'p none';
    p.textContent = s.scored ? `${(s.probability * 100).toFixed(1)}%` : 'not scored';
    if (!s.scored && s.reason) li.title = s.reason;
    li.append(kind, what, p);
    return li;
  }).reverse());
}

document.querySelectorAll('[data-event]').forEach((button) =>
  button.addEventListener('click', () => addEvent(button.dataset.event)));

document.querySelectorAll('[data-wait]').forEach((button) =>
  button.addEventListener('click', () => {
    if (!session.length) return;
    clock += Number(button.dataset.wait);
    // Idle time only becomes visible to the model through the next event, so
    // waiting is followed by a view rather than being an event of its own.
    addEvent('view');
  }));

$('reset').addEventListener('click', () => {
  session = [];
  clock = 1570000000;
  $('timeline').replaceChildren();
  render(null);
});

/* ── live index ────────────────────────────────────────────────────── */

/* Events this visitor indexed, so their own rows can be marked in the
 * results. Session-scoped and never sent anywhere. */
const justIndexed = new Set();

let stream = null;          // the open EventSource, when streaming
let keepStreaming = false;  // whether to reconnect when one burst ends

function tick(text, value, kind) {
  const li = document.createElement('li');
  const label = document.createElement('span');
  label.textContent = text;
  const bold = document.createElement('b');
  bold.textContent = value;
  if (kind) li.className = kind;
  li.append(label, bold);
  $('ticker').prepend(li);
  while ($('ticker').children.length > 40) $('ticker').lastChild.remove();
}

/* `documents` is the whole index; `ingested` is only the partition this
 * service writes. Showing the latter as the total made the header collapse
 * from 142,152 to 2,153 the moment a burst began. */
function setCounts(d) {
  if (d.documents != null) {
    $('live-count').textContent = commas(d.documents);
    $('fact-docs').textContent = commas(d.documents);
  }
}

function setStreaming(on) {
  keepStreaming = on;
  $('stream-toggle').textContent = on ? 'Stop streaming' : 'Start streaming';
  $('stream-toggle').classList.toggle('primary', !on);
  $('pulse').hidden = !on;
  $('stream-rate').disabled = on;
}

function openStream() {
  // A burst is bounded by the server, because Cloud Run throttles CPU outside
  // a request and work left running after a response would stall silently.
  // Reconnecting is what makes it continuous for as long as someone watches.
  const rate = $('stream-rate').value;
  stream = new EventSource(`${API}/stream?seconds=45&rate=${rate}`);

  stream.addEventListener('start', (e) => {
    const d = JSON.parse(e.data);
    setCounts(d);
    tick('burst started', `${d.rate}/sec`);
  });

  stream.addEventListener('progress', (e) => {
    const d = JSON.parse(e.data);
    setCounts(d);
    if (d.sealed) {
      tick('segment sealed', d.sealed.split('/').pop().slice(0, 22), 'seal');
    }
  });

  stream.addEventListener('exhausted', () => {
    tick('feed exhausted', 'stopping');
    setStreaming(false);
    closeStream();
  });

  stream.addEventListener('done', (e) => {
    const d = JSON.parse(e.data);
    setCounts(d);
    closeStream();
    // The server ended the burst, not the visitor, so start another.
    if (keepStreaming) openStream();
  });

  stream.onerror = () => {
    // EventSource retries by itself, which would fight the reconnect above.
    closeStream();
    if (keepStreaming) {
      tick('reconnecting', '…');
      setTimeout(() => { if (keepStreaming) openStream(); }, 1500);
    }
  };
}

function closeStream() {
  if (stream) { stream.close(); stream = null; }
}

$('stream-toggle').addEventListener('click', () => {
  if (keepStreaming) {
    setStreaming(false);
    closeStream();
    tick('stopped', 'by you');
  } else {
    setStreaming(true);
    openStream();
  }
});

// A tab in the background should not keep indexing.
document.addEventListener('visibilitychange', () => {
  if (document.hidden && keepStreaming) {
    setStreaming(false);
    closeStream();
    tick('paused', 'tab hidden');
  }
});

/* ── indexing one event by hand ─────────────────────────────────────── */

$('event-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const note = $('event-result');
  const button = $('event-form').querySelector('button');
  button.disabled = true;
  note.className = 'note';
  note.textContent = 'indexing…';

  const title = $('ev-title').value.trim();
  const payload = {
    ts: Math.floor(Date.now() / 1000),
    event_type: $('ev-type').value,
    session_id: `web-${Math.random().toString(36).slice(2, 8)}`,
    product_id: `custom-${Math.random().toString(36).slice(2, 8)}`,
    title,
    brand: $('ev-brand').value.trim() || null,
    category: $('ev-category').value.trim() || null,
    price: Number($('ev-price').value) || null,
  };

  try {
    const result = await api('/events', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ events: [payload] }),
    });
    justIndexed.add(payload.product_id);
    // The header counts documents too, and leaving it behind makes the page
    // look like the event did not land.
    const index = await api('/index');
    $('live-count').textContent = commas(index.documents);
    $('fact-docs').textContent = commas(index.documents);
    $('fact-segments').textContent = index.segments;
    note.className = 'note good';
    note.textContent =
      `sealed into ${result.segment.split('/').pop()} in ${result.took_ms} ms `
      + '- searching for it now';
    tick('indexed by hand', title.slice(0, 22), 'seal');

    // Prove the round trip rather than asserting it: search for what was
    // just typed, and show the visitor the result.
    $('q').value = title;
    document.querySelector('[data-panel="search"]').click();
    await runSearch(null, 0);
  } catch (error) {
    note.className = 'note bad';
    note.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

/* ── go ────────────────────────────────────────────────────────────── */

loadFacts();
runSearch();

// The live panel's counter shares the index count the header already fetches.
api('/index')
  .then((d) => { $('live-count').textContent = commas(d.documents); })
  .catch(() => { $('live-count').textContent = '\u2014'; });
