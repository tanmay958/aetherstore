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
    $('fact-docs').textContent = '—';
  }
  try {
    const model = await api('/model');
    const pr = model.metrics?.model?.pr_auc;
    $('fact-model').textContent = pr ? pr.toFixed(3) : (model.loaded ? 'loaded' : '—');
    $('fact-model').title = pr ? 'PR-AUC on a held-out future' : '';
  } catch {
    $('fact-model').textContent = '—';
  }
}

/* ── search ────────────────────────────────────────────────────────── */

const form = $('search-form');

async function runSearch(event) {
  event?.preventDefault();
  const q = $('q').value.trim();
  if (!q) return;

  const button = form.querySelector('button');
  button.disabled = true;
  $('search-status').className = 'status';
  $('search-status').textContent = 'searching…';

  try {
    const params = new URLSearchParams({ q, k: '10', mode: $('mode').value, explain: 'true' });
    const data = await api(`/search?${params}`);

    $('cost').hidden = false;
    $('cost-took').textContent = `${Math.round(data.took_ms)} ms`;
    $('cost-requests').textContent = commas(data.cost.requests);
    $('cost-bytes').textContent = bytes(data.cost.bytes);
    $('cost-segments').textContent = `${data.cost.segments_searched}/${data.cost.segments_total}`;
    $('cost-pruned').textContent = data.cost.segments_pruned;

    $('search-status').textContent =
      `${commas(data.total)} matching event${data.total === 1 ? '' : 's'}, showing ${data.hits.length}`;

    $('results').replaceChildren(...data.hits.map((hit) => {
      const d = hit.document;
      const li = document.createElement('li');
      const title = document.createElement('div');
      title.className = 'title';
      title.textContent = d.title || d.product_id || '(untitled)';
      const price = document.createElement('div');
      price.className = 'price';
      price.textContent = d.price != null ? money(d.price) : '—';
      const meta = document.createElement('div');
      meta.className = 'meta';
      // The index holds events, not products, so the same product legitimately
      // appears several times. Without the timestamp those rows look like a
      // bug rather than like three people viewing one phone.
      const when = new Date(d.ts * 1000).toISOString().slice(0, 19).replace('T', ' ');
      for (const [cls, text] of [
        ['score', `BM25 ${hit.score.toFixed(3)}`],
        ['', d.event_type],
        ['', when],
        ['', d.brand || 'no brand'],
        ['', d.category || 'no category'],
      ]) {
        const span = document.createElement('span');
        if (cls) span.className = cls;
        span.textContent = text;
        meta.append(span);
      }
      li.append(title, price, meta);
      return li;
    }));

    if (!data.hits.length) {
      $('search-status').textContent = 'nothing matched. Try "any term", or a broader query.';
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

form.addEventListener('submit', runSearch);

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
    value.textContent = '—';
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

/* ── go ────────────────────────────────────────────────────────────── */

loadFacts();
runSearch();
