/**
 * Cloudflare Pages Function: everything under /api/* is forwarded to Cloud Run.
 *
 * This exists because a browser cannot keep a secret. Any key shipped to the
 * page is visible in the network tab, so "authenticate the frontend" is not a
 * thing that can be done in the frontend. Here it can: this code runs on
 * Cloudflare's edge, the visitor never sees it, and the shared secret it adds
 * is the one thing Cloud Run will insist on.
 *
 * CORS is not an alternative. It is enforced by browsers and ignored by curl,
 * so it stops another website's JavaScript and nobody else.
 *
 * Configure in the Pages project:
 *   AETHER_ORIGIN   https://aether-441173057461.us-central1.run.app
 *   AETHER_API_KEY  the same value as the service's own AETHER_API_KEY
 *
 * With no AETHER_API_KEY set on either side this is a plain proxy, which is a
 * deliberate first step: the page works before the lock is turned.
 */

const DEFAULT_ORIGIN = 'https://aether-441173057461.us-central1.run.app';

// A visitor can reach exactly these. Without it, /api/<anything> would be a
// tunnel to any path the service ever grows, including ones added later by
// someone not thinking about this file.
const ALLOWED = new Set(['search', 'predict', 'index', 'model']);

export async function onRequest(context) {
  const { request, env, params } = context;
  const route = Array.isArray(params.path) ? params.path.join('/') : String(params.path ?? '');

  if (!ALLOWED.has(route)) {
    return json({ detail: `no such endpoint: ${route}` }, 404);
  }
  if (!['GET', 'POST'].includes(request.method)) {
    return json({ detail: 'method not allowed' }, 405);
  }

  const incoming = new URL(request.url);
  const target = new URL(`/api/${route}${incoming.search}`, env.AETHER_ORIGIN || DEFAULT_ORIGIN);

  // Built from scratch rather than forwarded, so nothing a visitor sets
  // (cookies, an Authorization header, an X-Forwarded-For they invented)
  // reaches the service by riding along.
  const headers = new Headers({ accept: 'application/json' });
  if (env.AETHER_API_KEY) headers.set('x-aether-key', env.AETHER_API_KEY);
  if (request.method === 'POST') headers.set('content-type', 'application/json');

  let response;
  try {
    response = await fetch(target, {
      method: request.method,
      headers,
      body: request.method === 'POST' ? await request.text() : undefined,
    });
  } catch (error) {
    // Cloud Run cold starts, and a scale-to-zero service can take a moment.
    return json({ detail: `upstream unreachable: ${error.message}` }, 502);
  }

  const body = await response.text();
  return new Response(body, {
    status: response.status,
    headers: {
      'content-type': response.headers.get('content-type') || 'application/json',
      // Same origin as the page, so no CORS is needed and none is granted.
      'cache-control': response.ok && request.method === 'GET'
        ? 'public, max-age=30'
        : 'no-store',
    },
  });
}

function json(body, status) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
  });
}
