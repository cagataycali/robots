// Run: npx esbuild src/lib/endpoints.ts --bundle --format=esm --outfile=/tmp/endpoints.mjs && node src/lib/endpoints.test.mjs
//
// endpoints.ts is the one place EVERY screen's fetch, auth header, token storage and 404 explanation
// passes through, and it had no test at all. Three defects fixed here are the reason it needs one, and
// each is a silent lie rather than a crash:
//   1. normalize() answered `new URL().origin` for any scheme, and `foo://bar` / `file:///x` return the
//      STRING "null" — so a typo in the backend field made every later request go to "nullapi/fleet".
//   2. absorbUrl() used to live inside backendBase(), but api() reads authToken() BEFORE resolving the
//      URL — so the first request of a `?token=` link went out unauthenticated and came back 401, which
//      the AuthGate shows as a login form to someone who just clicked an authorised link.
//   3. The live route list (Q79) belongs to ONE server. Kept across a backend switch, the old server's
//      routes explain the new server's 404s: "restart your dashboard" about a route that exists.
// Module state (cachedBase, absorbedUrl, _liveRoutes) is per-import, so a case needing a different
// location/localStorage re-imports the bundle with a `?case=N` cache-buster.
import assert from 'node:assert/strict'

const store = new Map()
globalThis.localStorage = {
  getItem: k => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: k => store.delete(k),
}
globalThis.location = { search: '', host: 'dash.local', origin: 'http://dash.local' }

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

let openapiFetches = 0
let routes = ['/api/fleet']
globalThis.fetch = async (url, init) => {
  if (String(url).includes('/openapi.json')) {
    openapiFetches += 1
    return json({ paths: Object.fromEntries(routes.map(r => [r, {}])) })
  }
  lastRequest = { url: String(url), headers: (init && init.headers) || {} }
  // A RESOURCE 404 from a route that exists. Deliberately not the SPA catch-all's wording
  // ("not found" / "no endpoint at ..."), which is the server's own signal for an unrouted path.
  return json({ detail: 'no dataset directory at /tmp/x' }, 404)
}
let lastRequest = null

const mod = await import('/tmp/endpoints.mjs')

// ── 1. a typo must land as "same origin", never as an address that cannot exist ──
assert.equal(mod.normalize('robot.lan:8080'), 'http://robot.lan:8080', 'a bare host gets http://')
assert.equal(mod.normalize(' robot.lan:8080/ '), 'http://robot.lan:8080', 'trimmed, no trailing slash')
assert.equal(mod.normalize('ws://robot.lan:8080'), 'http://robot.lan:8080', 'ws:// typed in is accepted')
assert.equal(mod.normalize('wss://robot.lan'), 'https://robot.lan', 'wss:// becomes https')
for (const poison of ['foo://bar', 'file:///x', 'about:blank']) {
  const got = mod.normalize(poison)
  assert.equal(got, '', `${poison} is not a fetchable base, so it means same origin`)
  assert.notEqual(got, 'null', `${poison} must never become the STRING "null"`)
}
assert.equal(mod.normalize('ftp://robot.lan:21'), '', 'a scheme fetch cannot speak is refused whole')
assert.equal(mod.normalize(''), '')
assert.equal(mod.normalize(null), '', 'no value is same origin, not a throw')

// same origin means the path is used verbatim — the shape every dev-proxy setup relies on
assert.equal(mod.apiUrl('/api/fleet'), '/api/fleet')
mod.setBackendBase('robot.lan:8080')
assert.equal(mod.apiUrl('/api/fleet'), 'http://robot.lan:8080/api/fleet')
mod.setBackendBase('foo://bar')
assert.equal(mod.apiUrl('/api/fleet'), '/api/fleet', 'a poisoned base falls back, it does not prefix')
assert.ok(!mod.apiUrl('/api/fleet').startsWith('null'), 'the "nullapi/fleet" regression')

// ── 3. the route list belongs to the server it came from ──
mod.setBackendBase('old.lan:8090')
routes = ['/api/fleet'] // this server does NOT route /api/record/open
let err = await mod.api('/api/record/open').then(() => null, e => e)
assert.equal(err.status, 404)
assert.equal(openapiFetches, 1, 'the route list is fetched on the FIRST 404 only, never on success')
assert.match(err.message, /restart/i, 'a route this server lacks is explained as a stale server')

err = await mod.api('/api/record/open').then(() => null, e => e)
assert.equal(openapiFetches, 1, 'and at most once per backend')

routes = ['/api/fleet', '/api/record/open'] // the NEW server has the route
mod.setBackendBase('new.lan:8090')
err = await mod.api('/api/record/open').then(() => null, e => e)
assert.equal(openapiFetches, 2, 'switching backends forgets the old server route list (it re-asks)')
assert.equal(err.message, 'no dataset directory at /tmp/x',
  "a route the new server HAS keeps the server's own words about the resource")
assert.doesNotMatch(err.message, /restart/i, 'no restart advice about a route that exists')

// ── 2. a ?token= link is authorised on its FIRST request, not its second ──
store.clear()
globalThis.location = { search: '?token=urltok&backend=robot.lan:9000', host: 'dash.local', origin: 'http://dash.local' }
const fresh = await import('/tmp/endpoints.mjs?case=token')
await fresh.api('/api/fleet').then(() => null, e => e)
assert.equal(lastRequest.headers.Authorization, 'Bearer urltok',
  'the very first request of a ?token= link carries the header (else the AuthGate shows a login form)')
assert.equal(lastRequest.url, 'http://robot.lan:9000/api/fleet', '?backend= is absorbed too')
assert.equal(fresh.authToken(), 'urltok', 'and the token persists for the rest of the session')
assert.match(fresh.wsUrl('/ws'), /^ws:\/\/robot\.lan:9000\/ws\?token=urltok$/,
  'a websocket cannot set headers, so the token rides the query string')

// a token given by the login form replaces it; clearing it removes it, so backendKey() remounts the tree
fresh.setAuthToken(' formtok ')
assert.equal(fresh.authToken(), 'formtok', 'trimmed — a pasted token usually carries whitespace')
assert.equal(fresh.backendKey(), 'http://robot.lan:9000|auth')
fresh.setAuthToken('')
assert.equal(fresh.authToken(), '')
assert.equal(fresh.backendKey(), 'http://robot.lan:9000|open', 'the key changes, so the tree remounts')

// ── a dead backend is a reachability error naming the address, never a silent idle fleet ──
globalThis.fetch = async () => { throw new TypeError('Failed to fetch') }
err = await fresh.api('/api/fleet').then(() => null, e => e)
assert.equal(err.status, 0, 'no HTTP status, because no HTTP happened')
assert.match(err.message, /cannot reach robot\.lan:9000/, 'the operator is told WHICH address is dead')

console.log('endpoints.test.mjs: all assertions passed')

// ── Q102: the page remembers being refused, so a socket-shaped failure can be read for what it is ──
{
  const m = await import('/tmp/endpoints.mjs?case=q102')
  assert.equal(m.authRefusedRecently(), false, 'a fresh page has not been refused')
  m.noteAuthRefusal(401, 1_000)
  assert.equal(m.authRefusedRecently(60_000, 5_000), true)
  assert.equal(m.authRefusedRecently(60_000, 90_000), false, 'a refusal from ten minutes ago says nothing about now')
  m.noteAuthRefusal(500, 100_000)
  assert.equal(m.authRefusedRecently(60_000, 100_001), false, 'a 500 is not a refusal')
  m.noteAuthRefusal(403, 200_000)
  assert.equal(m.authRefusedRecently(60_000, 200_001), true, '403 counts too — the middleware uses both')
  m.noteAuthAccepted()
  assert.equal(m.authRefusedRecently(60_000, 200_002), false, 'a successful request must not leave an accusation behind')
}

console.log('endpoints.test.mjs: Q102 refusal memory ok')

// ── Q103: a PUBLIC 200 must not absolve a refused page ─────────────────────────────────────────────
// Measured in a browser against the live dashboard: after the token was rotated, /api/auth/status
// (public — the middleware never looks at it) kept answering 200 and cleared the refusal within a
// second, so AuthGate's watcher never saw one and the page stayed open, deaf, exactly as before.
{
  const m = await import('/tmp/endpoints.mjs?case=q103')
  m.noteAuthRefusal(401, 1_000)
  m.noteAuthAccepted('/api/auth/status')
  assert.equal(m.authRefusedRecently(60_000, 2_000), true, 'a public route proves nothing about credentials')
  m.noteAuthAccepted('/api/health')
  assert.equal(m.authRefusedRecently(60_000, 2_000), true)
  m.noteAuthAccepted('/api/auth/login/begin')
  assert.equal(m.authRefusedRecently(60_000, 2_000), true)
  m.noteAuthAccepted('/api/fleet')
  assert.equal(m.authRefusedRecently(60_000, 2_000), false, 'a GUARDED success is the proof, and it clears')
}

console.log('endpoints.test.mjs: Q103 public-200 absolution ok')

// ── Q104: EVERY fetcher in this module records a refusal, not just api() ────────────────────────────
// apiBlob is the camera-preview rail: on the fleet screen it is usually the FIRST thing a rotated token
// refuses, and it had its own fetch() with no accounting — so the tiles went dark while the refusal
// memory stayed empty, and both planRetry and AuthGate's watcher were left with nothing to act on.
{
  const m = await import('/tmp/endpoints.mjs?case=q104-blob')
  globalThis.localStorage = { getItem: () => 'a-token', setItem() {}, removeItem() {} }
  globalThis.fetch = async () => ({ ok: false, status: 401, statusText: 'Unauthorized', text: async () => '' })
  await assert.rejects(() => m.apiBlob('/api/devices/camera/0/preview'))
  assert.equal(m.authRefusedRecently(), true, 'a refused camera preview is evidence like any other')
}
{
  // ...and a successful GUARDED blob clears it, the same rule api() follows.
  const m = await import('/tmp/endpoints.mjs?case=q104-blob-ok')
  globalThis.localStorage = { getItem: () => 'a-token', setItem() {}, removeItem() {} }
  globalThis.URL.createObjectURL = () => 'blob:stub'
  m.noteAuthRefusal(401)
  globalThis.fetch = async () => ({ ok: true, status: 200, blob: async () => 'bytes' })
  await m.apiBlob('/api/devices/camera/0/preview')
  assert.equal(m.authRefusedRecently(), false)
}

console.log('endpoints.test.mjs: Q104 every fetcher accounts ok')

// --- U21: a sliding session renews itself on an ordinary response ---------------------
// The JWT lives 24h and had no renewal route, so the phone that signed in on Monday was
// refused on Tuesday and its socket knocked 18,968 times over 44 hours (Q109). The server
// now hands a fresh token back on X-Session-Token; absorbing it HERE is why no screen
// needs to know renewal exists.
const JWT_A = 'aaa.bbb.ccc'
const JWT_B = 'ddd.eee.fff'

function storeWith(initial) {
  // A previous section left ?token=urltok in the stubbed location, and absorbUrl() would
  // adopt it INTO this case — the same per-module state trap the ?case= imports exist for.
  globalThis.location = { search: '', host: 'dash.local', origin: 'http://dash.local' }
  const cell = { value: initial }
  globalThis.localStorage = {
    getItem: k => (k === 'strands.token' ? cell.value : null),
    setItem: (k, v) => { if (k === 'strands.token') cell.value = v },
    removeItem: () => { cell.value = null },
  }
  return cell
}
const withHeader = value => ({
  ok: true, status: 200, statusText: 'OK',
  headers: { get: n => (n.toLowerCase() === 'x-session-token' ? value : null) },
  text: async () => '{}',
})

{
  const m = await import('/tmp/endpoints.mjs?case=u21-renew')
  const cell = storeWith(JWT_A)
  globalThis.fetch = async () => withHeader(JWT_B)
  await m.api('/api/fleet')
  assert.equal(cell.value, JWT_B, 'the renewed session must be STORED, or tomorrow logs the phone out again')
  // AuthGate reads this to tell "renewal stopped working" apart from "this server never renews"
  assert.ok(m.lastRenewalAt() > 0, 'an accepted renewal is an OBSERVED fact the banner can use')
}
{
  // silence changes nothing: an older server sends no such header, and a session must
  // not be disturbed by the absence of news.
  const m = await import('/tmp/endpoints.mjs?case=u21-silent')
  const cell = storeWith(JWT_A)
  globalThis.fetch = async () => withHeader(null)
  await m.api('/api/fleet')
  assert.equal(cell.value, JWT_A)
  assert.equal(m.lastRenewalAt(), 0, 'an older server that never renews must not look like one that stopped')
}
{
  // a header with NO stored token is ignored: we never had a session to renew, and
  // accepting a credential we did not ask for is how a shared proxy hands you someone
  // else's session.
  const m = await import('/tmp/endpoints.mjs?case=u21-unasked')
  const cell = storeWith(null)
  globalThis.fetch = async () => withHeader(JWT_B)
  await m.api('/api/fleet')
  assert.equal(cell.value, null)
}
{
  // garbage never replaces a WORKING credential — a renewal is an improvement or nothing.
  const m = await import('/tmp/endpoints.mjs?case=u21-garbage')
  const cell = storeWith(JWT_A)
  for (const bad of ['not-a-jwt', 'two.parts', '   ', 'a.b.c.d']) {
    globalThis.fetch = async () => withHeader(bad)
    await m.api('/api/fleet')
    assert.equal(cell.value, JWT_A, `refused: ${bad}`)
  }
  // the same token back is not a write either (listeners must not wake per request)
  assert.equal(m.absorbRenewedSession(withHeader(JWT_A)), false)
  assert.equal(m.absorbRenewedSession(withHeader(JWT_B)), true, 'and a real one still lands')
}
{
  // a Response-like with no headers at all (stubs, blob shims) must not throw INSIDE the
  // request it is decorating: a renewal failure may never cost the caller its answer.
  const m = await import('/tmp/endpoints.mjs?case=u21-headerless')
  storeWith(JWT_A)
  assert.equal(m.absorbRenewedSession({}), false)
  assert.equal(m.absorbRenewedSession(null), false)
  assert.equal(m.absorbRenewedSession({ headers: { get() { throw new Error('no headers here') } } }), false)
}

// --- Q156: the remount key IS the change notification -------------------------------
// A listener set lived here with no subscriber, poked by both setters. It was deleted, so
// this pins what replaces it: BOTH halves of the identity change the key, which is what
// remounts App. If someone reintroduces a callback rail, this still passes — but if a
// setter ever stops moving the key, the switch becomes invisible and this fails.
{
  const m = await import('/tmp/endpoints.mjs?case=remountkey')
  m.setBackendBase('one.lan:8090')
  m.setAuthToken('t1')
  const first = m.backendKey()
  m.setBackendBase('two.lan:8090')
  assert.notEqual(m.backendKey(), first, 'a new backend must move the key — nothing else tells the tree')
  const second = m.backendKey()
  m.setAuthToken('')
  assert.notEqual(m.backendKey(), second, 'clearing the token must move it too (AuthGate depends on this)')
}


// --- Q161: a remembered route list must not outlive the server that gave it ------------
// The cache is consulted ONLY to explain a 404, and its wrong answer is the dangerous one:
// after a restart it keeps saying "restart the dashboard to pick it up" about routes that
// now exist. Asserted against the REAL TTL, with Date.now driven so the test does not sleep.
{
  const m = await import('/tmp/endpoints.mjs?case=q161')
  const realNow = Date.now
  let t = 1_800_000_000_000
  globalThis.Date.now = () => t
  let openapiFetches = 0
  globalThis.fetch = async (url) => {
    const u = String(url)
    if (u.includes('/openapi.json')) {
      openapiFetches++
      return { ok: true, status: 200, json: async () => ({ paths: { '/api/fleet': {} } }) }
    }
    // Every other call is a 404 for a path the list above does not contain, which is what
    // makes the route list be read at all.
    return {
      ok: false, status: 404, headers: { get: () => 'application/json' },
      json: async () => ({ detail: 'Not Found' }), text: async () => '{}',
    }
  }
  m.setAuthToken('tkn')

  const ask = async () => {
    try { await m.api('/api/deploy/snippet') } catch (e) { return e }
  }
  const first = await ask()
  assert.match(first.message, /restart/i, 'a route the server does not have is explained, not left as HTTP 404')
  assert.equal(openapiFetches, 1, 'the list is fetched once, not per call')

  await ask()
  assert.equal(openapiFetches, 1, 'inside the TTL the remembered list still speaks — no request per 404')

  // The server restarts and now HAS the route. Nothing tells the tab; only time passes.
  t += LIVE_ROUTES_TTL_MS_FROM(m) + 1
  await ask()
  assert.equal(openapiFetches, 2,
    'past the TTL the list is re-read, so the restart advice cannot outlive the restart')

  // The explicit seam still works, and is what a backend switch uses.
  m.forgetLiveRoutes()
  await ask()
  assert.equal(openapiFetches, 3, 'forgetLiveRoutes() drops the list immediately, without waiting for the TTL')

  globalThis.Date.now = realNow
}
function LIVE_ROUTES_TTL_MS_FROM(m) {
  assert.equal(typeof m.LIVE_ROUTES_TTL_MS, 'number', 'the TTL is exported so this test asserts the REAL number')
  return m.LIVE_ROUTES_TTL_MS
}

console.log('endpoints.test.mjs: U21 sliding session ok')
