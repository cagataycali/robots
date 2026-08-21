/**
 * Where the backend lives, and how we talk to it.
 *
 * The dashboard is a mesh *peer*, not a hub - the API it drives can be on this
 * origin, on a robot across the LAN, or on a box behind a VPN. Every fetch and
 * every WebSocket therefore resolves its URL here instead of hardcoding
 * `location.host`, which silently pins the whole app to whoever served the
 * bundle (and breaks entirely once the PWA is installed and opened offline).
 *
 * An empty base means "same origin" - the default, and what the dev proxy
 * expects.
 */

import { routeKnown, staleRouteMessage, unroutedByDetail } from './serverAge'
import { detailSentence } from './detailSentence'

const BASE_KEY = 'strands.backend'
const TOKEN_KEY = 'strands.token'

/** `robot.lan:8080` -> `http://robot.lan:8080`; trailing slashes trimmed. */
export function normalize(raw: string): string {
  const value = (raw ?? '').trim()
  if (!value) return ''
  const withScheme = /^[a-z]+:\/\//i.test(value) ? value : `http://${value}`
  try {
    const url = new URL(withScheme)
    // ws:// typed into the field is a natural mistake - accept it.
    if (url.protocol === 'ws:') url.protocol = 'http:'
    if (url.protocol === 'wss:') url.protocol = 'https:'
    // Only a scheme fetch can actually speak. `new URL()` is happy with far more than that, and
    // `.origin` answers for the rest in ways that quietly poison every later request: `foo://bar`
    // and `file:///x` return the STRING "null", so the base became "null" and every call went to
    // "nullapi/fleet"; `ftp://robot.lan:21` returns "ftp://robot.lan", silently dropping the port.
    // A typo in this field must land as "same origin", which is visible in the connection chip,
    // rather than as an address that cannot exist.
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return ''
    return url.origin
  } catch {
    return ''
  }
}

let cachedBase: string | null = null
let absorbedUrl = false
/** `?backend=` from the URL, once — null when the URL said nothing. */
let urlBase: string | null = null

/**
 * Take the credentials off the URL. `?backend=` makes a QR code or bookmark point a phone straight at
 * one robot; `?token=` is how a share link (and every frontend/scripts/audit-*.mjs run) arrives already
 * authorised.
 *
 * This USED TO LIVE INSIDE backendBase(), and api() reads authToken() BEFORE it resolves the URL — so
 * on a page opened with ?token=..., the very first request of the session went out with no
 * Authorization header at all and came back 401, which is the AuthGate's cue to show a login form to
 * someone who just clicked an authorised link. Absorbing has to happen before either reader answers,
 * so both call this and it runs exactly once.
 */
function absorbUrl(): void {
  if (absorbedUrl) return
  absorbedUrl = true
  try {
    const params = new URLSearchParams(location.search)
    const fromToken = params.get('token')
    if (fromToken) localStorage.setItem(TOKEN_KEY, fromToken)
    urlBase = params.get('backend')
  } catch {
    urlBase = null // no location (a test, a worker): the stored values are the whole truth
  }
}

export function backendBase(): string {
  absorbUrl()
  if (cachedBase !== null) return cachedBase
  // ?backend=... wins once, then persists.
  if (urlBase !== null) {
    cachedBase = normalize(urlBase)
    localStorage.setItem(BASE_KEY, cachedBase)
    return cachedBase
  }
  cachedBase = normalize(localStorage.getItem(BASE_KEY) ?? '')
  return cachedBase
}

export function authToken(): string {
  absorbUrl()
  return (localStorage.getItem(TOKEN_KEY) ?? '').trim()
}

export function setAuthToken(token: string): void {
  const value = token.trim()
  if (value) localStorage.setItem(TOKEN_KEY, value)
  else localStorage.removeItem(TOKEN_KEY)
  // Nothing to notify (Q156): the token is half of backendKey(), so writing it remounts
  // the App tree by itself — AuthGate.tsx:199 relies on exactly that ("remounts App via
  // backendKey(); gate re-checks and opens"). The listener set this used to poke had no
  // subscriber in either of its two call sites.
}

/** Human label for the connection chip. */
export function backendLabel(): string {
  const base = backendBase()
  return base ? base.replace(/^https?:\/\//, '') : `${location.host} (this origin)`
}

/**
 * Identity of the current connection. Used as a React `key` so switching
 * backends *remounts* the tree: every socket, frame buffer and peer map belongs
 * to one backend, and carrying them across a switch shows the old fleet under
 * the new address.
 */
export function backendKey(): string {
  return `${backendBase()}|${authToken() ? 'auth' : 'open'}`
}

export function setBackendBase(raw: string): void {
  cachedBase = normalize(raw)
  if (cachedBase) localStorage.setItem(BASE_KEY, cachedBase)
  else localStorage.removeItem(BASE_KEY)
  // The route list belongs to the server we were talking to. Kept across a switch, the OLD server's
  // routes explain the NEW one's 404s — telling the operator to restart a dashboard whose route was
  // never missing, about a resource that genuinely is not there. forgetLiveRoutes() was written for
  // exactly this and had no caller.
  forgetLiveRoutes()
  // No change notification, deliberately (Q156). There WAS a listener set here with no
  // subscriber — and it could never have gained a useful one: the only caller of this
  // function (SettingsDrawer.goConnect) calls location.reload() immediately after, and
  // App remounts on backendKey() anyway. A callback that fires microseconds before the
  // page is destroyed is not a rail, it is a decoy: the next agent reads it as the way
  // screens react to a backend switch and wires new work to something that never runs.
  // The remount IS the mechanism, and it is stronger — sockets, peer maps and frame
  // buffers all belong to the backend they came from.
}

export function apiUrl(path: string): string {
  const base = backendBase()
  return base ? `${base}${path}` : path
}

export function wsUrl(path: string): string {
  const base = backendBase()
  const origin = base || location.origin
  const url = new URL(path, origin)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  // A browser cannot set headers on a WebSocket handshake, so the token has to
  // ride in the query string (the server accepts it there for /ws only).
  const token = authToken()
  if (token) url.searchParams.set('token', token)
  return url.toString()
}

export class HttpError extends Error {
  status: number
  body: any
  constructor(status: number, message: string, body?: any) {
    super(message)
    this.name = 'HttpError'
    this.status = status
    this.body = body
  }
}

/**
 * fetch + auth + JSON + *real* errors.
 *
 * `catch {}` around a fetch is how a dashboard ends up showing a robot as idle
 * when the command never landed. Everything here throws an `HttpError` carrying
 * the server's own message so callers can show it.
 */
/**
 * The running server's own route list, fetched at most once per page (Q79). It exists to tell a 404
 * "this feature is not in this server" apart from a 404 "that camera/peer/dataset is not here" — the
 * first is fixed by restarting the dashboard, the second is real news about the resource.
 *
 * Deliberately lazy and best-effort: the fetch happens on the FIRST 404 only (never on the happy path,
 * where it would cost every page load a request for nothing), it carries no auth requirement of its own,
 * and any failure leaves the value null, which routeKnown() reports as "unknown" and nobody renders.
 */
let _liveRoutes: string[] | null = null
let _liveRoutesTried = false

async function liveRoutes(): Promise<string[] | null> {
  if (_liveRoutesTried) return _liveRoutes
  _liveRoutesTried = true
  try {
    const token = authToken()
    const res = await fetch(apiUrl('/openapi.json'), {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
    if (!res.ok) {
      // Guarded route, so a 401/403 here is as good a witness as any other refusal. It stays SILENT
      // otherwise (a server without /openapi.json is not an error) — only the accounting is added.
      noteAuthRefusal(res.status)
      return null
    }
    noteAuthAccepted('/openapi.json')
    const doc = await res.json()
    const paths = doc && doc.paths && typeof doc.paths === 'object' ? Object.keys(doc.paths) : []
    _liveRoutes = paths.length ? paths : null
  } catch {
    _liveRoutes = null // a server without /openapi.json, or no network: stay silent
  }
  return _liveRoutes
}

/** Test seam + a place for a reconnect to forget what the OLD backend routed. */
export function forgetLiveRoutes(): void {
  _liveRoutes = null
  _liveRoutesTried = false
}

/**
 * Q102: WHEN DID THIS PAGE LAST GET REFUSED?
 *
 * A websocket refused for credentials is closed by the middleware BEFORE it accepts, which uvicorn
 * turns into a failed handshake — the browser reports code 1006, never the 1008 the server sent. Q88
 * covered the case where the token is decodably EXPIRED; a token that is merely INVALID (rotated by a
 * dashboard restart, a stale ?token= link, a revoked passkey session) looks exactly like a camera that
 * will not stream, and AuthGate cannot rescue it: its check runs once on mount, with no polling.
 *
 * The evidence exists anyway, in the one place every request passes: an HTTP 401/403. Remember when it
 * last happened so a socket-shaped failure can be read for what it is.
 */
let _refusedAt: number | null = null

export function noteAuthRefusal(status: number, at: number = Date.now()): void {
  if (status === 401 || status === 403) _refusedAt = at
}

/**
 * The server's own PUBLIC_PATHS (server.py): these answer 200 whether or not the credentials are any
 * good, because the middleware never looks at them. Mirrored here because a success on one of them is
 * NOT evidence that this page is signed in — measured on the live rig, /api/auth/status is polled
 * constantly, so clearing the memory on "any success" erased every refusal within a second and the
 * gate never came back. If this list drifts from the server's, it drifts safely: an unlisted public
 * route only clears a refusal that a guarded route will re-record on its next call.
 */
const PROVES_NOTHING = [
  '/api/health',
  '/api/auth/status',
  '/api/auth/register/',
  '/api/auth/login/',
]

/**
 * Clear it when a GUARDED request succeeds — that is the only answer which proves the credentials
 * work. A stale refusal must not accuse a working session, and a public 200 must not absolve a
 * refused one.
 */
export function noteAuthAccepted(path?: string): void {
  if (path !== undefined && PROVES_NOTHING.some(p => path.startsWith(p))) return
  _refusedAt = null
}

/**
 * Has this page been refused recently enough to explain a socket that never opened? Deliberately a
 * WINDOW, not a flag: sockets and HTTP calls fail seconds apart, and a refusal from ten minutes ago
 * says nothing about now.
 */
export function authRefusedRecently(withinMs = 60_000, now: number = Date.now()): boolean {
  return _refusedAt !== null && now - _refusedAt <= withinMs
}

/**
 * U21: a sliding session renews itself on an ordinary response.
 *
 * The server hands a fresh token back on `X-Session-Token` once the current one is
 * past its half-life. Storing it here — in the ONE place every request already goes
 * through — is why no screen needs to know renewal exists.
 *
 * Measured reason this exists: the JWT lives 24h with no renewal route, so the phone
 * that signed in on Monday was refused on Tuesday and its socket knocked 18,968 times
 * over 44 hours (Q109).
 *
 * The refusals matter as much as the write:
 * · no header → nothing happens. An older server never sends one, and a session must
 *   not be disturbed by silence.
 * · the same token back → no write, so listeners are not woken on every request.
 * · a header with NO stored token → ignored. We never had a session to renew, and
 *   accepting a credential we did not ask for is how a shared proxy hands you
 *   somebody else's.
 * · a value that is not a three-part JWT → ignored, keeping the credential that is
 *   currently WORKING. A renewal is an improvement or it is nothing.
 * Returns true only when the stored token actually changed (what a test can read).
 */
let lastRenewalAtS = 0

/**
 * When this page last had a session renewal accepted (seconds, 0 = never).
 *
 * Deliberately an OBSERVATION, not a copy of the server's renewal policy: the cap and the
 * TTL live in auth.py, and a second copy here would drift. What the UI needs to know is
 * narrower and answerable locally — "is this server renewing me at all?" An older build
 * (the one running on :8090 right now) sends no X-Session-Token and leaves this at 0.
 */
export function lastRenewalAt(): number {
  return lastRenewalAtS
}

export function absorbRenewedSession(res: { headers?: { get(name: string): string | null } } | null): boolean {
  let offered: string | null = null
  try {
    offered = res?.headers?.get('X-Session-Token') ?? null
  } catch {
    return false // a Response-like without real headers (a stub, a blob shim) is not an error
  }
  const fresh = (offered ?? '').trim()
  if (!fresh) return false
  const current = (localStorage.getItem(TOKEN_KEY) ?? '').trim()
  if (!current || fresh === current) return false
  if (fresh.split('.').length !== 3) return false
  setAuthToken(fresh)
  lastRenewalAtS = Date.now() / 1000
  return true
}

export async function api<T = any>(path: string, init: RequestInit = {}): Promise<T> {
  const token = authToken()
  const headers: Record<string, string> = { ...(init.headers as Record<string, string>) }
  if (init.body && !headers['Content-Type']) headers['Content-Type'] = 'application/json'
  if (token) headers['Authorization'] = `Bearer ${token}`

  let res: Response
  try {
    res = await fetch(apiUrl(path), { ...init, headers })
  } catch (e) {
    throw new HttpError(0, `cannot reach ${backendLabel()}: ${e instanceof Error ? e.message : e}`)
  }
  absorbRenewedSession(res)
  const text = await res.text()
  let body: any = text
  try { body = text ? JSON.parse(text) : null } catch { /* keep raw text */ }
  if (!res.ok) {
    noteAuthRefusal(res.status)
    const detail = (body && (body.detail ?? body.error)) || text || res.statusText
    // Q99: the server's richest errors arrive as an OBJECT ({error, hint, and the alternatives that
    // exist}) — JSON.stringify put braces and quotes on the screen and buried the answer in the middle
    // of them. detailSentence composes the sentence it was written to be, and falls back to the JSON
    // for a shape it does not recognise, so nothing is ever quietly dropped.
    let message = detailSentence(detail) || text || res.statusText
    // Q79: a 404 on a route this server does not have at all is the server being older than the
    // bundle, not the resource being absent. Only ever ADDS an explanation - the server's own words
    // stay, because when the route does exist they are the truth.
    // The authoritative test is the server's own route list; when /openapi.json cannot be read
    // routeKnown() stays null on purpose, and the catch-all's own words are the fallback evidence.
    // Both live here, in the one place every call passes through — a second mechanism at a single
    // call site would explain one screen and leave the rest reading "HTTP 404".
    if (res.status === 404 && (routeKnown(await liveRoutes(), path) === false
        || unroutedByDetail(body && (body.detail ?? null)))) {
      message = staleRouteMessage(path)
    }
    throw new HttpError(res.status, message, body)
  }
  noteAuthAccepted(path)
  return body as T
}

export const post = <T = any>(path: string, body?: unknown) =>
  api<T>(path, { method: 'POST', body: body === undefined ? '{}' : JSON.stringify(body) })

/**
 * DELETE. Added for Q124 (revoking a passkey) — the first caller that needed the verb, and it goes
 * through api() like everything else so auth, session renewal and the 404 route explanation apply.
 * No body: FastAPI's DELETE routes here take the id in the path.
 */
export const del = <T = any>(path: string) => api<T>(path, { method: 'DELETE' })

/**
 * Authed fetch of a binary endpoint (camera previews), returned as an object
 * URL. An <img src> cannot carry an Authorization header, so the bytes come
 * through fetch and the caller MUST revoke the URL when done with it.
 */
export async function apiBlob(path: string): Promise<string> {
  const token = authToken()
  const headers: Record<string, string> = {}
  if (token) headers['Authorization'] = `Bearer ${token}`
  let res: Response
  try {
    res = await fetch(apiUrl(path), { headers })
  } catch (e) {
    throw new HttpError(0, `cannot reach ${backendLabel()}: ${e instanceof Error ? e.message : e}`)
  }
  absorbRenewedSession(res)
  if (!res.ok) {
    // Q104: a camera preview is a GUARDED request like any other, and on the fleet screen it is often
    // the FIRST thing a rotated token refuses — the tiles are what the operator is looking at. Without
    // this the refusal memory never hears about it, so planRetry keeps reopening and AuthGate's watcher
    // has no evidence to verify.
    noteAuthRefusal(res.status)
    const text = await res.text()
    let detail: unknown = text || res.statusText
    try { detail = JSON.parse(text).detail ?? detail } catch { /* raw text */ }
    // Same rail as api(): a camera preview refused with a structured detail (409 PermissionError,
    // 503 with the driver's words) is read by a person too.
    throw new HttpError(res.status, detailSentence(detail) || text || res.statusText)
  }
  noteAuthAccepted(path)
  return URL.createObjectURL(await res.blob())
}
