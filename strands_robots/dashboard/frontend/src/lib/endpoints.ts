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
  notify()
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
  notify()
}

const listeners = new Set<() => void>()

function notify(): void {
  listeners.forEach(fn => fn())
}

export function onBackendChange(fn: () => void): () => void {
  listeners.add(fn)
  return () => listeners.delete(fn)
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
    if (!res.ok) return null
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
  const text = await res.text()
  let body: any = text
  try { body = text ? JSON.parse(text) : null } catch { /* keep raw text */ }
  if (!res.ok) {
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
  return body as T
}

export const post = <T = any>(path: string, body?: unknown) =>
  api<T>(path, { method: 'POST', body: body === undefined ? '{}' : JSON.stringify(body) })

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
  if (!res.ok) {
    const text = await res.text()
    let detail: unknown = text || res.statusText
    try { detail = JSON.parse(text).detail ?? detail } catch { /* raw text */ }
    // Same rail as api(): a camera preview refused with a structured detail (409 PermissionError,
    // 503 with the driver's words) is read by a person too.
    throw new HttpError(res.status, detailSentence(detail) || text || res.statusText)
  }
  return URL.createObjectURL(await res.blob())
}
