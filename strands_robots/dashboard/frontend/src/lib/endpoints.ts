/**
 * Where the backend lives, and how we talk to it. The dashboard is a mesh *peer*, not a hub -
 * the API it drives can be on this origin, on a robot across the LAN, or on a box behind a
 * VPN.
 */

import { routeKnown, staleRouteMessage, unroutedByDetail } from './serverAge'
import { detailSentence } from './detailSentence'
import { connectionChange, hostOf, needsConfirm, type ConnectionVerdict } from './connectionChange'
import { sessionVerdict, tokenClaims, tokenExpiry } from './sessionExpiry'

const BASE_KEY = 'strands.backend'
const TOKEN_KEY = 'strands.token'
/** The host the stored token was given for: the only host it is ever sent to. */
const TOKEN_HOST_KEY = 'strands.token.host'

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
    // Only a scheme fetch can actually speak.
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return ''
    return url.origin
  } catch {
    return ''
  }
}

let cachedBase: string | null = null
let absorbedUrl = false
/** `?backend=` from the URL, once — null when the URL said nothing usable. */
let urlBase: string | null = null
/** The question a `?backend=` raised that only the operator can answer; null when none is pending. */
let urlVerdict: ConnectionVerdict | null = null
/** `?token=` from the URL, parked here and NOWHERE else until the backend has vouched for it. */
let offeredToken: string | null = null
/** A `?token=` was present but dropped unseen (it arrived beside a `?backend=` that moves the page). */
let offeredDropped = false

function pageHost(): string {
  try {
    return (location.host || '').toLowerCase()
  } catch {
    return ''
  }
}

/** The host a base means: '' is the origin that served the page. */
function hostOfBase(base: string): string {
  return hostOf(base, pageHost())
}

function storedToken(): string {
  return (localStorage.getItem(TOKEN_KEY) ?? '').trim()
}

/** Record which host the stored token is for; a token with no record is for `base`. */
function bindToken(base: string): void {
  if (!storedToken()) {
    localStorage.removeItem(TOKEN_HOST_KEY)
    return
  }
  localStorage.setItem(TOKEN_HOST_KEY, hostOfBase(base))
}

/**
 * Take the credentials off the URL. A `?token=` is only ever PARKED here: it becomes the sign-in
 * when redeemUrlToken() has asked the backend this page is configured for and been told yes
 * (it used to be written straight into storage on load). A `?backend=` is judged by the rule the
 * Settings drawer applies to a typed address (connectionChange): when the token this browser holds
 * was given for another host, the page dials the new host WITHOUT it and keeps the parameter in the
 * address bar until the operator says yes. A URL is not a more trusting entry point than a field.
 */
function absorbUrl(): void {
  if (absorbedUrl) return
  absorbedUrl = true
  try {
    const params = new URLSearchParams(location.search)
    const fromToken = params.get('token')
    const fromBackend = params.get('backend')
    const stored = normalize(localStorage.getItem(BASE_KEY) ?? '')
    const next = fromBackend === null ? null : normalize(fromBackend)
    // A token already here with no recorded issuer was minted for the backend this browser has
    // been talking to; decide that BEFORE the URL is allowed to move the page.
    if (storedToken() && !(localStorage.getItem(TOKEN_HOST_KEY) ?? '').trim()) bindToken(stored)
    // One link may not choose both the server and the credential: a `?token=` beside a
    // `?backend=` that moves the page is dropped unseen. (The hand-off link the AuthGate
    // advertises names no backend; the page it opens IS the robot.)
    const moves = next !== null && next !== stored
    // Otherwise the token only waits for the backend's answer (redeemUrlToken); nothing is stored here.
    offeredToken = fromToken && !moves ? fromToken.trim() || null : null
    offeredDropped = !!fromToken && moves
    let scrubBackend = fromBackend !== null
    if (next) {
      const token = storedToken()
      const verdict = connectionChange({
        currentBase: stored,
        currentToken: token,
        nextBase: next,
        nextToken: token,
        pageHost: pageHost(),
      })
      const moving = verdict.kind === 'token_follows_host' && needsConfirm(verdict)
      urlBase = next
      if (moving) {
        // The evidence stays visible and nothing is persisted: a reload asks the same question.
        urlVerdict = verdict
        scrubBackend = false
      }
    }
    // Scrub what was absorbed: a ?token= URL outlives its token in history,
    // share sheets and screenshots, and must not be re-sent on reload.
    if (fromToken !== null || scrubBackend) {
      try {
        params.delete('token')
        if (scrubBackend) params.delete('backend')
        const rest = params.toString()
        history.replaceState(null, '', `${location.pathname}${rest ? `?${rest}` : ''}${location.hash || ''}`)
      } catch { /* no history (a test stub): the values are absorbed either way */ }
    }
  } catch {
    urlBase = null // no location (a test, a worker): the stored values are the whole truth
    offeredToken = null
  }
}

export function backendBase(): string {
  absorbUrl()
  if (cachedBase !== null) return cachedBase
  // ?backend=... wins once, then persists, unless the token would have to follow it.
  if (urlBase !== null) {
    cachedBase = urlBase
    if (urlVerdict === null) localStorage.setItem(BASE_KEY, cachedBase)
    return cachedBase
  }
  cachedBase = normalize(localStorage.getItem(BASE_KEY) ?? '')
  return cachedBase
}

/** The question a `?backend=` in the address bar is waiting on, if any. */
export function urlBackendVerdict(): ConnectionVerdict | null {
  absorbUrl()
  return urlVerdict
}

/** The operator's yes: the token is now for the URL's backend, which persists like a typed one. */
export function carryTokenToBackend(): void {
  const base = backendBase()
  urlVerdict = null
  localStorage.setItem(BASE_KEY, base)
  bindToken(base)
  try {
    const params = new URLSearchParams(location.search)
    if (params.has('backend')) {
      params.delete('backend')
      const rest = params.toString()
      history.replaceState(null, '', `${location.pathname}${rest ? `?${rest}` : ''}${location.hash || ''}`)
    }
  } catch { /* no location or history: nothing to scrub */ }
  forgetLiveRoutes()
  notifyAuth()
}

/** The stored token, when it was given for the host the page is talking to; '' otherwise. */
export function authToken(): string {
  absorbUrl()
  const token = storedToken()
  if (!token) return ''
  const issuer = (localStorage.getItem(TOKEN_HOST_KEY) ?? '').trim()
  // A credential for one machine is not handed to another: the request goes out bare and the
  // operator lands on the sign-in for that host instead.
  if (issuer !== hostOfBase(backendBase())) return ''
  return token
}

/** The server puts exactly one kind of token in a link (auth.issue_handoff), and it is short-lived. */
const URL_TOKEN_VIA = 'handoff'

export type UrlTokenOutcome = 'none' | 'adopted' | 'refused'

/**
 * Redeem a `?token=` the page arrived with: ONE probe of the public status route on the backend
 * this page is already configured for, carrying the offered token as its bearer. The token is
 * adopted only when that backend answers `authenticated: true`. Refused without a probe when it
 * is not a hand-off token, has lapsed, or this browser already holds a valid bearer; refused
 * after one BARE probe when the backend already knows this browser (the HttpOnly passkey cookie
 * rides that same-origin fetch and no script can read it). A working session is never silently
 * replaced by a link, whichever kind it is. The AuthGate awaits this before it decides.
 */
export async function redeemUrlToken(): Promise<UrlTokenOutcome> {
  absorbUrl()
  const offered = offeredToken
  offeredToken = null // one attempt, whatever happens
  if (!offered) {
    const dropped = offeredDropped
    offeredDropped = false
    return dropped ? 'refused' : 'none'
  }
  const nowS = Date.now() / 1000
  const claims = tokenClaims(offered)
  const exp = tokenExpiry(offered)
  if (!claims || claims.via !== URL_TOKEN_VIA || exp === null || exp <= nowS) return 'refused'
  const held = sessionVerdict(authToken(), nowS)
  if (held.state === 'valid' || held.state === 'expiring' || held.state === 'opaque') return 'refused'
  try {
    // The primary sign-in is the passkey cookie, which this module cannot see: the server prefers
    // a bearer over the cookie, so a link's hand-off would shadow that session for every api()
    // call. Ask bare first; a yes means someone is already signed in here and the link loses.
    // An answer that cannot be read is treated the same way: without a no there is no adoption.
    const bare = await fetch(apiUrl('/api/auth/status'), { credentials: 'same-origin' })
    if ((await statusSaysAuthenticated(bare)) !== false) return 'refused'
    const res = await fetch(apiUrl('/api/auth/status'), { headers: { Authorization: `Bearer ${offered}` } })
    if ((await statusSaysAuthenticated(res)) === true) {
      setAuthToken(offered)
      return 'adopted'
    }
  } catch {
    // no network, no JSON: the link did not prove anything
  }
  return 'refused'
}

/** What `/api/auth/status` said: true, false, or null when the answer cannot be read (not ok, no JSON, wrong shape). */
async function statusSaysAuthenticated(res: Response): Promise<boolean | null> {
  if (!res.ok) return null
  let body: unknown
  try {
    body = JSON.parse(await res.text())
  } catch {
    return null
  }
  const authenticated = body !== null && typeof body === 'object' ? (body as { authenticated?: unknown }).authenticated : undefined
  return authenticated === true ? true : authenticated === false ? false : null
}


// Auth/backend changes must reach React: localStorage writes emit no event in the
// writing tab, so components subscribe here (App keys ConfigProvider off backendKey()).
const authListeners = new Set<() => void>()
export function subscribeAuth(fn: () => void): () => void {
  authListeners.add(fn)
  return () => { authListeners.delete(fn) }
}
function notifyAuth(): void {
  for (const fn of authListeners) fn()
}

// A passkey session lives in the HttpOnly cookie the ceremony set. The page holds only WHEN it
// lapses and a counter that changes the connection identity, so nothing script-readable is a
// credential. Both are page memory: a reload asks the server again (the cookie still answers).
let cookieSessionExp: number | null = null
let cookieSessionEpoch = 0

/** A ceremony finished on this backend and the cookie is set; `exp` is when it lapses. */
export function noteCookieSession(exp: number | null): void {
  cookieSessionExp = typeof exp === 'number' && Number.isFinite(exp) ? exp : null
  cookieSessionEpoch += 1
  notifyAuth() // backendKey() changed
}

/** When the cookie session this page established lapses (seconds), or null when unknown. */
export function cookieSessionExpiry(): number | null {
  return cookieSessionExp
}

export function setAuthToken(token: string): void {
  const value = token.trim()
  if (value) localStorage.setItem(TOKEN_KEY, value)
  else localStorage.removeItem(TOKEN_KEY)
  bindToken(backendBase()) // a token set now is for the backend the page is talking to now
  notifyAuth()
}

/** Human label for the connection chip. */
export function backendLabel(): string {
  const base = backendBase()
  return base ? base.replace(/^https?:\/\//, '') : `${location.host} (this origin)`
}

/** Identity of the current connection. */
export function backendKey(): string {
  return `${backendBase()}|${authToken() ? 'auth' : cookieSessionEpoch ? `cookie${cookieSessionEpoch}` : 'open'}`
}

export function setBackendBase(raw: string): void {
  absorbUrl()
  cachedBase = normalize(raw)
  urlVerdict = null // a typed address answers the URL's question by replacing it
  if (cachedBase) localStorage.setItem(BASE_KEY, cachedBase)
  else localStorage.removeItem(BASE_KEY)
  // The route list belongs to the server we were talking to.
  forgetLiveRoutes()
  notifyAuth() // backendKey() changed
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
  // A browser cannot set headers on a WebSocket handshake. The server reads the
  // session from the `strands_dash` cookie the login ceremony set and never from
  // the query string (a token there lands in access logs), so nothing is added
  // here: a socket is admitted exactly when the page's cookie is.
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
 * fetch + auth + JSON + *real* errors. `catch {}` around a fetch is how a dashboard ends up
 * showing a robot as idle when the command never landed.
 */
let _liveRoutes: string[] | null = null
let _liveRoutesTried = false
let _liveRoutesAt = 0

export const LIVE_ROUTES_TTL_MS = 60_000

/** The running server's route table (openapi.json), cached with a TTL. */
export async function serverRoutePaths(): Promise<string[] | null> { return liveRoutes() }

async function liveRoutes(): Promise<string[] | null> {
  if (_liveRoutesTried && Date.now() - _liveRoutesAt < LIVE_ROUTES_TTL_MS) return _liveRoutes
  _liveRoutesTried = true
  _liveRoutesAt = Date.now()
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

/** Test seam, and what a backend switch calls to forget what the OLD server routed. */
export function forgetLiveRoutes(): void {
  _liveRoutes = null
  _liveRoutesTried = false
  _liveRoutesAt = 0
}

let _refusedAt: number | null = null

export function noteAuthRefusal(status: number, at: number = Date.now()): void {
  if (status === 401 || status === 403) _refusedAt = at
}

/**
 * The server's own PUBLIC_PATHS (server.py): these answer 200 whether or not the credentials
 * are any good, because the middleware never looks at them.
 */
const PROVES_NOTHING = [
  '/api/health',
  '/api/auth/status',
  '/api/auth/register/',
  '/api/auth/login/',
]

/**
 * Clear it when a GUARDED request succeeds — that is the only answer which proves the
 * credentials work.
 */
export function noteAuthAccepted(path?: string): void {
  if (path !== undefined && PROVES_NOTHING.some(p => path.startsWith(p))) return
  _refusedAt = null
}

/** Has this page been refused recently enough to explain a socket that never opened? */
export function authRefusedRecently(withinMs = 60_000, now: number = Date.now()): boolean {
  return _refusedAt !== null && now - _refusedAt <= withinMs
}

let lastRenewalAtS = 0

/** When this page last had a session renewal accepted (seconds, 0 = never). */
export function lastRenewalAt(): number {
  return lastRenewalAtS
}

/** The one route whose answer may carry a renewed session: the page asked for exactly that. */
export const RENEWAL_PATH = '/api/auth/renew'

/**
 * A renewed session offered in a response header. No route in this repository sends one today
 * (`POST /api/auth/renew` renews by re-setting the cookie), so the channel is held to the one
 * shape a renewal has: a SUCCESSFUL answer to the renewal route this page called, while the page
 * holds a token it is sending to that host, offering the same subject with a later expiry. Any
 * other response, whatever host or status, leaves the sign-in exactly as it was: a header on a
 * 404 from a re-pointed or injected host used to swap the credential silently.
 */
export function absorbRenewedSession(
  res: { ok?: boolean; headers?: { get(name: string): string | null } } | null,
  path: string,
): boolean {
  if (!res || res.ok !== true) return false
  if (path !== RENEWAL_PATH) return false
  let offered: string | null = null
  try {
    offered = res.headers?.get('X-Session-Token') ?? null
  } catch {
    return false // a Response-like without real headers (a stub, a blob shim) is not an error
  }
  const fresh = (offered ?? '').trim()
  if (!fresh) return false
  // The token this page presented to that host, not merely one in storage.
  const current = authToken()
  if (!current || fresh === current) return false
  const was = tokenClaims(current)
  const now = tokenClaims(fresh)
  if (!was || !now) return false
  if (typeof now.sub !== 'string' || now.sub !== was.sub) return false
  const wasExp = tokenExpiry(current)
  const nowExp = tokenExpiry(fresh)
  if (wasExp === null || nowExp === null || nowExp <= wasExp) return false
  if (nowExp <= Date.now() / 1000) return false
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
  const text = await res.text()
  let body: any = text
  try { body = text ? JSON.parse(text) : null } catch { /* keep raw text */ }
  if (!res.ok) {
    noteAuthRefusal(res.status)
    const detail = (body && (body.detail ?? body.error)) || text || res.statusText
    let message = detailSentence(detail) || text || res.statusText
    if (res.status === 404 && (routeKnown(await liveRoutes(), path) === false
        || unroutedByDetail(body && (body.detail ?? null)))) {
      message = staleRouteMessage(path)
    }
    throw new HttpError(res.status, message, body)
  }
  noteAuthAccepted(path)
  // A renewal answers an authenticated request only: a request that went out bare (an
  // unconfirmed ?backend= host, a host the token was not minted for) gets no say over
  // the stored credential, whatever header it answers with.
  if (token) {
    absorbRenewedSession(res, path)
  }
  return body as T
}

export const post = <T = any>(path: string, body?: unknown) =>
  api<T>(path, { method: 'POST', body: body === undefined ? '{}' : JSON.stringify(body) })

/** DELETE. */
export const del = <T = any>(path: string) => api<T>(path, { method: 'DELETE' })

/** Authed fetch of a binary endpoint (camera previews), returned as an object URL. */
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
    noteAuthRefusal(res.status)
    const text = await res.text()
    let detail: unknown = text || res.statusText
    try { detail = JSON.parse(text).detail ?? detail } catch { /* raw text */ }
    // Same rail as api(): a camera preview refused with a structured detail (409 PermissionError,
    // 503 with the driver's words) is read by a person too.
    throw new HttpError(res.status, detailSentence(detail) || text || res.statusText)
  }
  noteAuthAccepted(path)
  if (token) {
    // same rule as api(): a bare request renews nothing
    absorbRenewedSession(res, path)
  }
  return URL.createObjectURL(await res.blob())
}
