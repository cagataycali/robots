/**
 * Where the backend lives, and how we talk to it. The dashboard is a mesh *peer*, not a hub -
 * the API it drives can be on this origin, on a robot across the LAN, or on a box behind a
 * VPN.
 */

import { routeKnown, staleRouteMessage, unroutedByDetail } from './serverAge'
import { detailSentence } from './detailSentence'
import { connectionChange, hostOf, needsConfirm, type ConnectionVerdict } from './connectionChange'
import { tokenClaims, tokenExpiry } from './sessionExpiry'

const BASE_KEY = 'strands.backend'
/** Where an older build kept a bearer. Never read: a copy found there is removed on load. */
const LEGACY_TOKEN_KEYS = ['strands.token', 'strands.token.host']

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
/** `?handoff=` from the URL: a one-time code, parked here until redeemUrlHandoff() spends it. */
let offeredCode: string | null = null
/** A sign-in rode the URL and was dropped unseen (a `?token=`, or a code beside a moving `?backend=`). */
let offeredDropped = false

/**
 * A bearer the operator typed (a static access token), in page memory only: a copy in storage
 * outlived the tab and any script in the origin could read it back. Gone on reload, on expiry and
 * after any sign-in ceremony (noteCookieSession); the passkey session itself is the HttpOnly cookie.
 */
let heldToken = ''
/** The host the held token was given for: the only host it is ever sent to. */
let heldTokenHost = ''
/** Bumped by every setAuthToken, so a new token remounts the app like a new backend does. */
let tokenEpoch = 0

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

/** The held bearer, or '' once it has lapsed (an opaque static token has no expiry to lapse). */
function storedToken(): string {
  const exp = heldToken ? tokenExpiry(heldToken) : null
  if (exp !== null && exp <= Date.now() / 1000) {
    heldToken = ''
    heldTokenHost = ''
  }
  return heldToken
}

/** Record which host the held token is for. */
function bindToken(base: string): void {
  heldTokenHost = storedToken() ? hostOfBase(base) : ''
}

function scrubParams(names: string[]): void {
  try {
    const params = new URLSearchParams(location.search)
    if (!names.some(n => params.has(n))) return
    for (const n of names) params.delete(n)
    const rest = params.toString()
    history.replaceState(null, '', `${location.pathname}${rest ? `?${rest}` : ''}${location.hash || ''}`)
  } catch { /* no location or history (a test stub): the values are absorbed either way */ }
}

/**
 * Take what the URL offers off it. A `?token=` is never adopted: the server mints no bearer for a
 * link, so one in the address bar is scrubbed and dropped. A `?handoff=` code is only PARKED here;
 * redeemUrlHandoff() spends it against this page's own backend, which answers with a cookie. A
 * `?backend=` is judged by the rule the Settings drawer applies to a typed address
 * (connectionChange): any move to another host, or to the same host over clear text, waits for the
 * operator. Until the yes the page keeps talking to the backend it had, nothing is persisted and the
 * parameter stays in the address bar. A URL is not a more trusting entry point than a field.
 */
function absorbUrl(): void {
  if (absorbedUrl) return
  absorbedUrl = true
  try {
    // A bearer an older build left in storage is a copy any script could read: drop it.
    for (const k of LEGACY_TOKEN_KEYS) localStorage.removeItem(k)
    const params = new URLSearchParams(location.search)
    const fromToken = params.get('token')
    const fromCode = params.get('handoff')
    const fromBackend = params.get('backend')
    const stored = normalize(localStorage.getItem(BASE_KEY) ?? '')
    const next = fromBackend === null ? null : normalize(fromBackend)
    // One link may not choose both the server and the sign-in: a code beside a `?backend=` that
    // moves the page is dropped unseen. (The handoff link names no backend; the page it opens IS
    // the robot.)
    const moves = next !== null && next !== stored
    offeredCode = fromCode && !moves ? fromCode.trim() || null : null
    offeredDropped = fromToken !== null || (!!fromCode && moves)
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
      urlBase = next
      if (needsConfirm(verdict)) {
        // The evidence stays visible and nothing is persisted: a reload asks the same question.
        urlVerdict = verdict
        scrubBackend = false
      }
    }
    // Scrub what was absorbed: a sign-in in a URL outlives it in history, share sheets and
    // screenshots, and must not be re-sent on reload.
    scrubParams(['token', 'handoff', ...(scrubBackend ? ['backend'] : [])])
  } catch {
    urlBase = null // no location (a test, a worker): the stored values are the whole truth
    offeredCode = null
  }
}

export function backendBase(): string {
  absorbUrl()
  if (cachedBase !== null) return cachedBase
  // ?backend=... wins once and persists, but only when it moves nothing the operator must judge;
  // a pending one is not dialled at all until acceptUrlBackend().
  if (urlBase !== null && urlVerdict === null) {
    cachedBase = urlBase
    localStorage.setItem(BASE_KEY, cachedBase)
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

/** The operator's yes: the page now talks to the URL's backend, which persists like a typed one. */
export function acceptUrlBackend(): void {
  absorbUrl()
  if (urlBase === null) return
  urlVerdict = null
  cachedBase = urlBase
  localStorage.setItem(BASE_KEY, urlBase)
  bindToken(urlBase)
  scrubParams(['backend'])
  forgetLiveRoutes()
  notifyAuth()
}

/** The operator's no: the URL's backend is forgotten and the page stays where it was. */
export function declineUrlBackend(): void {
  absorbUrl()
  urlBase = null
  urlVerdict = null
  scrubParams(['backend'])
  notifyAuth()
}

/**
 * The persistent notice while the page talks to a backend other than the origin that served it,
 * naming that backend; null when they are the same. A re-pointed page looks exactly like the real
 * one, so the only tell is the one this sentence gives.
 */
export function foreignBackendNotice(): string | null {
  const host = hostOfBase(backendBase())
  const page = pageHost()
  if (!host || host === page) return null
  return `This page is talking to ${host}, not to ${page || 'the address that served it'}. Sign-ins and commands go to ${host}.`
}

/** The stored token, when it was given for the host the page is talking to; '' otherwise. */
export function authToken(): string {
  absorbUrl()
  const token = storedToken()
  if (!token) return ''
  const base = backendBase()
  // While a `?backend=` is waiting on the operator the dial goes out bare whatever the host: the
  // binding below is by host, and an https->http downgrade keeps the host.
  if (urlVerdict !== null) return ''
  const issuer = heldTokenHost
  // A credential for one machine is not handed to another: the request goes out bare and the
  // operator lands on the sign-in for that host instead.
  if (issuer !== hostOfBase(base)) return ''
  return token
}

export type UrlTokenOutcome = 'none' | 'adopted' | 'refused'

/**
 * Redeem a `?handoff=` code the page arrived with, against the backend this page is configured for.
 * A BARE status probe first: when the backend already knows this browser (the HttpOnly passkey
 * cookie rides that same-origin fetch and no script can read it) the link loses, so a working
 * session is never silently replaced by a link. Otherwise the code is posted to the redeem route
 * once; the session arrives as the cookie and the page keeps only its expiry. A `?token=` the page
 * arrived with is 'refused' without a request: no link carries a bearer any more. The AuthGate
 * awaits this before it decides.
 */
export async function redeemUrlHandoff(): Promise<UrlTokenOutcome> {
  absorbUrl()
  const code = offeredCode
  offeredCode = null // one attempt, whatever happens
  if (!code) {
    const dropped = offeredDropped
    offeredDropped = false
    return dropped ? 'refused' : 'none'
  }
  try {
    const bare = await fetch(apiUrl('/api/auth/status'), { credentials: 'same-origin' })
    if ((await statusSaysAuthenticated(bare)) !== false) return 'refused'
    const res = await fetch(apiUrl('/api/auth/handoff/redeem'), {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    })
    if (res.ok) {
      let exp: number | null = null
      try {
        const body = JSON.parse(await res.text())
        exp = body && typeof body.exp === 'number' ? body.exp : null
      } catch { /* the cookie is set either way; the expiry is only for the warning */ }
      noteCookieSession(exp)
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


// Auth/backend changes must reach React: page-memory and localStorage writes emit no event in
// the writing tab, so components subscribe here (App keys ConfigProvider off backendKey()).
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
  // A ceremony replaces whatever bearer was typed before it: the cookie is the sign-in now.
  heldToken = ''
  heldTokenHost = ''
  cookieSessionExp = typeof exp === 'number' && Number.isFinite(exp) ? exp : null
  cookieSessionEpoch += 1
  notifyAuth() // backendKey() changed
}

/** When the cookie session this page established lapses (seconds), or null when unknown. */
export function cookieSessionExpiry(): number | null {
  return cookieSessionExp
}

/** Hold a typed bearer in page memory, bound to the backend the page is talking to now. */
export function setAuthToken(token: string): void {
  heldToken = token.trim()
  bindToken(backendBase())
  tokenEpoch += 1
  notifyAuth()
}

/** Human label for the connection chip. */
export function backendLabel(): string {
  const base = backendBase()
  return base ? base.replace(/^https?:\/\//, '') : `${location.host} (this origin)`
}

/** Identity of the current connection. */
export function backendKey(): string {
  return `${backendBase()}|${authToken() ? `auth${tokenEpoch}` : cookieSessionEpoch ? `cookie${cookieSessionEpoch}` : 'open'}`
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
