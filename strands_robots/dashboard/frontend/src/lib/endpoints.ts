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
    return url.origin
  } catch {
    return ''
  }
}

let cachedBase: string | null = null

export function backendBase(): string {
  if (cachedBase !== null) return cachedBase
  // ?backend=... wins once, then persists: it makes a QR code / bookmark that
  // points a phone straight at a specific robot.
  const params = new URLSearchParams(location.search)
  const fromQuery = params.get('backend')
  const fromToken = params.get('token')
  if (fromToken) localStorage.setItem(TOKEN_KEY, fromToken)
  if (fromQuery !== null) {
    cachedBase = normalize(fromQuery)
    localStorage.setItem(BASE_KEY, cachedBase)
    return cachedBase
  }
  cachedBase = normalize(localStorage.getItem(BASE_KEY) ?? '')
  return cachedBase
}

export function authToken(): string {
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
    throw new HttpError(res.status, typeof detail === 'string' ? detail : JSON.stringify(detail), body)
  }
  return body as T
}

export const post = <T = any>(path: string, body?: unknown) =>
  api<T>(path, { method: 'POST', body: body === undefined ? '{}' : JSON.stringify(body) })
