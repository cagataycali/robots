/**
 * AuthGate - the passkey door in front of the dashboard.
 *
 * Decision on mount (and after every backend/token change, because App remounts):
 *   1. GET /api/auth/status (public) + probe GET /api/fleet (guarded) in parallel.
 *   2. Fleet answered 200  -> open. Local dev, static token, or a valid session -
 *      whatever the server accepted, the UI has no business second-guessing it.
 *   3. Fleet answered 401  -> gate. setup_required decides enroll vs login.
 *
 * The session JWT rides the existing token plumbing (setAuthToken -> localStorage
 * -> Authorization: Bearer on fetches, ?token= on WebSockets), so nothing below
 * this component knows passkeys exist.
 */
import { useEffect, useRef, useState } from 'react'
import { api, setAuthToken, HttpError } from '../lib/endpoints'
import {
  fetchAuthStatus, enroll, webauthnReady, type AuthStatus,
  beginLogin, completeLogin, loginFresh, type PreparedLogin,
} from '../lib/passkey'
import StrandsMark from './StrandsMark'
import { useRegisterSW } from 'virtual:pwa-register/react'

/** Build stamp so a human (or a screenshot) can tell WHICH gate they're on. */
const BUILD = (import.meta as any).env?.VITE_BUILD ?? 'dev'

type Mode = 'checking' | 'open' | 'enroll' | 'login' | 'unreachable'

export default function AuthGate({ children }: { children: React.ReactNode }) {
  const [mode, setMode] = useState<Mode>('checking')
  const [status, setStatus] = useState<AuthStatus | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [label, setLabel] = useState('')
  const [bootstrap, setBootstrap] = useState('')
  const [showToken, setShowToken] = useState(false)
  const [tokenValue, setTokenValue] = useState('')
  // Login challenge fetched AHEAD of the tap: iOS Safari only opens the Face ID
  // sheet while the tap's user-activation is alive, so the click handler must
  // reach credentials.get() without awaiting the network first.
  const prepared = useRef<PreparedLogin | null>(null)

  // The App's update prompt lives BEHIND the gate — a visitor stuck out here
  // (exactly where auth bugs strand them) could otherwise be pinned to a stale
  // bundle by the service worker forever. At the gate nothing is mid-task, so
  // updating immediately is safe: take the new worker and reload.
  const {
    needRefresh: [gateNeedsRefresh],
    updateServiceWorker,
  } = useRegisterSW({ immediate: true })
  useEffect(() => {
    if (gateNeedsRefresh && mode !== 'open' && mode !== 'checking') {
      void updateServiceWorker(true) // activate waiting SW + reload
    }
  }, [gateNeedsRefresh, mode, updateServiceWorker])

  useEffect(() => {
    if (mode !== 'login' || !webauthnReady()) return
    let alive = true
    const arm = () => {
      if (loginFresh(prepared.current)) return
      beginLogin().then(p => { if (alive) prepared.current = p }).catch(() => {})
    }
    arm()
    const t = setInterval(arm, 200_000) // refresh before the 300s server TTL
    window.addEventListener('focus', arm)
    document.addEventListener('visibilitychange', arm)
    return () => {
      alive = false; clearInterval(t)
      window.removeEventListener('focus', arm)
      document.removeEventListener('visibilitychange', arm)
    }
  }, [mode])

  /** Tap handler: synchronous path into the authenticator when armed. */
  function signIn() {
    const p = loginFresh(prepared.current) ? prepared.current : null
    prepared.current = null
    if (p) {
      void run(() => completeLogin(p)) // first await inside = credentials.get()
    } else {
      // Not armed (first paint, stale, or begin failed): fetch then re-arm and
      // ask for one more tap rather than risk a dead sheet on iOS.
      setBusy(true); setError('')
      beginLogin()
        .then(np => { prepared.current = np; setBusy(false); setError('ready — tap sign in again') })
        .catch(e => { setBusy(false); setError(String((e as Error).message ?? e)) })
    }
  }

  useEffect(() => {
    let alive = true
    ;(async () => {
      try {
        const [st, fleet] = await Promise.allSettled([fetchAuthStatus(), api('/api/fleet')])
        if (!alive) return
        if (fleet.status === 'fulfilled') { setMode('open'); return }
        const denied = fleet.reason instanceof HttpError && (fleet.reason.status === 401 || fleet.reason.status === 403)
        if (!denied) { setMode('unreachable'); setError(String((fleet.reason as Error)?.message ?? fleet.reason)); return }
        if (st.status !== 'fulfilled') { setMode('unreachable'); setError('auth status unavailable'); return }
        setStatus(st.value)
        setMode(st.value.setup_required ? 'enroll' : 'login')
      } catch (e) {
        if (alive) { setMode('unreachable'); setError(String((e as Error).message ?? e)) }
      }
    })()
    return () => { alive = false }
  }, [])

  async function run(fn: () => Promise<string>) {
    setBusy(true); setError('')
    try {
      const token = await fn()
      setAuthToken(token) // remounts App via backendKey(); gate re-checks and opens
    } catch (e) {
      const msg = e instanceof HttpError ? (e.body?.detail ?? e.message) : (e as Error).message
      setError(String(msg || 'the passkey ceremony failed'))
      setBusy(false)
    }
  }

  if (mode === 'open') return <>{children}</>

  if (mode === 'checking') {
    return (
      <div className="authgate" role="status" aria-live="polite">
        <div className="authcard"><StrandsMark size={40} /><p className="dim">checking access…</p></div>
      </div>
    )
  }

  const noWebauthn = (mode === 'enroll' || mode === 'login') && !webauthnReady()

  return (
    <div className="authgate">
      <div className="authcard" role="dialog" aria-labelledby="authgate-title">
        <StrandsMark size={40} />
        <h1 id="authgate-title">
          {mode === 'unreachable' ? 'backend unreachable'
            : mode === 'enroll' ? 'create the admin passkey'
            : 'unlock with your passkey'}
        </h1>

        {mode === 'unreachable' && (
          <p className="dim">
            The dashboard API did not answer. {error && <code>{error}</code>}
          </p>
        )}

        {noWebauthn && (
          <p className="authwarn">
            Passkeys need a secure context. Open this page over <code>https://</code> or{' '}
            <code>http://localhost</code> - on a plain LAN address the browser disables WebAuthn.
          </p>
        )}

        {mode === 'enroll' && !noWebauthn && (
          <form onSubmit={e => { e.preventDefault(); void run(() => enroll(label.trim() || 'admin', bootstrap.trim())) }}>
            <p className="dim">
              No passkey is enrolled yet. The first one becomes the admin key and seals the
              dashboard - every later visit signs in with it.
            </p>
            <div className="field">
              <label htmlFor="authgate-label">key label</label>
              <input id="authgate-label" value={label} placeholder="e.g. cagatay-iphone"
                     onChange={e => setLabel(e.target.value)} autoComplete="off" />
            </div>
            {status?.bootstrap_required && (
              <div className="field">
                <label htmlFor="authgate-bootstrap">bootstrap token</label>
                <input id="authgate-bootstrap" type="password" value={bootstrap}
                       placeholder="from the machine running the dashboard"
                       onChange={e => setBootstrap(e.target.value)} autoComplete="off" />
              </div>
            )}
            <button className="btn go" type="submit"
                    disabled={busy || (status?.bootstrap_required && !bootstrap.trim())}>
              {busy ? 'waiting for the authenticator…' : 'create passkey'}
            </button>
          </form>
        )}

        {mode === 'login' && !noWebauthn && (
          <>
            <p className="dim">This dashboard is sealed with a passkey.</p>
            <button className="btn go" onClick={signIn} disabled={busy}>
              {busy ? 'waiting for the authenticator…' : 'sign in'}
            </button>
            {!showToken && (
              <button className="btn linklike" type="button" onClick={() => setShowToken(true)}>
                passkey not working? sign in with an access token
              </button>
            )}
            {showToken && (
              <form onSubmit={e => { e.preventDefault(); if (tokenValue.trim()) setAuthToken(tokenValue.trim()) }}>
                <div className="field">
                  <label htmlFor="authgate-token">access token</label>
                  <input id="authgate-token" type="password" value={tokenValue}
                         placeholder="from the dashboard machine (tiny can mint one)"
                         onChange={e => setTokenValue(e.target.value)} autoComplete="off" />
                </div>
                <p className="dim">
                  The escape hatch when the passkey ceremony fails on this device: paste a
                  session token minted on the machine running the dashboard. Wrong or expired
                  tokens simply land back on this screen.
                </p>
                <button className="btn go" type="submit" disabled={!tokenValue.trim()}>unlock</button>
              </form>
            )}
          </>
        )}

        {error && mode !== 'unreachable' && <p className="autherror" role="alert">{error}</p>}
        <p className="dim" style={{ fontSize: 11, opacity: 0.55, marginTop: 12 }}>build {BUILD}</p>
      </div>
    </div>
  )
}
