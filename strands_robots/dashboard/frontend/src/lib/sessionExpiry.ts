/**
 * Whether the sign-in this page is holding is still valid — read from the token itself.
 *
 * MEASURED INCIDENT (BUGS.md Q88, 2026-08-21): the live dashboard's log ended in an unbroken run of
 *
 *     "WebSocket /ws/camera/so101-arm-1/top?token=eyJ..." 403
 *
 * from one cellular client — 100,506 camera-socket lines in total, and the last 19.3 HOURS of them
 * were 403s, because the JWT in that tab expired (`exp` 1787209480) while the tab stayed open. The
 * phone kept reopening two tiles forever against a door that had already said no, and the screen
 * said nothing about it: `AuthGate` decides login-vs-open ONCE on mount, so a session that dies
 * mid-session leaves a dashboard that looks alive and is entirely deaf.
 *
 * The retry rules could not see it either. Q40's rule ("a completed handshake proves nothing") and
 * Q51's churn floor both reason about sockets that OPEN; a 403 handshake is refused before the
 * socket exists, which the browser reports as an ordinary failed connection — indistinguishable
 * from a camera that is not publishing. `planRetry` already stops dead on close code 1008, but a
 * refused HANDSHAKE never gets to send a close code.
 *
 * The evidence was in the page's own pocket the whole time: our token is a JWT, and its `exp` is
 * readable without any network call. Reading it is NOT trusting it — the server still verifies
 * every request, and a client that decodes its own token cannot grant itself anything. It is used
 * for exactly two honest things: stop hammering a door that will keep saying no, and tell the
 * operator the true reason ("sign in again") instead of a camera-shaped shrug.
 *
 * DELIBERATELY SILENT in two cases, because a false "your session expired" is worse than no
 * sentence at all:
 *   - no token: this is a LAN dashboard with auth off, which is a supported way to run;
 *   - a token that is not a decodable JWT with a numeric `exp`: the bootstrap/`--auth-token`
 *     credential is an opaque string that never expires, and audits drive the page with it.
 */

/** Warn this long before the sign-in actually lapses. */
export const EXPIRING_SOON_S = 300

export type SessionState = 'none' | 'opaque' | 'valid' | 'expiring' | 'expired'

export interface SessionVerdict {
  state: SessionState
  /** seconds until it lapses (negative once it has), or null when the token cannot say */
  expiresInS: number | null
  /** the sentence for the operator, or null when there is nothing honest to say */
  text: string | null
  /** true only when retrying cannot possibly succeed until the operator acts */
  refusesUntilSignIn: boolean
}

/** base64url → string, without throwing on anything a caller might hold. */
function decodeSegment(seg: string): string | null {
  try {
    const norm = seg.replace(/-/g, '+').replace(/_/g, '/')
    const pad = '='.repeat((4 - (norm.length % 4)) % 4)
    const bin = atob(norm + pad)
    // A JWT payload is UTF-8; escape/decodeURIComponent is the dependency-free way back.
    return decodeURIComponent(Array.from(bin, c =>
      '%' + c.charCodeAt(0).toString(16).padStart(2, '0')).join(''))
  } catch {
    return null
  }
}

/** The `exp` claim in seconds, or null when this token does not carry one. */
export function tokenExpiry(token: string | null | undefined): number | null {
  const raw = (token ?? '').trim()
  if (!raw) return null
  const parts = raw.split('.')
  if (parts.length !== 3) return null
  const json = decodeSegment(parts[1])
  if (!json) return null
  try {
    const claims = JSON.parse(json) as { exp?: unknown }
    const exp = claims?.exp
    return typeof exp === 'number' && Number.isFinite(exp) ? exp : null
  } catch {
    return null
  }
}

/** "19.3 hours", "4 minutes", "40 seconds" — for a sentence, not for arithmetic. */
export function humaniseSeconds(s: number): string {
  const abs = Math.abs(s)
  if (abs < 90) return `${Math.round(abs)} seconds`
  if (abs < 5400) return `${Math.round(abs / 60)} minutes`
  const hours = abs / 3600
  // One decimal while the number is small ("19.3 hours" is real information), but never a
  // decimal POINT ZERO: "expired 4.0 hours ago" reads like an instrument, not like a sentence.
  const shown = hours < 10 ? Number(hours.toFixed(1)) : Math.round(hours)
  return `${shown} hour${shown === 1 ? '' : 's'}`
}

/**
 * What to do about the sign-in this page is holding.
 *
 * Args:
 *   token: the stored credential (`getAuthToken()`), whatever shape it is in.
 *   nowS: current time in SECONDS (JWT units), so tests need no clock.
 */
export function sessionVerdict(token: string | null | undefined, nowS: number): SessionVerdict {
  const raw = (token ?? '').trim()
  if (!raw) {
    return { state: 'none', expiresInS: null, text: null, refusesUntilSignIn: false }
  }
  const exp = tokenExpiry(raw)
  if (exp === null) {
    // An opaque credential. It cannot lapse on its own, so this rule must not speculate about it.
    return { state: 'opaque', expiresInS: null, text: null, refusesUntilSignIn: false }
  }
  const left = exp - nowS
  if (left <= 0) {
    return {
      state: 'expired',
      expiresInS: left,
      // The two facts the operator needs: it is not the robot's fault, and one tap fixes it.
      text: `this sign-in expired ${humaniseSeconds(left)} ago — sign in again to see cameras and `
        + 'control the fleet. Nothing is wrong with the robots; the page is being refused.',
      refusesUntilSignIn: true,
    }
  }
  if (left <= EXPIRING_SOON_S) {
    return {
      state: 'expiring',
      expiresInS: left,
      text: `this sign-in lapses in ${humaniseSeconds(left)} — sign in again before starting a `
        + 'recording, or it will be refused part-way through.',
      refusesUntilSignIn: false,
    }
  }
  return { state: 'valid', expiresInS: left, text: null, refusesUntilSignIn: false }
}
