/**
 * What the dashboard may claim when the STOP ALL request itself fails.
 *
 * The sheet used to answer every failure with "Nothing was sent." That is a
 * claim the browser cannot make. `api()` throws HttpError(0, …) whenever fetch
 * rejects — and fetch rejects for a request that never left the machine AND for
 * one that reached the server, executed, and lost the connection before the
 * answer came back. A 5xx is the same story from the other side: the handler
 * ran, so the per-peer stops and the signed lockout envelope may already be on
 * the wire.
 *
 * Getting this wrong on an e-stop is expensive twice over: the operator is told
 * the fleet is untouched (so they misread the very next symptom — every peer
 * refusing commands — as a new fault), and a "nothing happened" reading invites
 * a calm retry where the truth is "assume the arms are still moving, hit the
 * hardware switch".
 *
 * Only a refusal the server issued BEFORE the handler (auth, missing route,
 * validation, rate limit) is safely "nothing ran".
 */

export interface RequestFailure {
  /** HttpError.status — 0 when fetch itself rejected. */
  status?: number | null
  /** HttpError.message — the server's own words when it answered. */
  message?: string | null
}

export interface FailureVerdict {
  /** 'no' = provably nothing ran. 'unknown' = it may have reached the fleet. */
  delivered: 'no' | 'unknown'
  headline: string
  advice: string
  /** True when pressing again could send a SECOND stop (harmless) — kept
   *  explicit so the retry button's copy never implies the first one failed. */
  retryRepeats: boolean
}

const HARDWARE = 'The arms’ power switch is the only brake that does not go through this page — use it now.'

function reason(f: RequestFailure): string {
  const m = String(f.message ?? '').trim()
  return m || 'no detail'
}

/** Did the server refuse before running any handler code? */
export function refusedBeforeActing(status: number | null | undefined): boolean {
  const s = Number(status ?? 0)
  if (!Number.isFinite(s) || s <= 0) return false // transport failure: unknowable
  if (s >= 500) return false                      // the handler ran and blew up
  return s === 400 || s === 401 || s === 403 || s === 404 || s === 405 || s === 422 || s === 429
}

export function estopFailureVerdict(f: RequestFailure): FailureVerdict {
  if (refusedBeforeActing(f.status)) {
    return {
      delivered: 'no',
      headline: `✗ the dashboard refused to send the stop (${f.status}: ${reason(f)}) — nothing was sent, no robot was told to stop.`,
      advice: `Fix that first (a token in Settings usually), and do not wait for it: ${HARDWARE}`,
      retryRepeats: false,
    }
  }
  const transport = !Number(f.status ?? 0)
  return {
    delivered: 'unknown',
    headline: transport
      ? `⚠ no answer came back (${reason(f)}) — the stop MAY already have reached the fleet, and it may not. This page cannot tell.`
      : `⚠ the server failed mid-stop (${f.status}: ${reason(f)}) — some peers and the fleet lockout MAY already have been signalled.`,
    advice: `Assume the robots are still moving: ${HARDWARE} If the lockout did engage, every peer will refuse commands until you resume with the override code — that is the stop working, not a new fault.`,
    retryRepeats: true,
  }
}

/** Same asymmetry on the way back out: a resume whose answer is lost may have cleared the lockout. */
export function resumeFailureVerdict(f: RequestFailure): { text: string; delivered: 'no' | 'unknown' } {
  if (refusedBeforeActing(f.status)) {
    return {
      text: `✗ rejected (${f.status}: ${reason(f)}) — the lockout is still in place (wrong code? brute-force cooldown?).`,
      delivered: 'no',
    }
  }
  return {
    text: `⚠ no answer (${reason(f)}) — the lockout may or may not have cleared. Check whether a robot accepts a command before resuming again.`,
    delivered: 'unknown',
  }
}
