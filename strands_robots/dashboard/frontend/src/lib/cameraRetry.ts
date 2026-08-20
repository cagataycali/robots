/**
 * When a camera socket may try again — and, crucially, when its attempt counter is
 * allowed to forget the failures.
 *
 * MEASURED INCIDENT (BUGS.md Q40): the live dashboard's log held 63,906 `connection
 * open` lines in ten hours — 1.7 sockets per second, nearly all
 * `/ws/camera/so101-arm-1/top`, from a phone on cellular data, for an arm that was not
 * running. CameraTile already had exponential backoff capped at 10s, which would be
 * ~3,600 attempts in that window, not 63,906. The backoff was not broken; it was being
 * RESET. `ws.onopen` set `tries = 0`, and this socket does open: the server accepts the
 * handshake, authenticates it, finds nothing publishing and closes. Every failure
 * therefore looked like a success followed by bad luck, so the delay was 1s forever.
 *
 * The lesson generalises past this file: a connection that COMPLETES A HANDSHAKE has
 * not proved anything. Progress is delivered data, or at least surviving long enough
 * that the next attempt is not obviously futile. So the counter resets on evidence —
 * a frame arrived, or the socket stayed open past `MIN_USEFUL_OPEN_MS` — and an
 * accepted-then-immediately-closed socket is counted as the failure it is.
 */

/** A socket that closes sooner than this proved nothing, whatever its handshake did. */
export const MIN_USEFUL_OPEN_MS = 20_000

/** Backoff ceiling. A camera the operator is waiting for should still recover on its own. */
export const MAX_RETRY_MS = 30_000

export interface SocketOutcome {
  /** attempts made before this one (0 for the first connection) */
  attempt: number
  /** frames this socket delivered */
  frames: number
  /** how long it stayed open, or undefined if it never opened */
  openMs?: number
  /** the close code, if the socket reported one */
  code?: number
}

export interface RetryPlan {
  /** attempt number to carry into the next connection */
  attempt: number
  /** how long to wait, or null when it must not retry at all */
  delayMs: number | null
  /** why — shown in the tile's status line and in tests */
  reason: string
}

/** Exponential with a ceiling: 1s, 2s, 4s, 8s, 16s, 30s… */
export function backoffMs(attempt: number): number {
  return Math.min(MAX_RETRY_MS, 1000 * Math.pow(2, Math.max(0, attempt - 1)))
}

export function planRetry({ attempt, frames, openMs, code }: SocketOutcome): RetryPlan {
  // A refusal is an answer. Hammering a door that said no is not resilience, and 1008
  // is what this server sends when the token is bad — retrying cannot fix it.
  if (code === 1008) {
    return { attempt, delayMs: null, reason: 'the server refused this socket (unauthorized)' }
  }
  // The two kinds of evidence that this endpoint actually works. Only these clear the
  // history, because only these distinguish a blip from a camera that does not exist.
  const proved = frames > 0 || (openMs !== undefined && openMs >= MIN_USEFUL_OPEN_MS)
  if (proved) {
    return {
      attempt: 1,
      delayMs: backoffMs(1),
      reason: frames > 0
        ? 'this socket delivered frames, so the drop is treated as a blip'
        : 'this socket stayed open long enough to count as working',
    }
  }
  const next = attempt + 1
  return {
    attempt: next,
    delayMs: backoffMs(next),
    reason: openMs !== undefined && openMs < MIN_USEFUL_OPEN_MS
      ? 'accepted, then closed with nothing sent — counted as a failure, not a success'
      : 'the socket never opened',
  }
}
