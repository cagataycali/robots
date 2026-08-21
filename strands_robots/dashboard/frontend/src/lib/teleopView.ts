/**
 * U22 slice 1, READ-ONLY: what is this arm's teleop actually doing?
 *
 * The server already computes the verdict (dashboard/teleop_health.py) — states `refusing`,
 * `unrouted`, `silent`, `stopped`, `following` for receivers, plus per-publisher rates — and it exists
 * because of a MEASURED failure: a real SO-101 leader published 176 frames that the follower refused
 * every one of, while /teleop/receive said "started", the receiver said running:true and /api/fleet
 * showed both peers healthy. The truth lived only in the follower's child log. So this rule's job is
 * not to judge; it is to make sure that verdict reaches a human, and that its worst news wins.
 *
 * THE LAW HERE IS THE DIFFERENCE BETWEEN "IDLE" AND "UNASKED": a missing payload returns null and the
 * screen says nothing, because "we did not ask" rendered as "not teleoping" is the same lie the
 * counters told — an arm that IS streaming would read as quiet. Only an answered request with no
 * receivers and no publishers may say the arm is idle.
 */
export type TeleopTone = 'ok' | 'warn' | 'idle'
export interface TeleopView {
  tone: TeleopTone
  headline: string
  detail?: string | null
  /** Present when the server's refusal is the widenable safety envelope: the operator is one consent
   *  away from working teleop, and ConsentSettings already renders this kind. Never widen it here. */
  consentKind?: 'teleop_degree_units'
  /** Is anything on the wire right now — the one fact a stop button should be enabled by. */
  streaming: boolean
}

/** `refusing` outranks everything, exactly as the server's own ordering does. */
const TONE: Record<string, TeleopTone> = {
  refusing: 'warn', unrouted: 'warn', silent: 'warn', stopped: 'idle', following: 'ok',
}

export function teleopView(payload: unknown): TeleopView | null {
  const health = (payload as { health?: unknown } | null | undefined)?.health
  if (!health || typeof health !== 'object') return null            // unasked, unreachable, or an old server
  const h = health as {
    receivers?: Record<string, { state?: string; headline?: string; detail?: string | null; refusal?: unknown }>
    publishers?: Record<string, { state?: string; headline?: string; detail?: string | null }>
    worst?: { state?: string; headline?: string; detail?: string | null; refusal?: unknown } | null
  }
  const receivers = h.receivers ?? {}
  const publishers = h.publishers ?? {}
  const pubs = Object.entries(publishers)
  const live = pubs.filter(([, p]) => p.state === 'publishing')

  const worst = h.worst ?? Object.values(receivers)[0] ?? null
  if (!worst && pubs.length === 0) {
    return { tone: 'idle', headline: 'no teleop on this arm', streaming: false,
      detail: 'it is neither following another arm nor publishing its own joints' }
  }
  if (!worst) {
    // Publishing only: this is a LEADER. Its own headline carries the measured rate, and a leader with
    // no follower is not broken — the pairing is the operator's next step, not a fault.
    const [name, p] = live[0] ?? pubs[0]
    return { tone: live.length ? 'ok' : 'idle', headline: `publishing ${name}: ${p.headline ?? p.state ?? 'unknown'}`,
      detail: p.detail ?? null, streaming: live.length > 0 }
  }
  const view: TeleopView = {
    tone: TONE[worst.state ?? ''] ?? 'warn',
    headline: worst.headline ?? worst.state ?? 'teleop state unknown',
    detail: worst.detail ?? null,
    // A receiver that is following counts as traffic too: frames are being applied to a real arm.
    streaming: live.length > 0 || worst.state === 'following' || worst.state === 'refusing',
  }
  if (worst.refusal) view.consentKind = 'teleop_degree_units'
  return view
}
