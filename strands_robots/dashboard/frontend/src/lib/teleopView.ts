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

/**
 * U22 slice 2: what may be claimed AFTER a stop was sent.
 *
 * "Stopped!" is a claim about a real arm, and the whole reason teleop_health.py exists is that the
 * optimistic version of this sentence was already believed once: /teleop/receive answered "Teleop
 * receive started" for a stream in which every single frame was refused. So the verdict after a stop
 * comes from ASKING AGAIN, never from the fact that the request returned 200 — and a re-ask that
 * still shows frames on the wire is a FAILURE, however cleanly the POST succeeded.
 */
export function stopVerdict(after: TeleopView | null): { ok: boolean; line: string } {
  if (!after) {
    return { ok: false, line: 'stop was sent, but the arm did not answer when asked again — nothing confirms it landed' }
  }
  if (after.streaming) {
    return { ok: false, line: `stop was sent, but frames are STILL on the wire: ${after.headline}` }
  }
  return { ok: true, line: `teleop stopped — ${after.headline}` }
}

/**
 * U22 slice 3b: what may be claimed after a START was sent.
 *
 * The same law as stopVerdict, in the more dangerous direction — and with the one outcome this fleet has
 * actually produced. When a real SO-101 led a follower, /teleop/receive answered "Teleop receive started"
 * and 176 frames were published while the follower applied NONE of them: degrees into a radian envelope.
 * A stream that is refusing is NOT a working teleop session, and the operator must read that on the
 * screen the moment it happens, next to the grant that widens the bound — not in a child log later.
 */
export function startVerdict(after: TeleopView | null): { ok: boolean; line: string } {
  if (!after) {
    return { ok: false, line: 'start was sent, but the arm did not answer when asked again — nothing confirms frames are flowing' }
  }
  if (after.consentKind) {
    return { ok: false, line: `started, but every frame is being REFUSED: ${after.headline} — the bound is widened at settings › consent › ${after.consentKind}, deliberately and by you` }
  }
  if (after.streaming) return { ok: true, line: `teleop live — ${after.headline}` }
  return { ok: false, line: 'start was sent, but nothing is on the wire yet — a follower can take up to 45s to declare its subscriber, so ask again before assuming it failed' }
}
