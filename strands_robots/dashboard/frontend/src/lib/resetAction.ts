/**
 * The Reset button on a robot card: when it may be pressed, and what its answer says.
 *
 * Reset returns every joint to the home pose at once. The rules are the card's, not the
 * server's: the server gates a real arm (POST /api/robots/{id}/reset needs `confirmed: true`,
 * routes_mesh.reset_robot) and refuses a peer with a task in flight (409). The button repeats
 * the second rule before the click so the operator is not told "stop it first" after pressing,
 * and it never repeats the first: the confirm sheet is where a human says yes.
 */
import type { Outcome } from './taskResponse'

export interface ResetInputs {
  /** a task is running on this peer (lib/taskPhase deriveTaskFlags) */
  running: boolean
  /** a request to this peer is in flight */
  busy: boolean
  /** no heartbeat: the card is stale */
  offline: boolean
}

export interface ResetVerdict {
  enabled: boolean
  /** the button's title: what a press does, or why it cannot be pressed */
  title: string
}

export function resetVerdict(i: ResetInputs): ResetVerdict {
  if (i.offline) return { enabled: false, title: 'No heartbeat from this robot: a reset would only wait out its timeout' }
  if (i.running) return { enabled: false, title: 'A task is running: stop it first (■), then reset' }
  if (i.busy) return { enabled: false, title: 'Waiting for this robot to answer' }
  return { enabled: true, title: 'Reset: return every joint to its home pose' }
}

export interface ResetResponse {
  ok?: boolean
  routed_to?: string | null
  result?: { error?: string; status?: string; content?: { text?: string }[]; [k: string]: any } | null
  error?: string
  [k: string]: any
}

/** The server's answer as the one line the card shows. A refusal is the peer's own sentence. */
export function interpretReset(res: ResetResponse | null | undefined): Outcome {
  if (!res || typeof res !== 'object') return { ok: false, text: 'reset: no answer', ambiguous: true }
  if (res.ok === true) {
    const via = res.routed_to ? ` (via ${res.routed_to})` : ''
    return { ok: true, text: `reset to home pose${via}` }
  }
  const r = res.result
  const sentence = (r && typeof r === 'object' && (r.error || r.content?.find(c => c?.text)?.text)) || res.error
  if (typeof sentence === 'string' && sentence.trim()) return { ok: false, text: `reset refused: ${sentence.trim()}` }
  return { ok: false, text: 'reset: the robot did not confirm', ambiguous: true }
}
