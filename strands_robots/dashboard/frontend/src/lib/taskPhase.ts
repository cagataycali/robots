import type { Peer } from '../types'

/**
 * What phase a task is in — as pure functions.
 *
 * This lived in useTask's body, where the only way to reach it was to render a card against a live
 * peer. It decides whether the operator sees a task as still running (and therefore whether the
 * STOP button is on screen), so it is worth a decision table.
 */

export type TaskPhase = 'idle' | 'starting' | 'running' | 'stopping' | 'failed' | 'done'

/** Statuses the robot uses for "a task is executing right now". */
const RUNNING = new Set(['running', 'executing'])

/**
 * The peer's own words about its task, or undefined when it said nothing.
 *
 * The distinction is the whole point of this module: hardware_robot builds `task_status` from
 * `self._task_state.status.value`, so a task that FINISHES reports an affirmative status ("idle",
 * "completed", "error"). Absence means something else entirely — the peer does no tasks at all, or
 * mesh/core's status read raised and was swallowed (core.py wraps it in a bare try/except), or a
 * payload arrived from a path that does not carry the field.
 */
export function reportedTaskStatus(peer: Pick<Peer, 'state' | 'presence'>): string | undefined {
  const s = (peer.state as any)?.task?.status ?? (peer.presence as any)?.task_status
  return typeof s === 'string' ? s : undefined
}

export function isRunningStatus(status: string | undefined): boolean {
  return status !== undefined && RUNNING.has(status)
}

/**
 * Reconcile our optimistic phase with the peer's report. Returns the new phase, or null when there
 * is nothing to change.
 *
 * ABSENCE OF A STATUS IS NOT A COMPLETION (Q87). This used to read `!reportedRunning && phase ===
 * 'running'` → done, and `reportedRunning` is false both when the robot says "idle" and when the
 * robot says NOTHING. So one presence payload that lost its task_status — a swallowed read error, a
 * peer type that never reports one — flipped the UI to "done" while the arm was still executing,
 * which also removes `running` and takes the STOP button off the screen. A completion must be
 * something the robot SAID.
 *
 * Holding the phase when a peer goes quiet is the safe direction: the card still offers stop, and
 * the peer's own staleness badge (meshPeers) is what tells the operator the report is old. The
 * cost of holding is a stale "running" label on a dead peer; the cost of guessing is a hidden stop
 * button on a moving arm.
 */
export function nextPhase(phase: TaskPhase, reported: string | undefined): TaskPhase | null {
  if (isRunningStatus(reported)) return phase === 'running' ? null : 'running'
  // Affirmative non-running status: the robot says it is not executing, so our optimistic
  // "running" is over. Only from 'running' — a 'starting' phase has not been reported yet, and
  // 'stopping'/'failed'/'done' are already settled words.
  if (reported !== undefined && phase === 'running') return 'done'
  return null
}

/**
 * Is this task running / is the UI busy, from the peer's report plus our optimistic phase.
 * The peer's own report WINS: it is the robot telling us what it is doing, we are only guessing.
 */
export function deriveTaskFlags(
  input: { phase: TaskPhase; reported: string | undefined; twinBusy: boolean },
): { running: boolean; busy: boolean } {
  const { phase, reported, twinBusy } = input
  return {
    running: isRunningStatus(reported) || phase === 'starting' || phase === 'running',
    busy: phase === 'starting' || phase === 'stopping' || twinBusy,
  }
}
