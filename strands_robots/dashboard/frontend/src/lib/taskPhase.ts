/** What phase a task is in — as pure functions. */
import type { Peer } from '../types'

/**
 * What phase a task is in — as pure functions. This lived in useTask's body, where the only
 * way to reach it was to render a card against a live peer.
 */

export type TaskPhase = 'idle' | 'starting' | 'running' | 'stopping' | 'failed' | 'done'

/** Statuses the robot uses for "a task is executing right now". */
const RUNNING = new Set(['running', 'executing'])

/**
 * A simulation peer publishes no `task` block; its rollouts show as
 * `state.robots.<name>.active` (mesh/core `_read_state`). Read as a status: any
 * robot active = running, every robot reported idle = idle, nothing reported =
 * undefined. A child peer `<sim>__<robot>` is asked about its own robot only.
 */
export function simActivityStatus(
  robots: unknown, peerId?: string,
): 'running' | 'idle' | undefined {
  if (!robots || typeof robots !== 'object') return undefined
  const own = peerId?.split('__')[1]
  const entries = Object.entries(robots as Record<string, unknown>)
    .filter(([name]) => !own || name === own)
  const flags = entries.map(([, v]) => (v as any)?.active).filter(f => typeof f === 'boolean')
  if (!flags.length) return undefined
  return flags.some(Boolean) ? 'running' : 'idle'
}

/** The peer's own words about its task, or undefined when it said nothing. */
export function reportedTaskStatus(peer: Pick<Peer, 'state' | 'presence'> & { peer_id?: string }): string | undefined {
  const s = (peer.state as any)?.task?.status ?? (peer.presence as any)?.task_status
  if (typeof s === 'string') return s
  // #4182: a sim peer never says `task.status`; its `robots.<name>.active` is the same fact.
  return simActivityStatus((peer.state as any)?.robots, peer.peer_id)
}

export function isRunningStatus(status: string | undefined): boolean {
  return status !== undefined && RUNNING.has(status)
}

/**
 * Reconcile our optimistic phase with the peer's report. Returns the new phase, or null when
 * there is nothing to change.
 */
export function nextPhase(phase: TaskPhase, reported: string | undefined): TaskPhase | null {
  if (isRunningStatus(reported)) return phase === 'running' ? null : 'running'
  // Affirmative non-running status: the robot says it is not executing, so our optimistic
  // "running" is over.
  if (reported !== undefined && phase === 'running') return 'done'
  return null
}

/**
 * Is this task running / is the UI busy, from the peer's report plus our optimistic phase. The
 * peer's own report WINS: it is the robot telling us what it is doing, we are only guessing.
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
