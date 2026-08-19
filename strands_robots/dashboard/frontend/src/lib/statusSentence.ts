/**
 * The 5-second answer: one plain-language sentence per robot card that says
 * what the arm is DOING and whether it is safe to approach.
 *
 * Today that answer is scattered across a colored dot, an "hw off" badge, a
 * Hz figure and a motion sparkline - the operator has to join four widgets
 * to decide if they can reach over the desk. This module joins them once,
 * in code, and is deliberately a pure function of observable facts so the
 * composition rules are testable without React.
 *
 * Two of the states are LIE DETECTORS, and they are the whole point:
 * - "running but still": a policy claims to be executing while the joints
 *   have not moved - the task wedged, or the policy is producing no-ops.
 * - "moving with no task": the joints are changing while the peer reports
 *   idle - a teleop leader, another client, or a runaway is commanding the
 *   arm. That is exactly when a hand should NOT be near the workspace,
 *   and exactly the state a task-status widget alone would render as calm.
 */

export type Severity = 'ok' | 'active' | 'warn' | 'danger'

export interface StatusFacts {
  /** peer marked stale by the server (no heartbeat ~15s) */
  stale: boolean
  /** seconds since last heartbeat, when known */
  lastSeenAgoS: number | null
  /** presence.connected - hardware attached (null = not reported) */
  hwConnected: boolean | null
  /** task.status from the state topic, e.g. 'idle' | 'running' */
  taskStatus: string | null
  /** task.instruction when one is set */
  instruction: string | null
  /** seconds the current task has been going, when known */
  taskDurationS: number | null
  /** joints changed over the last ~1s (from the telemetry ring) */
  moving: boolean | null
  /** seconds since the last state-topic sample (null = no samples yet) */
  stateAgeS: number | null
}

export interface StatusLine {
  severity: Severity
  /** the sentence itself */
  text: string
  /** short word for the leading glyph/chip */
  word: string
}

const quote = (s: string, max = 44) =>
  `\u201C${s.length > max ? s.slice(0, max - 1) + '\u2026' : s}\u201D`

export function statusSentence(f: StatusFacts): StatusLine {
  // Dead peer first: nothing else in the snapshot can be trusted.
  if (f.stale) {
    const ago = f.lastSeenAgoS != null ? ` for ${Math.round(f.lastSeenAgoS)}s` : ''
    return {
      severity: 'danger',
      word: 'offline',
      text: `no heartbeat${ago} — state unknown, treat the arm as unpredictable`,
    }
  }

  // Peer alive but its state stream froze: the joints below are a photograph.
  if (f.stateAgeS != null && f.stateAgeS > 5) {
    return {
      severity: 'warn',
      word: 'frozen',
      text: `peer is alive but its state stream stopped ${Math.round(f.stateAgeS)}s ago — joints shown are stale`,
    }
  }

  if (f.hwConnected === false) {
    return {
      severity: 'warn',
      word: 'no hw',
      text: 'hardware not connected — the arm is unplugged or unpowered, nothing can move',
    }
  }

  const running = (f.taskStatus ?? '').toLowerCase() === 'running'

  if (running) {
    const what = f.instruction ? ` ${quote(f.instruction)}` : ''
    const since = f.taskDurationS != null && f.taskDurationS >= 1
      ? `, ${Math.round(f.taskDurationS)}s in` : ''
    if (f.moving === false) {
      return {
        severity: 'warn',
        word: 'wedged?',
        text: `policy${what} says running but the arm is not moving${since} — wedged, or producing no-ops`,
      }
    }
    return {
      severity: 'active',
      word: 'running',
      text: `running${what}${since} — arm is under policy control, keep hands clear`,
    }
  }

  // Idle per the task state - but is it actually still?
  if (f.moving === true) {
    return {
      severity: 'warn',
      word: 'moving',
      text: 'arm is MOVING with no task — teleop or another client is commanding it, keep hands clear',
    }
  }

  return {
    severity: 'ok',
    word: 'idle',
    text: 'idle and still — safe to approach',
  }
}
