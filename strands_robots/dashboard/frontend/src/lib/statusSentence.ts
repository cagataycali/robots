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
  /** joints changed over the last ~1s (from the telemetry ring); null = not measured */
  moving: boolean | null
  /**
   * Whether the peer publishes joint positions AT ALL (null = unknown yet).
   * Separates the two reasons `moving` can be null - "still filling the ring"
   * from "this peer will never tell us" - which are the same silence to the ring
   * and very different sentences to a human.
   */
  jointsSeen?: boolean | null
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

  const status = (f.taskStatus ?? '').trim().toLowerCase()
  const running = status === 'running'

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

  // Coming up. The SDK's TaskStatus has SIX values (idle, connecting, running, completed, stopped,
  // error) and this function used to branch on 'running' alone, so every other one fell through to the
  // green "safe to approach" sentence below (Q93). `connecting` is the worst possible moment for that
  // claim: it is the instant BEFORE torque engages.
  if (status === 'connecting') {
    return f.moving === true
      ? {
          severity: 'warn',
          word: 'starting',
          text: 'bringing the hardware up and the arm is ALREADY MOVING — homing or a queued command, keep hands clear',
        }
      : {
          severity: 'active',
          word: 'starting',
          text: 'bringing the hardware up — torque can engage and the arm move without warning, keep hands clear',
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

  // A task that ENDED BADLY is not the same thing as an idle robot (Q93). Neither of these claims
  // safety; both say where the arm is and what would move it.
  if (status === 'error') {
    return {
      severity: 'warn',
      word: 'failed',
      text: 'the task ended in ERROR — nothing is commanding the arm now, but it stopped wherever it '
        + 'got to; read the robot log before approaching',
    }
  }
  if (status === 'stopped') {
    // Deliberately not a warning: an operator pressing stop is normal, and an amber card after every
    // normal stop is alarm fatigue. It simply must not say "safe to approach" about an arm parked
    // mid-task that a resume would move.
    return {
      severity: 'ok',
      word: 'stopped',
      text: 'the task was stopped before finishing — the arm is holding where it stopped, and a resume '
        + 'would move it from there',
    }
  }
  // An unrecognised status is NO EVIDENCE, and no evidence cannot earn the green sentence. This is the
  // future-proofing the old code lacked: 'paused' or any state a newer SDK invents used to render as
  // "idle and still - safe to approach" simply because it was not the string 'running'.
  if (status !== '' && status !== 'idle' && status !== 'completed') {
    return {
      severity: 'warn',
      word: 'unknown',
      text: `the robot reports task status ${quote(status, 24)} — this dashboard does not know that `
        + 'state, so stillness is not confirmed here',
    }
  }

  // THE THIRD LIE DETECTOR, and the one that was missing: "safe to approach" is
  // a claim about the physical world, so it must be EARNED by a motion
  // measurement. `moving === false` is measured stillness. `moving == null` is
  // no measurement at all, and this branch used to treat the two identically -
  // so a peer publishing zero joint positions rendered a green
  // "idle and still - safe to approach" beside a panel reading "no joint data on
  // this peer" (observed on the live dashboard: so101-arm-1, whose power state
  // was in doubt at that moment). Silence is not stillness.
  //
  // `jointsSeen === false` is AUTHORITATIVE here, ahead of `moving`: the ring
  // computes motion from joint positions, so an empty stream yields motion 0 on
  // every sample and used to harden into `moving: false` - a measurement
  // fabricated from nothing. Fixed at the source too (useTelemetry now reports
  // null), and refused again here, because two layers agreeing is what keeps the
  // green sentence honest if either one is edited later.
  if (f.moving == null || f.jointsSeen === false) {
    if (f.jointsSeen === false) {
      return {
        severity: 'warn',
        word: 'idle?',
        text: 'the robot reports idle, but it publishes no joint positions — stillness cannot be '
          + 'confirmed here, so treat the arm as able to move',
      }
    }
    // Transient: the ring needs ~1s of samples. Not a warning (every page load
    // would cry wolf), but it does not get to say "safe" either.
    return {
      severity: 'ok',
      word: 'idle',
      text: 'idle per the robot — motion not measured yet, a second of telemetry decides',
    }
  }

  return {
    severity: 'ok',
    word: 'idle',
    text: 'idle and still — safe to approach',
  }
}

/**
 * The ribbon renders the word as a coloured chip and the sentence beside it, so
 * a sentence that OPENS with that same word reads "IDLE idle and still…".
 * Strips the duplicate lead only when it really is a duplicate — never rewrites
 * the sentence otherwise, and never returns an empty detail.
 */
export function ribbonDetail(line: StatusLine): string {
  const word = line.word.trim().toLowerCase()
  const text = line.text
  if (!word || !text.toLowerCase().startsWith(word)) return text
  // A WORD BOUNDARY IS REQUIRED: 'idle' must not eat the front of 'idling
  // along' and leave 'ing along'. Only a whole leading word is a duplicate.
  const after = text.charAt(word.length)
  if (after && /[A-Za-z0-9]/.test(after)) return text
  const rest = text.slice(word.length).replace(/^[\s\u2014:,-]+/, '')
  return rest.length > 0 ? rest : text
}
