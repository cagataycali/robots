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
  /** Q150: peer ids of children this peer HOSTS (from armHosts). A host process publishes no joints
   *  BY DESIGN, so silence must not be read to it as "an arm that might move". */
  hostsChildren?: string[] | null
  /** seconds since the last state-topic sample (null = no samples yet) */
  stateAgeS: number | null
  /**
   * The e-stop lockout as the server understands it (Q43's peer.lockout.state): 'locked', 'clear',
   * 'unknown' or absent. Only 'locked' changes the sentence — see the branch below for why 'unknown'
   * deliberately does not.
   */
  lockout?: string | null
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

  // AN E-STOP LOCKOUT IS THE REASON THE ARM IS STILL (Q95), so the sentence has to say it. Q43 put a
  // loud "e-stop locked" badge on the card but left this function blind to the field, and the two
  // widgets then contradicted each other ON THE SAME CARD: a locked arm read "idle and still — safe to
  // approach" beside a red lockout badge. Joining widgets so the operator does not have to is the whole
  // reason this function exists.
  //
  // Only 'locked' speaks here. 'unknown' is the COMMON case — the mesh does not advertise lockout state,
  // so most peers report it — and letting doubt suppress the green sentence would gut it fleet-wide;
  // the dashed "lockout unknown" badge already carries that doubt at the right volume.
  if ((f.lockout ?? '').trim().toLowerCase() === 'locked') {
    if (f.moving === true) {
      // A lockout means commands are refused. Joints moving anyway means the lockout is not holding,
      // or something outside the mesh is driving the arm. That is the worst state on this card.
      return {
        severity: 'danger',
        word: 'locked?!',
        text: 'an e-stop lockout is in place but the joints are MOVING — the lockout is not holding, or '
          + 'something outside the mesh is driving the arm; keep hands clear',
      }
    }
    return {
      severity: 'warn',
      word: 'locked',
      text: 'e-stop lockout — commands are refused and the arm is holding where the stop caught it; '
        + 'clearing the lockout is what makes it live again',
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
    if (f.jointsSeen === false && f.hostsChildren && f.hostsChildren.length) {
      // A PROCESS IS NOT AN ARM (armHosts' law, which until now only the record screen knew). The
      // simulator parent reports zero joints while its child publishes six, so the warn sentence below
      // accused the one peer whose silence is correct — and every false warning spends the credibility
      // of the true one beside it, which is the whole reason the mute arms are worth flagging at all.
      const kids = f.hostsChildren
      return {
        severity: 'ok',
        word: 'process',
        text: kids.length === 1
          ? `hosts ${kids[0]} — this is the process, not an arm; the joints are on that card`
          : `hosts ${kids.length} robots (${kids.join(', ')}) — this is the process, not an arm`,
      }
    }
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


/**
 * Q151: build statusSentence's fields from a peer + its telemetry, in ONE place.
 *
 * The card had this mapping inline, and the DETAIL STAGE — the full-screen view an operator has open
 * while walking up to the arm — had no status sentence at all. So the surface you read from two feet
 * away said nothing, while the card behind it said "idle and still — safe to approach" or "treat the
 * arm as able to move". Two screens rendering the same judgement from two copies of the mapping is
 * how they drift; one builder is why they cannot.
 */
export function peerStatusFields(
  peer: { last_seen?: number | null; stale?: boolean | null; state?: any; lockout?: any; presence?: any },
  telemetry: { moving?: boolean | null; jointsSeen?: boolean | null; stateAgeS?: number | null },
  hostsChildren?: string[] | null,
): StatusFacts {
  const p = peer.presence
  return {
    stale: !!peer.stale,
    lastSeenAgoS: peer.last_seen ? Date.now() / 1000 - peer.last_seen : null,
    hwConnected: p?.connected ?? null,
    taskStatus: peer.state?.task?.status ?? p?.task_status ?? null,
    instruction: peer.state?.task?.instruction || p?.instruction || null,
    taskDurationS: peer.state?.task?.duration ?? null,
    moving: telemetry.moving ?? null,
    jointsSeen: telemetry.jointsSeen ?? null,
    stateAgeS: telemetry.stateAgeS ?? null,
    lockout: peer.lockout?.state ?? null,
    hostsChildren: hostsChildren ?? null,
  }
}
