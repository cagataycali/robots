/**
 * WHY a joint strip is empty — because "no joint data on this peer" is the one
 * sentence that is almost never true.
 *
 * MEASURED 2026-08-20 on the live fleet: so101-arm-1 was rendering that sentence
 * while `presence.connected` was true, `presence.hw` was "so_follower" (an arm
 * with six joints, which had been streaming them the day before), and its state
 * frames were arriving 0.3s old — carrying `t` and `task` but no `joints` key at
 * all. The real reason was in its log and nowhere else: a remote e-stop had locked
 * the arm out ten hours earlier, so the state probe published without positions.
 *
 * The peer document already contains enough to separate the cases that need
 * different actions from the operator:
 *   - nothing has arrived yet          -> wait
 *   - state IS arriving, without joints -> the process is alive and the JOINTS are
 *                                          the missing part (lockout, failed bus
 *                                          read, a robot that has none)
 *   - state has gone quiet             -> the peer or the mesh, not the bus
 *
 * What this must NOT do is GUESS the cause. A lockout and a wedged serial bus look
 * identical from here, and a confident wrong diagnosis costs more than an honest
 * "here is what I can see, here is where the answer is" — so it says which of the
 * three situations it is, and points at the log for the rest.
 *
 * SINCE Q80 (2026-08-20) the cause sometimes arrives WITH the peer: the backend reads the child's
 * own log and annotates `joint_problem` when its `hw_joints` probe failed — a held serial port and
 * an uncalibrated board are different faults with opposite remedies, and both used to render as
 * this module's honest shrug. When that verdict is present it is used verbatim, because it is
 * evidence rather than inference; when it is absent NOTHING changes, so a fleet whose backend
 * cannot tell still gets the pointer-at-the-log wording instead of a fabricated reason.
 */

/** State frames older than this are not "current" any more. */
export const STATE_QUIET_S = 10

export interface AbsenceInput {
  /** the peer's state document, if any */
  state?: { t?: number | null; joints?: Record<string, unknown> | null } | null
  presence?: {
    connected?: boolean | null
    hw?: string | null
    action_keys?: string[] | null
  } | null
  /** The backend's own verdict on WHY joints are missing (peer annotation `joint_problem`, Q80). */
  problem?: {
    kind?: string | null
    headline?: string | null
    remedy?: string | null
    detail?: string | null
    /** 'peer' when the robot itself reported the fault (mesh `degraded`), absent for a log-derived verdict. */
    source?: string | null
    /** How many consecutive reads have failed, when the peer reports it. */
    failures?: number | null
    /** How long it has been failing, in seconds, when the peer reports it. */
    for_seconds?: number | null
  } | null
  nowS: number
}

export interface AbsenceNote {
  /** the sentence for the empty strip */
  text: string
  /** 'waiting' = neutral, 'attention' = something is wrong and actionable */
  tone: 'waiting' | 'attention' | 'none'
  /** where the answer actually is, when we cannot know it from here */
  hint: string | null
  /** the raw exception behind a backend verdict, for a title/tooltip — never the whole sentence */
  detail?: string | null
}

/**
 * "for 3.5h" -- how long a fault has lasted, in the same brackets agoText uses.
 *
 * Duration is not decoration here: a probe that failed once and a probe that has been failing since
 * before lunch call for different responses (retry versus go and look at the arm), and it is the
 * difference cagatay hit -- two arms silent for 3.5 hours while their cards said only "no joints".
 * Under 10s nothing is said: a fault that young is as likely to be a transient as a condition, and
 * naming it would invite chasing noise.
 */
export function failingForText(seconds: number | null | undefined): string | null {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 10) return null
  if (seconds < 90) return `for ${Math.round(seconds)}s`
  if (seconds < 5400) return `for ${Math.round(seconds / 60)}m`
  return `for ${(seconds / 3600).toFixed(1)}h`
}

/** Does this peer look like something that HAS joints? */
export function expectsJoints(presence: AbsenceInput['presence']): number | 'yes' | 'unknown' {
  const n = presence?.action_keys?.length
  if (typeof n === 'number' && n > 0) return n
  // `hw` names the arm family ("so_follower", "so_leader"): a peer that declares
  // one is an arm, even before it has told us how many joints it has.
  if (presence?.hw) return 'yes'
  return 'unknown'
}

export function jointAbsence(input: AbsenceInput): AbsenceNote {
  const { state, presence, problem, nowS } = input
  const verdict = problem?.headline ? problem : null
  const expects = expectsJoints(presence)
  const ageS = typeof state?.t === 'number' && state.t > 0 ? nowS - state.t : null
  // A NEGATIVE age is clock skew between two machines, and it counts as arriving:
  // the alternative (my first version) printed "state went quiet -5s ago", which is
  // the kind of sentence that makes an operator distrust every other number on the
  // page. Caught by the test, not by review.
  const stateArriving = ageS !== null && ageS <= STATE_QUIET_S

  // A peer with no state document at all, or one that stopped sending.
  if (state == null || ageS === null) {
    // A verdict outranks the shrug even here (Q89 follow-up). The backend reads it from the CHILD'S
    // LOG, which exists whether or not a single state frame was ever published — so discarding it
    // because no frame arrived throws away the only explanation on offer, in the case where the
    // operator has least to go on. Measured on the live rig: an arm that fails its probe at spawn is
    // exactly the arm least likely to be publishing.
    if (verdict) {
      return {
        text: `no state frames yet — ${verdict.headline}`,
        tone: 'attention',
        hint: verdict.remedy ?? null,
        detail: verdict.detail ?? null,
      }
    }
    if (expects === 'unknown') return { text: 'no joint data on this peer', tone: 'none', hint: null }
    const count = expects === 'yes' ? '' : ` (${expects} joints expected)`
    return { text: `waiting for the first state frame${count}`, tone: 'waiting', hint: null }
  }

  if (!stateArriving) {
    const ago = ageS < 90 ? `${Math.round(ageS)}s` : ageS < 5400 ? `${Math.round(ageS / 60)}m` : `${(ageS / 3600).toFixed(1)}h`
    // The default hint RULES OUT the servo bus ("the peer or the mesh, not the servo bus"), which is a
    // claim — and a claim the child's own log can contradict: an arm whose probe died on "Port is in
    // use!" went quiet BECAUSE of the bus. Where a verdict exists, it is evidence and the guess is not,
    // so the guess must not be printed over it.
    return {
      text: `state went quiet ${ago} ago — no joints since`,
      tone: 'attention',
      hint: verdict?.remedy
        ?? 'the peer or the mesh, not the servo bus: its process may have exited',
      detail: verdict?.detail ?? null,
    }
  }

  // The interesting case, and the one that was reading as "no joint data": the
  // process is alive and publishing, and the JOINTS are what is missing. When the backend read the
  // reason out of the child's log, say IT — this is the one branch where the shrug used to live.
  if (verdict) {
    const lasting = failingForText(verdict.for_seconds)
    // The count and the provenance go in the TOOLTIP, not the sentence: the card must stay one
    // readable line, and the fact that decides what to do (the headline) is already in it.
    const extras: string[] = []
    if (verdict.detail) extras.push(verdict.detail)
    if (typeof verdict.failures === 'number' && verdict.failures > 1) {
      extras.push(`${verdict.failures} consecutive failed reads`)
    }
    // Provenance QUALIFIES the evidence, so it is only added when there IS evidence: a tooltip
    // consisting of nothing but "read from its log" would give an operator no fact to weigh, and an
    // existing contract in this file says a verdict with nothing to show keeps detail null rather
    // than rendering an attribute made of filler.
    if (extras.length > 0) {
      extras.push(verdict.source === 'peer'
        // A peer-reported fault disappears the moment the probe recovers, so its presence means NOW.
        ? 'reported by the robot itself, and it clears when the read works again'
        // A log-derived one clears only if the log later says the probe recovered -- which mesh.core
        // now logs, but a robot running older code never does, so its complaint can outlive the fault.
        : 'read from this robot\'s log, and it clears only when the log records a recovery')
    }
    return {
      text: lasting ? `no joint positions ${lasting} — ${verdict.headline}` : `no joint positions — ${verdict.headline}`,
      tone: 'attention',
      hint: verdict.remedy ?? 'check its log (devices → logs)',
      detail: extras.length > 0 ? extras.join(' \u00b7 ') : null,
    }
  }
  if (expects === 'unknown') {
    return { text: 'this peer publishes state without joint positions', tone: 'none', hint: null }
  }
  const count = expects === 'yes' ? '' : ` — ${expects} expected`
  return {
    text: `state is arriving, but carries no joint positions${count}`,
    tone: 'attention',
    hint: 'the arm is alive and talking; a safety lockout and a failed bus read both look like this — check its log (devices → logs)',
  }
}
