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
 * What this must NOT do is name the cause. A lockout and a wedged serial bus look
 * identical from here, and a confident wrong diagnosis costs more than an honest
 * "here is what I can see, here is where the answer is" — so it says which of the
 * three situations it is, and points at the log for the rest.
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
  nowS: number
}

export interface AbsenceNote {
  /** the sentence for the empty strip */
  text: string
  /** 'waiting' = neutral, 'attention' = something is wrong and actionable */
  tone: 'waiting' | 'attention' | 'none'
  /** where the answer actually is, when we cannot know it from here */
  hint: string | null
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
  const { state, presence, nowS } = input
  const expects = expectsJoints(presence)
  const ageS = typeof state?.t === 'number' && state.t > 0 ? nowS - state.t : null
  // A NEGATIVE age is clock skew between two machines, and it counts as arriving:
  // the alternative (my first version) printed "state went quiet -5s ago", which is
  // the kind of sentence that makes an operator distrust every other number on the
  // page. Caught by the test, not by review.
  const stateArriving = ageS !== null && ageS <= STATE_QUIET_S

  // A peer with no state document at all, or one that stopped sending.
  if (state == null || ageS === null) {
    if (expects === 'unknown') return { text: 'no joint data on this peer', tone: 'none', hint: null }
    const count = expects === 'yes' ? '' : ` (${expects} joints expected)`
    return { text: `waiting for the first state frame${count}`, tone: 'waiting', hint: null }
  }

  if (!stateArriving) {
    const ago = ageS < 90 ? `${Math.round(ageS)}s` : ageS < 5400 ? `${Math.round(ageS / 60)}m` : `${(ageS / 3600).toFixed(1)}h`
    return {
      text: `state went quiet ${ago} ago — no joints since`,
      tone: 'attention',
      hint: 'the peer or the mesh, not the servo bus: its process may have exited',
    }
  }

  // The interesting case, and the one that was reading as "no joint data": the
  // process is alive and publishing, and the JOINTS are what is missing.
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
