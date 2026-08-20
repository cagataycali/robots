/**
 * How old are the numbers on a joint strip?
 *
 * The strip renders `state.joints` as bold values and filled bars. Nothing in it
 * was a function of TIME, so a state stream that stops leaves the last frame on
 * screen looking exactly like a robot holding position — the frozen-counter class
 * again, on the one surface an operator reads while their hands are on the arm.
 *
 * The card pairs the strip with TelemetryStrip ("stale 7s") and the status
 * sentence ("joints shown are stale"), so there it is covered. RecordPanel does
 * NOT: it renders the strip alone, while the operator hand-guides the leader and
 * watches these numbers to see the follower tracking. That is the worst possible
 * place for a freeze to look like stillness, so the strip has to carry its own
 * age instead of relying on a neighbour.
 *
 * Thresholds are about what the number is USED for, not about the transport:
 *   < 2s   a 10Hz stream with jitter, or a poll boundary: say nothing.
 *   2-10s  lagging - the value is probably still roughly right, but it is not now.
 *   > 10s  frozen - past the ~15s presence heartbeat window, so this is a dead
 *          stream, not a slow one, and the numbers must stop looking authoritative.
 * A strip that never received a frame is NOT stale; it has nothing to be stale
 * about, and the empty state already explains itself.
 */

export type JointFreshness = 'live' | 'lagging' | 'frozen' | 'unknown'

export interface JointAgeNote {
  level: JointFreshness
  /** null when there is nothing worth saying (fresh, or never received). */
  text: string | null
  /** true when the values must not be presented as the arm's position now. */
  dim: boolean
}

export const LAGGING_MS = 2_000
export const FROZEN_MS = 10_000

function human(ms: number): string {
  const s = ms / 1000
  return s >= 10 ? `${Math.round(s)}s` : `${s.toFixed(1)}s`
}

export function jointAgeNote(ageMs: number | null | undefined): JointAgeNote {
  if (ageMs === null || ageMs === undefined || !Number.isFinite(ageMs)) {
    return { level: 'unknown', text: null, dim: false }
  }
  const age = Math.max(0, ageMs)
  if (age < LAGGING_MS) return { level: 'live', text: null, dim: false }
  if (age < FROZEN_MS) {
    return {
      level: 'lagging',
      text: `⚠ these numbers are ${human(age)} old — the state stream is lagging`,
      dim: false,
    }
  }
  return {
    level: 'frozen',
    // Name the failure AND the wrong conclusion, because the wrong conclusion is
    // the one a hand acts on.
    text: `⚠ frozen ${human(age)} ago — this is the last frame received, not where the arm is now`,
    dim: true,
  }
}
