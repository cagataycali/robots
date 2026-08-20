/**
 * How many episodes the operator actually asked for — and whether we understood them.
 *
 * The record form used `Math.max(1, Number(raw) || 20)`, which is three silent corrections in one
 * expression: "3o" (a typo for 3) became TWENTY, "0" became 1, "-5" became 1, and the session
 * opened without a word. The operator then watches a counter read "0 / 20" while believing they
 * asked for three, and a teleop recording session is not a cheap thing to restart — the arms have
 * left the fleet and the follower is energised.
 *
 * So: parse, and say what we made of it. Nothing here corrects the number the operator can see;
 * the caller either uses `value` or refuses with `problem`.
 */
export interface EpisodeTarget {
  /** the number to send, only meaningful when `problem` is null */
  value: number
  /** why this input cannot be used, in words for the operator — null when it is fine */
  problem: string | null
  /** a change we DID make and must therefore admit (e.g. 3.7 → 3) */
  note: string | null
}

export const EPISODE_MAX = 500

export function episodeTarget(raw: string): EpisodeTarget {
  const text = (raw ?? '').trim()
  if (!text) return { value: 0, problem: 'how many episodes? enter a number', note: null }
  // Number('') is 0 and Number('12 ') is 12; both are handled above. Anything with a stray
  // character is a TYPO, not a zero — the old code turned it into the default.
  const n = Number(text)
  if (!Number.isFinite(n)) return { value: 0, problem: `“${text}” is not a number`, note: null }
  if (n <= 0) {
    return { value: 0, problem: n === 0 ? 'zero episodes would record nothing' : 'that is a negative number of episodes', note: null }
  }
  if (n > EPISODE_MAX) {
    return { value: 0, problem: `${n} episodes is more than this screen will start (max ${EPISODE_MAX}) — record in batches`, note: null }
  }
  if (!Number.isInteger(n)) {
    const floored = Math.floor(n)
    return { value: floored, problem: null, note: `recording ${floored} episodes — ${text} is not a whole number` }
  }
  return { value: n, problem: null, note: null }
}
