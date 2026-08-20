/**
 * A number an operator typed, and whether we understood it — the general form of the record
 * sheet's episode parse (lib/episodeTarget.ts, BUGS.md Q59), for the training screen.
 *
 * `type="number"` is not validation. It stops most junk keystrokes and then hands you `""`, so
 * `Number(raw) || fallback` reads as harmless and is not:
 *
 *   steps:      cleared field → 10000 steps of fine-tuning (GPU hours, silently)
 *   n_episodes: cleared field → 5 episodes;  duration: cleared → 10s
 *   AND: `||` only catches 0 and NaN, so a MINUS sign sails straight through — `steps: -100` and
 *   `n_episodes: -3` were sent to the API verbatim, and a negative collect still grabs the arms
 *   before anything downstream can complain.
 *
 * So bounds are stated, not assumed, and a value we cannot use is refused with the reason in front
 * of the operator instead of replaced behind their back.
 */
export interface FieldNumber {
  /** the number to send — only meaningful when `problem` is null */
  value: number
  /** why this cannot be used, in words for the operator; null when it is fine */
  problem: string | null
  /** a change we DID make, and therefore must admit (e.g. 3.7 → 3) */
  note: string | null
}

export interface NumFieldRules {
  /** what this number IS, used in the sentence: "steps", "episodes", "seconds per episode" */
  what: string
  min: number
  max: number
  /** whole numbers only (steps, episodes) — a float is floored AND admitted */
  integer?: boolean
  /** how the max is phrased when refused, e.g. "record in batches" */
  remedy?: string
}

export function numField(raw: string, rules: NumFieldRules): FieldNumber {
  const text = (raw ?? '').trim()
  const { what, min, max, integer = true, remedy } = rules
  if (!text) return { value: 0, problem: `how many ${what}?`, note: null }
  const n = Number(text)
  if (!Number.isFinite(n)) return { value: 0, problem: `“${text}” is not a number`, note: null }
  if (n < min) {
    return {
      value: 0,
      note: null,
      problem: n < 0
        ? `${what} cannot be negative`
        : `${n} is below the minimum of ${min} ${what}`,
    }
  }
  if (n > max) {
    return {
      value: 0,
      note: null,
      problem: `${n} is more than this screen will start (max ${max} ${what})${remedy ? ` — ${remedy}` : ''}`,
    }
  }
  if (integer && !Number.isInteger(n)) {
    const floored = Math.floor(n)
    if (floored < min) return { value: 0, problem: `${what} cannot be less than ${min}`, note: null }
    return { value: floored, problem: null, note: `using ${floored} ${what} — ${text} is not a whole number` }
  }
  return { value: n, problem: null, note: null }
}
