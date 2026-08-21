/**
 * Telling the operator that a training run ENDED (Q159).
 *
 * A job's state is a polled word in the card — `running`, then `succeeded` or `failed`.
 * Nothing announced the change, and this is the longest-running thing the dashboard does:
 * minutes to hours, watched by nobody for most of it. So the one moment that matters was
 * the one moment with no signal, and a failure looked identical to still-working until you
 * came back and read the word.
 *
 * A TRANSITION rule, not a state rule, and the difference is the whole design:
 *   * a polled state announced every tick would repeat itself forever;
 *   * a job whose FIRST observed state is already terminal is NOT news — it finished
 *     before this tab existed, exactly like the activity history of Q158;
 *   * several finishing together become ONE sentence, failures named first, because two
 *     live regions racing produce interleaved half-sentences.
 */
export type JobStateMap = Record<string, string>

/** Terminal states the training providers actually report. Anything else is in flight. */
const DONE_OK = new Set(['succeeded', 'success', 'completed', 'complete', 'finished'])
const DONE_BAD = new Set(['failed', 'error', 'crashed', 'cancelled', 'canceled', 'stopped', 'killed'])

export function isTerminal(state: string | null | undefined): boolean {
  const s = (state ?? '').toLowerCase()
  return DONE_OK.has(s) || DONE_BAD.has(s)
}

/** Short id for speech: a uuid read aloud in full is unusable. */
export function shortJob(id: string): string {
  const s = String(id ?? '')
  return s.length <= 12 ? s : `${s.slice(0, 8)}…`
}

export function jobTransitions(prev: JobStateMap, now: JobStateMap): string {
  const ok: string[] = []
  const bad: string[] = []
  for (const [id, state] of Object.entries(now)) {
    const was = prev[id]
    // Unseen before means it was already finished when we first looked: not news.
    if (was === undefined) continue
    if (!isTerminal(state) || isTerminal(was)) continue
    const s = (state ?? '').toLowerCase()
    ;(DONE_BAD.has(s) ? bad : ok).push(`${shortJob(id)} ${s}`)
  }
  if (!bad.length && !ok.length) return ''
  // Failures lead: the sentence may be cut short by the next announcement, and the half
  // that survives should be the half that needs a human.
  const parts = [...bad, ...ok]
  const head = bad.length ? 'training job failed' : 'training job finished'
  return parts.length === 1
    ? `${head}: ${parts[0]}`
    : `${bad.length ? 'training jobs ended, some badly' : 'training jobs finished'}: ${parts.join(', ')}`
}
