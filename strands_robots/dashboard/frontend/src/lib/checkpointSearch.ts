/**
 * What the checkpoint type-ahead may claim about an empty result — and which
 * answer it is allowed to show at all.
 *
 * Two lies lived in this widget's silence:
 *
 * 1. `no checkpoints match “q”` is a claim about the WHOLE catalogue, but when
 *    the backend reports `hub_problem` ("Hub search unavailable … showing local
 *    cache only") the only thing searched was this machine's HF cache. An
 *    operator reading "no matches" concludes the checkpoint does not exist, when
 *    in fact nothing asked the Hub.
 * 2. Type-ahead requests race. The 300ms debounce cancels a pending TIMER, not
 *    an in-flight fetch, so a slow search for "act" could resolve after a fast
 *    one for "smolvla" and paint act's rows — or an older failure — under the
 *    newer query. Results shown next to a query they do not belong to are the
 *    most confident kind of wrong.
 */

export interface EmptyNoteInput {
  /** What the operator typed (may be empty: focus triggers a search). */
  query: string
  /** Backend's reason the Hub half did not answer, if any. */
  hubProblem?: string | null
}

/** The honest sentence for "the search returned nothing". */
export function emptyNote({ query, hubProblem }: EmptyNoteInput): string {
  const q = String(query ?? '').trim()
  const named = q ? `“${q}”` : 'that'
  const problem = String(hubProblem ?? '').trim()
  if (problem) {
    // Scope the claim to what was actually consulted, and repeat WHY here: this
    // line is the one being read, and the warning above it may have scrolled.
    return q
      ? `no checkpoint already on this machine matches ${named} — and the Hub was not searched (${problem}), so this is not "it does not exist".`
      : `nothing in this machine's local cache — and the Hub was not searched (${problem}), so the catalogue is unknown from here.`
  }
  return q
    ? `no checkpoints match ${named} (local cache + Hub).`
    : 'type part of a checkpoint name — local cache and the Hub are both searched.'
}

/**
 * True when a response may be rendered: only the newest request speaks. Sequence
 * numbers, not timestamps — two responses in the same millisecond still have an
 * order, and a clock that jumps cannot reorder them.
 */
export function isCurrent(responseSeq: number, latestSeq: number): boolean {
  return Number.isFinite(responseSeq) && responseSeq === latestSeq
}
