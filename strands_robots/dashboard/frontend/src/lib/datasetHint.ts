/**
 * Which query the dataset list on screen actually belongs to.
 *
 * The picker searches the Hub as you type, and the copy under it is confident:
 * "nothing here or on the Hub matches 'so10'". But the results, the problem
 * verdict AND that sentence were three independent pieces of state, so they could
 * describe three different moments:
 *
 *   - Hub round trips are not ordered. A slow answer for "so" landing after a
 *     fast answer for "so101" repopulates the list with the WRONG rows, and the
 *     user picks a dataset a multi-hour job will then read.
 *   - The sentence always quoted the CURRENT input, so a verdict measured for a
 *     shorter query was attributed to the one being typed — the list said "no
 *     match for so101" while nobody had ever asked the Hub about so101.
 *   - While a search is in flight, an empty list from the PREVIOUS query renders
 *     "nothing matches", which is the wrong answer to a question still open.
 *
 * So: the shown results carry the query they were measured for, and this decides
 * what may be claimed. "I am still looking" is a legitimate, honest state.
 */

export interface DatasetHintInput {
  /** what is in the box right now */
  query: string
  /** the query the visible rows + problem were measured for (null = never searched) */
  shownQuery: string | null
  /** how many rows are visible */
  count: number
  /** the Hub half's failure, if it failed */
  problem?: string | null
  /** hf auth block from the API, when it says we are anonymous */
  anonymous?: boolean
  authDetail?: string | null
}

export interface DatasetHint {
  /** the main sentence under the picker, or null when the list speaks for itself */
  text: string | null
  tone: 'info' | 'warn' | 'pending'
  /** the extra "public results only" line, or null */
  auth: string | null
}

export function datasetHint(input: DatasetHintInput): DatasetHint {
  const q = input.query.trim()
  const shown = input.shownQuery === null ? null : input.shownQuery.trim()
  const stale = shown !== q

  // Anonymous Hub search is worth saying whenever the user is asking the Hub
  // something, because it changes what "no match" MEANS (gated repos are hidden).
  const auth = input.anonymous && q
    ? `Hub results are public only${input.authDetail ? ` (${input.authDetail})` : ''} — a private or gated dataset will look like “no match”.`
    : null

  // A failure that was measured for THIS query is the loudest true thing.
  if (input.problem && !stale) {
    return { text: `⚠ ${input.problem}`, tone: 'warn', auth }
  }
  // Results (or a verdict) that belong to a different question may not answer
  // this one. Say which state we are in instead of borrowing the old answer.
  if (stale) {
    return {
      text: q
        ? `searching for “${q}”…${input.count > 0 && shown ? ` — the rows below are still the results for “${shown}”` : ''}`
        : null,
      tone: 'pending',
      auth,
    }
  }
  if (input.count === 0 && q) {
    return {
      text: `nothing here or on the Hub matches “${q}” — the Hub answered, it simply has no match.`,
      tone: 'info',
      auth,
    }
  }
  return { text: null, tone: 'info', auth }
}

/**
 * Should this response be applied to the screen?
 *
 * The last request WINS, not the last response. Anything else is the out-of-order
 * bug above, and it is invisible in testing because a fast local search always
 * answers before a slow Hub one.
 */
export function isCurrentResponse(seq: number, latest: number): boolean {
  return seq === latest
}
