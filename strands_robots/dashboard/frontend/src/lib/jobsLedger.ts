/**
 * What the Jobs list may claim about a history it might not have.
 *
 * The job ledger is a file on the dashboard's disk, and it is the only record that
 * a training run happened — the thing that turns an hours-long run back into a card
 * with a status, a loss curve and an export button. When that file cannot be read,
 * the API answers with an empty list and a `problem` sentence, and the empty list is
 * indistinguishable from the honest empty list of a dashboard that has never trained
 * anything. One of those two means the operator's runs were forgotten *while they
 * are still running*, and it deserves different words and a different tone.
 *
 * Three states, not two:
 *   - jobs present, ledger fine        -> the list speaks for itself
 *   - no jobs, ledger fine             -> "nothing here yet", an ordinary answer
 *   - ledger unreadable                -> a warning, WHATEVER the count, because a
 *                                         partial list is the dangerous case: some
 *                                         cards render and the missing ones look
 *                                         like runs that never existed.
 */

export interface JobsLedgerInput {
  /** how many job rows the API returned */
  count: number
  /** the API's sentence about the LEDGER file, if it could not be read */
  problem?: string | null
}

export interface JobsLedgerVerdict {
  /** the line to render, or null when the list needs no explanation */
  text: string | null
  tone: 'info' | 'warn'
  /** true when rows may be missing — the caller must not present the list as complete */
  partial: boolean
}

export function jobsLedgerNotice({ count, problem }: JobsLedgerInput): JobsLedgerVerdict {
  const trimmed = (problem ?? '').trim()
  if (trimmed) {
    // The API's sentence already says what happened, where the bad file went and
    // that running jobs are unaffected. Repeating that here in different words is
    // how two surfaces come to disagree, so it is quoted, not paraphrased - only
    // the framing differs by count, because "some cards are missing" and "no cards
    // at all" are different things to be looking at.
    const lead = count > 0
      ? 'Some earlier runs may be missing from this list'
      : 'This list is empty because the history could not be read, not because nothing ran'
    return { text: `⚠ ${lead} — ${trimmed}`, tone: 'warn', partial: true }
  }
  if (count === 0) {
    return { text: 'No training jobs yet.', tone: 'info', partial: false }
  }
  return { text: null, tone: 'info', partial: false }
}
