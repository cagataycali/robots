/**
 * Newest training run first — and, more importantly, the five that get POLLED.
 *
 * TrainingTab reversed the ledger (`jobs.slice().reverse()`) and then polled
 * `jobs.slice(0, 5)`. Both halves are one unstated assumption: that the file's order
 * is submission order, so the tail is the newest. When that holds, reversing works.
 * When it does not — a hand-edited ledger, a file merged from two machines, a restore
 * from a quarantined copy, rows rewritten by a future writer — the five jobs that get
 * status polls are the five OLDEST, which are exactly the runs that already finished.
 * The run the operator just started then sits there with no state, no loss curve and
 * no export button, looking like a submit that silently failed, while the poller
 * spends its budget on runs whose answers cannot change.
 *
 * So the order is derived from the DATA (`submitted_at`), not from the file's shape.
 *
 * A row with no timestamp is not guessed about: legacy rows predate the field, so they
 * keep their file order (reversed, the old behaviour) and sit after the rows that can
 * prove when they started. A ledger where nothing carries a timestamp therefore
 * behaves exactly as before — this replaces an assumption with a measurement without
 * changing a single well-formed case.
 */

export interface OrderableJob {
  /** epoch seconds recorded at submit; absent on rows written before the field existed */
  submitted_at?: number
}

/** True when the value is a usable epoch-second timestamp. */
function knownTime(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v) && v > 0
}

/**
 * Newest first. Rows that cannot say when they started keep their (reversed) file
 * order and follow the ones that can.
 */
export function orderJobsNewestFirst<T extends OrderableJob>(jobs: readonly T[]): T[] {
  const timed: T[] = []
  const untimed: T[] = []
  for (const j of jobs) (knownTime(j?.submitted_at) ? timed : untimed).push(j)
  // Sort is stable in every engine we target, so two runs submitted within the same
  // second keep the order the ledger recorded them in rather than swapping around on
  // each refresh.
  timed.sort((a, b) => (b.submitted_at as number) - (a.submitted_at as number))
  untimed.reverse()
  return [...timed, ...untimed]
}
