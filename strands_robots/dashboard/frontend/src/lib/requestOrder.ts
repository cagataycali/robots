/**
 * Which answer is allowed to reach the screen.
 *
 * Three places in this dashboard fire a request per keystroke or per tick, and a
 * response is not ordered with respect to its siblings. The rule is always the
 * same — the newest REQUEST owns the screen, and an answer to a superseded one is
 * discarded rather than painted — so it lives here once, and the two existing
 * copies (lib/checkpointSearch.isCurrent, lib/datasetHint.isCurrentResponse)
 * delegate to it.
 *
 * The training status poll is the sharpest case, because there the stale answer
 * does not merely look wrong, it DESTROYS data: lossTrace.pushLoss reads a step
 * lower than the last one as "the job restarted" and drops the whole curve. So an
 * older tick landing late wipes a healthy run's loss history and claims a restart
 * that never happened, while polledAt is refreshed to now — the freshness rail
 * says fresh and the numbers have gone backwards. Ordering has to be enforced
 * before a response is interpreted, not after.
 */

/** The newest request wins: only the answer whose seq is still the latest speaks. */
export function isLatestRequest(seq: number, latest: number): boolean {
  return seq === latest
}

/**
 * For per-key state (one job's status, one camera's frame): apply an answer only
 * if it is newer than what is already displayed. `applied` is the seq whose data
 * is on screen; undefined means nothing has been applied yet, so anything is
 * newer. Equal seq is NOT newer — a retry of the same request carries no new
 * information and must not re-stamp the freshness clock.
 */
export function newerThanApplied(seq: number, applied: number | undefined): boolean {
  return applied === undefined || seq > applied
}
