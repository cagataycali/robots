/**
 * What the collect panel may claim when a record action fails.
 *
 * Last screen carrying this bug (after estop, training, robot cards, devices).
 * `run()` printed the raw error, i.e. "the click did nothing" — but api() throws
 * HttpError(0) for a request that never left this machine AND for one that
 * reached the recorder, ACTED, and lost the answer; a 5xx means the handler ran.
 * What the five actions leave behind when that happens:
 *
 *   open    -> the session may already be open: both arms despawned, their ports
 *              handed to the recorder, the follower energised and stiff.
 *   start   -> the episode may already be RECORDING. Believing it is not is how
 *              you get a take of an empty workspace, or walk away mid-capture.
 *   stop    -> the take may already be SAVED, or not. Nobody can tell from here.
 *   redo    -> the take may already be THROWN AWAY. That is not undoable.
 *   discard -> the same, for an episode chosen by index.
 *   close   -> the dataset may already be finished, and if upload was ticked it
 *              may already be PUSHED TO THE HUB, which is public and not local.
 *
 * The observer to hand off to is unusually good here: the panel re-reads
 * /api/record/session every second, so the episode list and the recording pill
 * answer the question within a tick — as long as something is polling, which is
 * why the ambiguous branch also forces one immediate read (an `open` that may
 * have landed leaves `s.dataset` null, and the poll only runs with a session).
 */
import { refusedBeforeActing } from './estopOutcome'

export type RecordActionKind = 'open' | 'start' | 'stop' | 'redo' | 'discard' | 'close'

export interface RecordFailureVerdict {
  text: string
  /** true = the recorder may have acted; re-read the session before retrying. */
  ambiguous: boolean
  /** true = the ambiguous outcome may have destroyed a take. */
  destructive: boolean
}

/** What did NOT happen, when the server refused before running anything. */
const INERT: Record<RecordActionKind, string> = {
  open: 'no session was opened — the arms are untouched and still in the fleet.',
  start: 'no episode was started — nothing is being recorded.',
  stop: 'the episode was NOT stopped — if one was recording, it still is.',
  redo: 'nothing was thrown away — the take is still there.',
  discard: 'nothing was discarded — that episode is still there.',
  close: 'the dataset was NOT finished — the session is still open, and nothing was uploaded.',
}

/** What MAY have happened, when the answer was lost. */
const MAYBE: Record<RecordActionKind, string> = {
  open: 'the session MAY be open: both arms despawned, ports handed to the recorder, follower energised and stiff.',
  start: 'the episode MAY already be recording — do the demonstration only once you know, and do not walk away assuming it is idle.',
  stop: 'the take MAY already be saved, or MAY still be recording — this panel cannot tell which.',
  redo: 'the take MAY already have been thrown away, and that cannot be undone.',
  discard: 'that episode MAY already have been discarded, and that cannot be undone.',
  close: 'the dataset MAY already be finished — and if you ticked upload, MAY already be on the Hub.',
}

const WATCH = 'Re-reading the session now — the episode list and the recording pill are the truth, they refresh every second.'

export function recordFailure(input: {
  kind: RecordActionKind
  status?: number | null
  message?: string | null
}): RecordFailureVerdict {
  const why = String(input.message ?? '').trim() || 'no detail'

  if (refusedBeforeActing(input.status)) {
    return {
      text: `✗ refused (${input.status}): ${why} — ${INERT[input.kind]}`,
      ambiguous: false,
      destructive: false,
    }
  }
  const head = Number(input.status ?? 0)
    ? `⚠ unknown — the server failed mid-request (${input.status}: ${why})`
    : `⚠ unknown — no answer came back (${why})`
  return {
    text: `${head}: ${MAYBE[input.kind]} ${WATCH}`,
    ambiguous: true,
    destructive: input.kind === 'redo' || input.kind === 'discard',
  }
}
