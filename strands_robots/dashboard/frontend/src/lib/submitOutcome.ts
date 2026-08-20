/**
 * What the training tab may claim when a request that STARTS SOMETHING fails.
 *
 * Every action here answered a failure with `⚠ <message>`, which reads as "it
 * did not happen". For a lost answer that is a guess, and the same guess the
 * estop sheet used to make (see lib/estopOutcome.ts): `api()` throws
 * HttpError(0) both for a request that never left the machine and for one that
 * reached the server, ran, and lost the connection — and a 5xx means the handler
 * executed.
 *
 * The consequence here is not a stopped robot, it is a DUPLICATE:
 *   - /api/training/submit  -> a second multi-hour run on the same GPU, both
 *                              writing checkpoints into the same output_dir
 *   - /api/collect          -> a second recorder peer appending to the dataset
 *   - /api/replay           -> a second peer DRIVING THE SAME ARM
 * So an ambiguous failure must say "it may have started", name what a second
 * press would do, and point at the list where the truth already is — never
 * imply the button did nothing.
 *
 * The classifier is shared with the estop path deliberately: one definition of
 * "the server refused before running anything" for the whole dashboard.
 */
import { refusedBeforeActing } from './estopOutcome'

export type SideEffectKind = 'training' | 'collect' | 'replay' | 'export'

export interface SideEffectVerdict {
  text: string
  delivered: 'no' | 'unknown'
  /** True when pressing the button again could run the action a SECOND time. */
  doubleRunRisk: boolean
}

const NOUN: Record<SideEffectKind, string> = {
  training: 'the training job',
  collect: 'the collection run',
  replay: 'the replay',
  export: 'the export',
}

const DUPLICATE: Record<SideEffectKind, string> = {
  training: 'Pressing train again could start a SECOND run on the same GPU, writing into the same output_dir — check the job list below (it refreshes now) before you do.',
  collect: 'Pressing collect again could spawn a SECOND recorder appending to the same dataset — look for a new peer in the fleet grid first.',
  replay: 'Pressing replay again could spawn a SECOND peer driving the same arm — look for a new peer in the fleet grid first.',
  export: 'It is safe to retry an export, but check the artifact path before assuming nothing was written.',
}

export function sideEffectVerdict(input: {
  kind: SideEffectKind
  status?: number | null
  message?: string | null
}): SideEffectVerdict {
  const noun = NOUN[input.kind] ?? 'the request'
  const why = String(input.message ?? '').trim() || 'no detail'

  if (refusedBeforeActing(input.status)) {
    return {
      text: `✗ refused (${input.status}: ${why}) — ${noun} was NOT started, nothing is running.`,
      delivered: 'no',
      doubleRunRisk: false,
    }
  }
  const transport = !Number(input.status ?? 0)
  const head = transport
    ? `⚠ no answer came back (${why}) — ${noun} MAY have started; this page cannot tell.`
    : `⚠ the server failed mid-request (${input.status}: ${why}) — ${noun} MAY have started anyway.`
  return {
    text: `${head} ${DUPLICATE[input.kind] ?? ''}`.trim(),
    delivered: 'unknown',
    doubleRunRisk: input.kind !== 'export',
  }
}
