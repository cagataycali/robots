/**
 * Q58 (frontend half): what the training form must say and refuse about `output_dir`.
 *
 * The trainer clears a pre-existing output_dir that holds no resumable checkpoint with
 * `shutil.rmtree`. The backend now refuses that launch unless `confirm_clear` is sent
 * (see dashboard/output_dir_check.py) — this file decides what the OPERATOR sees and
 * whether the train button may be armed at all.
 *
 * Rules encoded here, each learned from a way this screen has misled before:
 *
 * - a destructive run is armed by a tick that names the loss, and the tick authorises
 *   ONE path. Retyping the field revokes it (the dsOverride/deployIntent rule: one click
 *   authorises one action — a sticky yes silences a check nobody re-made).
 * - `resumable` and `not_a_dir` are NOT confirmable. No tick exists for them, because the
 *   run cannot succeed: the first is refused by lerobot itself (a failure that would
 *   otherwise appear only in the run's log, after the job row says started) and the second
 *   is a typo.
 * - `unknown` (the path could not be read) never becomes silent permission. It cannot be
 *   confirmed either: nobody can consent to a loss they were not shown.
 * - while the verdict is in flight the button is not blocked. Blocking on a pending read
 *   would make a slow filesystem look like a broken form; the backend still refuses, so
 *   the worst case is an error toast rather than a delete.
 */

export type OutputDirState = 'free' | 'occupied' | 'resumable' | 'not_a_dir' | 'unknown'

export interface OutputDirVerdict {
  path?: string
  state?: OutputDirState
  destructive?: boolean
  needs_confirm?: boolean
  detail?: string
  entries?: string[]
  total?: number
}

export interface OutputDirSay {
  /** null when there is nothing worth saying (a free path is the normal case). */
  text: string | null
  tone: 'info' | 'warn' | 'bad'
  /** show a confirmation tick? Only a destructive-but-possible run gets one. */
  confirmable: boolean
  /** the tick's label — it must name the consequence, not ask "are you sure?". */
  confirmLabel: string | null
  /** hard stop: no tick can help. */
  blocked: boolean
}

export function outputDirSay(v: OutputDirVerdict | null | undefined): OutputDirSay {
  const none: OutputDirSay = { text: null, tone: 'info', confirmable: false, confirmLabel: null, blocked: false }
  if (!v || !v.state) return none
  const detail = (v.detail ?? '').trim()
  switch (v.state) {
    case 'free':
      return none
    case 'occupied':
      return {
        text: `⚠ ${detail}`,
        tone: 'bad',
        confirmable: true,
        confirmLabel: `delete ${v.total ?? 'the'} item(s) in ${v.path ?? 'this directory'} and train here`,
        blocked: false,
      }
    case 'resumable':
      return { text: `⚠ ${detail}`, tone: 'warn', confirmable: false, confirmLabel: null, blocked: true }
    case 'not_a_dir':
      return { text: `✗ ${detail}`, tone: 'bad', confirmable: false, confirmLabel: null, blocked: true }
    case 'unknown':
      return { text: `⚠ ${detail}`, tone: 'warn', confirmable: false, confirmLabel: null, blocked: true }
    default:
      return none
  }
}

/**
 * May the run start, and does it need to carry `confirm_clear`?
 *
 * `armedFor` is the path the operator ticked the box for — not a boolean, so a tick made
 * for `/tmp/run-a` cannot authorise deleting `/tmp/run-b` after the field is edited.
 */
export function trainGate(args: {
  path: string
  verdict: OutputDirVerdict | null | undefined
  armedFor: string | null
  pending?: boolean
}): { ok: boolean; confirmClear: boolean; why: string | null } {
  const path = (args.path ?? '').trim()
  if (!path) return { ok: false, confirmClear: false, why: 'an output dir is required' }
  const say = outputDirSay(args.verdict)
  if (say.blocked) {
    return { ok: false, confirmClear: false, why: args.verdict?.detail ?? 'that output dir cannot be used' }
  }
  if (say.confirmable) {
    const armed = !!args.armedFor && args.armedFor === (args.verdict?.path ?? path)
    if (!armed) {
      return {
        ok: false,
        confirmClear: false,
        why: 'that directory is not empty — tick the box to confirm what gets deleted',
      }
    }
    return { ok: true, confirmClear: true, why: null }
  }
  // free, or no verdict yet: a pending read must not look like a broken form.
  return { ok: true, confirmClear: false, why: null }
}
