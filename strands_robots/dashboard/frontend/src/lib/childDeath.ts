/**
 * What a managed child's DEATH means — U22.
 *
 * THE DEFECT, measured on the live rig 2026-08-21. The sim twin (the only peer
 * publishing joints) was SIGKILLed at 11:52:02Z. The devices drawer said, for the
 * whole day after:
 *
 *     so101-follower-twin   sim · exited
 *     14:00:32 [safety:so101-follower-twin] No emergency-stop resume code set...
 *
 * Both halves of that mislead:
 *
 *  1. "exited" is the word this drawer uses for EVERY death — a clean finish, a
 *     Python traceback, and a kill -9 all read identically. The server has known
 *     better all along: /api/devices reports `returncode` (-9 here), and the
 *     dashboard even logs "child exited (code=-9)" once at INFO. The frontend
 *     simply never asked — `grep returncode` across src/ found nothing.
 *  2. The log tail under it is the child's STARTUP output, not its last words. The
 *     ring buffer keeps 10 lines; a quiet sim that printed nothing for 22h leaves
 *     its birth cry sitting there looking like a cause of death. Read together the
 *     row says "it printed some warnings and exited", which is a story about a
 *     robot that stopped itself. The truth was that something killed it.
 *
 * Why that mattered: two days of supervisor iterations went into rediscovering,
 * from a 38MB server log and hand-written curls, what the exit status says in one
 * word. An operator with no shell has no path to it at all.
 *
 * Both functions are pure and take their evidence as arguments, so the sentences
 * can be tested rather than eyeballed.
 */

export interface DeathVerdict {
  /** the phrase shown where "exited" used to be */
  phrase: string
  /**
   * true when the status alone does not explain WHO ended it, so the row must not
   * imply the robot stopped itself. A SIGKILL is the honest example: nothing in
   * this dashboard sends one (despawn asks with SIGTERM first), so the sender was
   * outside — the OS under memory pressure, a script, or a person.
   */
  unexplained: boolean
}

/** Signals worth naming; anything else is reported by number rather than guessed at. */
const SIGNALS: Record<number, DeathVerdict> = {
  9: {
    phrase: 'killed (SIGKILL) — nothing here sends that: despawn asks with SIGTERM first, so the OS under memory pressure, a script, or a person ended it',
    unexplained: true,
  },
  15: { phrase: 'asked to stop (SIGTERM) — a despawn, a shutdown, or a script', unexplained: false },
  6: { phrase: 'crashed (abort) — its own output names where', unexplained: false },
  11: { phrase: 'crashed (segfault) — its own output names where', unexplained: false },
  2: { phrase: 'interrupted (SIGINT) — a Ctrl-C in whatever started it', unexplained: false },
  1: { phrase: 'hung up (SIGHUP) — the terminal that owned it went away', unexplained: false },
}

/**
 * `returncode` as Python reports it: 0 clean, >0 a Python-level failure, negative
 * the signal that ended it, and null/undefined "no status recorded".
 */
export function deathVerdict(returncode: number | null | undefined): DeathVerdict {
  if (returncode === null || returncode === undefined) {
    // Not the same as a clean exit, and saying "exited" here is the old bug: a child
    // that never started reports no status either.
    return { phrase: 'gone, with no exit status recorded — its log is the only witness', unexplained: true }
  }
  if (returncode === 0) return { phrase: 'exited cleanly (code 0) — it finished, or something asked it to', unexplained: false }
  if (returncode > 0) return { phrase: `exited with code ${returncode} — a failure inside the robot; its own output names it`, unexplained: false }
  const sig = -returncode
  return SIGNALS[sig] ?? { phrase: `killed by signal ${sig}`, unexplained: true }
}

/**
 * Is the retained log tail the child's STARTUP burst rather than its last words?
 *
 * The lines carry a bare wall clock ("14:00:32 …") and `started_at` is an epoch
 * second, so the only honest comparison is seconds-past-the-hour: the viewer's
 * timezone may differ from the server's by whole hours, and a claim that flips
 * with the reader's location is worse than no claim. A half-hour zone (India,
 * Nepal) therefore yields `null` — no claim — by design.
 *
 * Every clocked line must fall inside the window, not just the last one: a child
 * whose 10 retained lines are spread across hours is genuinely still talking, and
 * one coincidental match must not relabel it.
 *
 * Returns null whenever the evidence cannot decide, and callers must render
 * nothing for null — a guess here would re-tell exactly the story this fixes.
 */
export function retainedOutputIsStartup(
  input: { lines?: string[] | null; startedAt?: number | null; windowS?: number },
): boolean | null {
  const lines = input.lines ?? []
  const startedAt = input.startedAt
  if (!lines.length || startedAt === null || startedAt === undefined || !Number.isFinite(startedAt)) return null
  const windowS = input.windowS ?? 120
  const d = new Date(startedAt * 1000)
  const startPastHour = d.getMinutes() * 60 + d.getSeconds()
  let clocked = 0
  for (const line of lines) {
    const m = /^(\d{2}):(\d{2}):(\d{2})\b/.exec(line ?? '')
    if (!m) continue
    clocked++
    const pastHour = Number(m[2]) * 60 + Number(m[3])
    let diff = Math.abs(pastHour - startPastHour)
    if (diff > 1800) diff = 3600 - diff // the hour wraps; 59:59 is 2s from 00:01
    if (diff > windowS) return false
  }
  return clocked === 0 ? null : true
}
