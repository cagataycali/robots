/**
 * Deploy intent: the one-way handoff from the Training tab to a robot's run
 * form.
 *
 * "Deploy this checkpoint" CANNOT start a policy by itself - the run form is
 * per-robot and running a policy moves a physical arm, so the only safe
 * meaning of the button is "carry this checkpoint to the form and let the
 * human press Run". This module is that carry: a stamped note in
 * sessionStorage (survives the overlay unmounting, dies with the browser
 * tab - a deploy decision should not ambush the user tomorrow).
 *
 * The intent expires after 10 minutes and is consumed by the first run form
 * that applies it: both rules exist so a forgotten click can never resurface
 * as a mystery prefill on some other robot later.
 */

const KEY = 'strands.deployIntent'
const TTL_MS = 10 * 60 * 1000
/** Tolerated backwards clock drift before a stamp is judged untrustworthy rather than fresh. */
const CLOCK_GRACE_MS = 60 * 1000

export interface DeployIntent {
  checkpoint: string
  policy_type: string | null
  /** where it came from, shown in the banner so the prefill explains itself */
  source: string
  at: number
}

export function setDeployIntent(i: Omit<DeployIntent, 'at'>): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify({ ...i, at: Date.now() }))
  } catch { /* storage full/blocked - the button still did nothing dangerous */ }
}

export function peekDeployIntent(now = Date.now()): DeployIntent | null {
  try {
    const raw = sessionStorage.getItem(KEY)
    if (!raw) return null
    const i = JSON.parse(raw) as DeployIntent
    if (!i || typeof i.checkpoint !== 'string' || !i.checkpoint) return null
    // Age outside [-GRACE, TTL] is not a valid intent. The upper bound is the 10-minute
    // expiry; the LOWER bound matters because `now - at > TTL` alone can never expire a
    // stamp from the FUTURE — a system clock that jumps back (sleep/resume, an NTP
    // correction, a VM snapshot) leaves an intent whose age is negative forever, i.e. a
    // deploy prefill with no expiry at all, which is exactly what the TTL exists to prevent.
    // A clock we cannot trust is treated as an expired intent, never as a fresh one: the
    // cost is re-clicking Deploy, and the alternative is a checkpoint prefilled into some
    // other robot's form tomorrow.
    const age = now - i.at
    if (typeof i.at !== 'number' || !Number.isFinite(age) || age > TTL_MS || age < -CLOCK_GRACE_MS) {
      sessionStorage.removeItem(KEY)
      return null
    }
    return i
  } catch {
    return null
  }
}

export function clearDeployIntent(): void {
  try { sessionStorage.removeItem(KEY) } catch { /* already gone */ }
}
