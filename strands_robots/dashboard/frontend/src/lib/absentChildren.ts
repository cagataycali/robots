/**
 * "Where did my robot go?" — the fleet-side half of U22.
 *
 * A peer that stops publishing is pruned from the snapshot, correctly: it really has
 * left the fleet. The consequence was that a robot the operator STARTED could die and
 * the only visible change was one fewer card. Measured on the live rig: the sole peer
 * publishing joints was SIGKILLed, and the fleet screen's answer was silence for a day.
 *
 * The server now sends `absent_children` inside the snapshot (dead managed children,
 * already pruned from `peers`). This module turns that list into ONE quiet line that
 * points at the devices drawer, where the full ledger and the log ring already live.
 *
 * Deliberately not corpse cards on the fleet grid: a card invites a click, and every
 * command on it would refuse. A card is for something you can act on.
 */
import { deathVerdict } from './childDeath'

export interface AbsentChild {
  peer_id: string
  robot_name?: string | null
  mode?: string | null
  returncode?: number | null
  started_at?: number | null
}

export interface AbsentNotice {
  /** the single line the fleet bar shows */
  headline: string
  /** one line per absent child — the tooltip, and what a test can read */
  detail: string
  /** how many children this notice speaks for */
  count: number
}

/** The cause without its explanatory clause — the bar has one line, the tooltip has room. */
export function shortCause(returncode?: number | null): string {
  return deathVerdict(returncode).phrase.split(' — ')[0]
}

export function absentNotice(children: readonly AbsentChild[] | null | undefined): AbsentNotice | null {
  // An older server sends no field at all. Absent means "this server cannot tell you",
  // which must render nothing — inventing "all present" from silence is the U15 lesson.
  if (!Array.isArray(children) || children.length === 0) return null
  const named = children.filter(c => c && typeof c.peer_id === 'string' && c.peer_id.length > 0)
  // A child that exited 0 finished its job: a recording that completed and left the mesh
  // is EXPECTED, and a bar that announces expected things gets ignored, taking the
  // surprises with it. Clean exits stay in the drawer's ledger.
  const surprises = named.filter(c => c.returncode !== 0)
  if (surprises.length === 0) return null
  const detail = surprises
    .map(c => `${c.peer_id} — ${deathVerdict(c.returncode).phrase}`)
    .join('\n')
  const headline =
    surprises.length === 1
      // The name FIRST: the operator is looking for a specific robot, not a count.
      ? `${surprises[0].peer_id} is gone — ${shortCause(surprises[0].returncode)}`
      : `${surprises.length} robots you started are gone`
  return { headline, detail, count: surprises.length }
}

/**
 * Q155b: the OTHER way a robot you started can be missing — it is still RUNNING and the
 * fleet has never heard of it. The server sends `managed_no_presence`: ids the dashboard
 * holds a live child process for, with no peer at all (measured on the real rig — a sim
 * child alive at 25h with no card, because a peer that does not appear cannot be rendered
 * stale, mute, or at all).
 *
 * A separate sentence from death on purpose. "Gone" would be a lie about a live process,
 * and the remedy is the opposite: nothing needs restarting, something needs READING — the
 * child's own log holds the refusal (a missing calibration, a busy servo bus), and despawn
 * is there if it is not wanted. Same destination as the death chip, since the drawer is
 * where logs and the managed ledger already live.
 */
export function quietNotice(
  ids: readonly string[] | null | undefined,
  dead: readonly AbsentChild[] | null | undefined = [],
): AbsentNotice | null {
  // An older server sends no field: absent means "this server cannot tell you", which
  // renders nothing. Claiming "all children reported in" from silence is the U15 lesson.
  if (!Array.isArray(ids) || ids.length === 0) return null
  const buried = new Set(
    (Array.isArray(dead) ? dead : []).map(c => c && c.peer_id).filter(Boolean) as string[],
  )
  // The server derives these from the LIVE managed set, so an id cannot be in both lists.
  // Enforced here anyway: if it ever happens, DEATH wins — it is the more specific claim
  // (it carries an exit status) and two chips about one robot would read as two robots.
  const quiet = ids.filter(id => typeof id === 'string' && id.length > 0 && !buried.has(id))
  if (quiet.length === 0) return null
  const detail = quiet
    .map(id => `${id} — the process is running, but it has never joined the fleet`)
    .join('\n')
  const headline =
    quiet.length === 1
      // Name first: the operator is hunting one robot, not counting.
      ? `${quiet[0]} started but never joined the fleet`
      : `${quiet.length} robots you started never joined the fleet`
  return { headline, detail, count: quiet.length }
}
