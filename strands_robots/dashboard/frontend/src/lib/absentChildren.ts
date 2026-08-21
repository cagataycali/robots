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
