/**
 * The one-line account of what a servo board was last spawned as (Q41), and the trap inside it.
 *
 * The row already carries the MEASURED role (12.6V = follower, 7.4V = leader). The memory carries a
 * lerobot calibration id and a peer name, both of which are just NAMES an operator once typed — so a
 * board measured as a follower can honestly read "last spawned as so101-arm-2 · calibration id
 * leader_arm". That is exactly the mislabel cagatay first reported one surface over, and the
 * calibrate expander already warns about it. This file exists because iteration 135 reintroduced the
 * same trap on the row itself, one click earlier, with no warning at all.
 *
 * The rule: a name is never evidence. It is shown, and where it contradicts a measurement the
 * contradiction is stated — the memory is still correct to reuse (the calibration file lives under
 * that id), so this is a note, never a refusal.
 */

export interface RememberedSpawn {
  peer_id: string
  robot_name?: string | null
  mode?: string | null
  cameras: string[]
  robot_id?: string
  saved_at?: number | null
}

/** Does a name claim the OTHER role? Only the two words count — an index like "arm-2" is not evidence. */
export function nameClaimsOtherRole(name: string | null | undefined, role: string): boolean {
  const n = (name ?? '').trim().toLowerCase()
  const r = (role ?? '').trim().toLowerCase()
  if (!n || (r !== 'follower' && r !== 'leader')) return false
  const other = r === 'follower' ? 'leader' : 'follower'
  return n.includes(other) && !n.includes(r)
}

export interface RememberedLine {
  /** "so101-arm-1 — so101, real, cameras top + wrist" */
  summary: string
  /** the calibration id, when one is remembered */
  calibrationId?: string
  /** present when a remembered NAME contradicts the measured role — a note, never a refusal */
  warning?: string
}

export function rememberedLine(
  r: RememberedSpawn | null | undefined,
  facts: { role?: string | null; role_volts?: number | null } = {},
): RememberedLine | null {
  if (!r || !r.peer_id) return null
  const role = (facts.role ?? '').trim().toLowerCase()
  const bits: string[] = []
  if (r.robot_name) bits.push(String(r.robot_name))
  if (r.mode) bits.push(String(r.mode))
  // Camera NAMES: the saved indices are what macOS renumbers between reboots, so printing them here
  // would be the confidently-stale kind of detail this dashboard keeps deleting.
  bits.push(r.cameras.length ? `cameras ${r.cameras.join(' + ')}` : 'no cameras')
  const line: RememberedLine = {
    summary: `${r.peer_id}${bits.length ? ' — ' + bits.join(', ') : ''}`,
  }
  if (r.robot_id) line.calibrationId = r.robot_id

  const badId = nameClaimsOtherRole(r.robot_id, role)
  const badPeer = nameClaimsOtherRole(r.peer_id, role)
  if (badId || badPeer) {
    const volts = facts.role_volts != null ? `${facts.role_volts}V` : 'its measured voltage'
    const named = badId ? `the calibration id is named "${r.robot_id}"` : `this peer is named "${r.peer_id}"`
    line.warning =
      `${named} while this bus measures ${volts} = ${role}. The measurement is the fact and the ` +
      `name is what is wrong — reuse the memory anyway (the calibration file lives under that id), ` +
      `but do not let the name convince you this is the other arm`
  }
  return line
}
