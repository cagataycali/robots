/**
 * WHY does this arm report no joints? — read out of its own log ring buffer.
 *
 * Both real arms on this fleet have been jointless since they spawned three days ago, while every other
 * surface calls them healthy: presence connected, camera frames flowing, and — the cruel part — the LAST
 * TWO LINES of their log are "hardware connected" and "<peer> (real @ /dev/cu.usbmodem…) online". The
 * failure is a WARNING several lines above the reassuring tail, and mesh/core logs repeats at DEBUG only,
 * so it never appears again. Nobody reads line 8 of 10.
 *
 * This turns that one line into a sentence with a remedy. It never guesses: an exception it does not
 * recognise is quoted VERBATIM, because a wrong remedy for a hardware fault is worse than no remedy.
 */
export interface JointFailure {
  /** the exception class, when the line named one */
  kind: string | null
  /** what happened, in the operator's terms */
  headline: string
  /** what to do about it — omitted when the failure is not one we recognise */
  remedy?: string
  /** the log line's own words, always kept */
  quote: string
  /** the log ends with reassuring lines that postdate the failure */
  tailMisleads?: boolean
}

const PROBE = /state probe 'hw_joints' failed[^:]*:\s*(.*)$/
const EXC = /^([A-Za-z_][A-Za-z0-9_]*)\(/
const PORT = /['"](\/dev\/[^'"]+)['"]/

/**
 * @param lines the ring buffer, oldest first (GET /api/devices/logs/{peer})
 * @returns null when nothing in the buffer explains a missing-joints state
 */
export function jointFailure(lines: string[] | null | undefined): JointFailure | null {
  const list = (lines ?? []).filter(l => typeof l === 'string')
  // The LAST probe failure: an arm can fail, be respawned and fail differently.
  let idx = -1
  for (let i = list.length - 1; i >= 0; i--) if (PROBE.test(list[i])) { idx = i; break }
  if (idx < 0) return null
  const rest = (list[idx].match(PROBE)?.[1] ?? '').trim()
  if (!rest) return null
  const kind = rest.match(EXC)?.[1] ?? null
  const port = rest.match(PORT)?.[1] ?? null
  // "hardware connected" / "online" printed AFTER the failure: true of the process, not of the joints.
  const tailMisleads = list.slice(idx + 1).some(l => /hardware connected|\bonline\b/.test(l))
  const out: JointFailure = { kind, quote: rest, headline: '', tailMisleads: tailMisleads || undefined }

  if (/Port is in use/i.test(rest) || /sync read 'Present_Position'/.test(rest)) {
    out.headline = `something else on this machine holds ${port ?? 'the serial port'}, so this arm cannot read its own position`
    out.remedy = 'the bus has more than one owner — stop the other holder (a leftover robot process or a script), then respawn this arm; nothing needs unplugging'
    return out
  }
  if (/has no calibration registered/i.test(rest)) {
    out.headline = 'this arm was spawned with a robot id that has no calibration file, so every joint read is refused'
    out.remedy = 'lerobot looks for calibration/robots/<type>/<robot_id>.json under ~/.cache/huggingface/lerobot — respawn it with an id that HAS a file, or calibrate it under this one. It is a name mismatch, not a hardware fault'
    return out
  }
  out.headline = kind
    ? `this arm could not read its joints — ${kind}, which the dashboard has no advice for`
    : 'this arm could not read its joints'
  return out
}

/** One line for a screen. */
export function jointFailureLine(f: JointFailure | null): string | null {
  if (!f) return null
  const tail = f.tailMisleads
    ? ' · its log then says "hardware connected" — that is the PROCESS, not the joints'
    : ''
  return `no joints: ${f.headline}${f.remedy ? ` — ${f.remedy}` : ''}${tail}`
}
