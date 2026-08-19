/**
 * U19: the camera reconfigure editor's pure half.
 *
 * Rows are what a human edits (strings from <input>s); the config is what the
 * backend takes (lerobot-shaped, typed). The conversion refuses locally with
 * the SAME bounds the server enforces (device_manager.validate_cameras), so
 * the operator reads the refusal next to the field instead of in an HTTP 422 —
 * but the server stays the law: this mirror existing does not excuse it.
 */

export interface CamRow {
  name: string
  /** OpenCV index ("0") or a device path ("/dev/video1") — as typed. */
  indexOrPath: string
  fps: string
  width: string
  height: string
}

export interface CamConfig {
  [name: string]: { index_or_path: number | string; fps?: number; width?: number; height?: number }
}

export function rowsFromConfig(config: CamConfig | null | undefined): CamRow[] {
  return Object.entries(config ?? {}).map(([name, c]) => ({
    name,
    indexOrPath: String(c.index_or_path ?? ''),
    fps: c.fps != null ? String(c.fps) : '',
    width: c.width != null ? String(c.width) : '',
    height: c.height != null ? String(c.height) : '',
  }))
}

/** "0" is an index, "/dev/video1" (or anything non-numeric) is a path. */
export function parseIndexOrPath(raw: string): number | string | null {
  const t = raw.trim()
  if (!t) return null
  if (/^\d+$/.test(t)) return Number(t)
  return t
}

const BOUNDS: Record<'fps' | 'width' | 'height', [number, number]> = {
  fps: [1, 240],
  width: [16, 7680],
  height: [16, 4320],
}

function intField(row: CamRow, field: 'fps' | 'width' | 'height'): { value?: number; error?: string } {
  const raw = row[field].trim()
  if (!raw) return {}
  if (!/^\d+$/.test(raw)) return { error: `${row.name || '(unnamed)'}: ${field} must be a whole number` }
  const v = Number(raw)
  const [lo, hi] = BOUNDS[field]
  if (v < lo || v > hi) return { error: `${row.name || '(unnamed)'}: ${field}=${v} is outside ${lo}..${hi}` }
  return { value: v }
}

/**
 * Rows -> backend payload. `{ cameras: null }` when every row was removed —
 * detaching all cameras is a legal, deliberate config, not an error.
 */
export function configFromRows(rows: CamRow[]): { cameras: CamConfig | null; error?: string } {
  const live = rows.filter(r => r.name.trim() || r.indexOrPath.trim() || r.fps.trim() || r.width.trim() || r.height.trim())
  if (live.length === 0) return { cameras: null }
  const out: CamConfig = {}
  for (const row of live) {
    const name = row.name.trim()
    if (!name) return { cameras: null, error: 'every camera needs a name (top / wrist / main…)' }
    if (out[name]) return { cameras: null, error: `two cameras are both named "${name}"` }
    const iop = parseIndexOrPath(row.indexOrPath)
    if (iop === null) return { cameras: null, error: `${name}: needs an index (0, 1…) or a device path` }
    const entry: CamConfig[string] = { index_or_path: iop }
    for (const field of ['fps', 'width', 'height'] as const) {
      const { value, error } = intField(row, field)
      if (error) return { cameras: null, error }
      if (value != null) entry[field] = value
    }
    // Width and height describe one rectangle: half of it is a typo, and the
    // driver "helpfully" filling the other half hides which half was meant.
    if ((entry.width == null) !== (entry.height == null)) {
      return { cameras: null, error: `${name}: give both width and height, or neither (driver default)` }
    }
    out[name] = entry
  }
  return { cameras: out }
}

/** What the confirm sheet must say — the cost is a respawn, named plainly. */
export function applySummary(rows: CamRow[], peerId: string): string {
  const { cameras } = configFromRows(rows)
  const n = cameras ? Object.keys(cameras).length : 0
  const what = n === 0 ? 'detach every camera from' : `apply ${n} camera${n === 1 ? '' : 's'} to`
  return `This will ${what} ${peerId} by restarting it — its streams (and any running task) stop during the restart.`
}
