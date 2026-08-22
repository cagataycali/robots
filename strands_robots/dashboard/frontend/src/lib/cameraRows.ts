/**
 * The multi-camera half of the spawn form: rows of {name, index} the operator composed,
 * judged into the exact `cameras` mapping /api/devices/spawn expects — or a refusal in
 * words. One rule owns this so the form cannot invent a second, slightly different one.
 *
 * The shapes that already bit us, pinned here:
 * - each entry must be a MAPPING ({index_or_path: N, ...}) — a bare int is the live
 *   ValueError "Camera 'main' config must be a mapping ... got int: 3";
 * - names become lerobot config keys, so they must be identifier-shaped;
 * - the same index twice is one physical camera with two capture threads — it fails at
 *   spawn with a claim error, so it is refused before the button.
 */

export interface CameraRow {
  /** the config key the operator chose: main, wrist, top… */
  name: string
  /** camera index as typed/selected; '' = row not filled in yet */
  index: string
}

export interface CameraShared {
  fps?: number | null
  width?: number | null
  height?: number | null
}

export interface CamerasField {
  /** the spawn payload — null when no row is filled in (spawn without cameras) */
  value: Record<string, Record<string, number>> | null
  /** why this cannot be sent, for the operator; null when it is fine */
  problem: string | null
}

const NAME_SHAPE = /^[A-Za-z][A-Za-z0-9_]*$/

export function camerasField(rows: CameraRow[], shared: CameraShared = {}): CamerasField {
  const filled = (rows ?? []).filter(r => (r.index ?? '').trim() !== '')
  if (filled.length === 0) return { value: null, problem: null }

  const seenNames = new Set<string>()
  const seenIndices = new Map<number, string>()
  const value: Record<string, Record<string, number>> = {}

  for (const row of filled) {
    const name = (row.name ?? '').trim()
    if (!name) {
      return { value: null, problem: 'every selected camera needs a name — main, wrist, top…' }
    }
    if (!NAME_SHAPE.test(name)) {
      return {
        value: null,
        problem: `“${name}” cannot be a camera name — letters, digits and _ only, starting with a letter`,
      }
    }
    const key = name.toLowerCase()
    if (seenNames.has(key)) {
      return { value: null, problem: `two cameras named “${name}” — the second would overwrite the first` }
    }
    seenNames.add(key)

    const index = Number(row.index)
    if (!Number.isInteger(index) || index < 0) {
      return { value: null, problem: `“${row.index}” is not a camera index` }
    }
    const claimant = seenIndices.get(index)
    if (claimant !== undefined) {
      return {
        value: null,
        problem: `index ${index} is used by both “${claimant}” and “${name}” — one capture thread per physical camera`,
      }
    }
    seenIndices.set(index, name)

    value[name] = {
      index_or_path: index,
      ...(shared.fps ? { fps: shared.fps } : {}),
      ...(shared.width ? { width: shared.width } : {}),
      ...(shared.height ? { height: shared.height } : {}),
    }
  }
  return { value, problem: null }
}
