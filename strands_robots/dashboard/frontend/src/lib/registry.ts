/**
 * Normaliser for `/api/robots/registry`.
 *
 * The endpoint answers a list of rich entries
 * (`{name, description, category, joints, has_sim, has_real}`), but it has also
 * answered a list of bare names and a `{name: definition}` map. Rendering an
 * entry straight into an `<option>` is what React error #31 is - "objects are
 * not valid as a React child" - and because that throws during render it takes
 * the whole dashboard down, not just the picker. So the shape is narrowed once,
 * here, and the UI only ever sees strings.
 */

export type RegistryRobot = {
  /** the id to send back to the spawner */
  name: string
  /** what to show in the dropdown */
  label: string
}

/**
 * `keyName` is the MAP KEY, and where it exists it is authoritative for the id.
 *
 * In `{name: definition}` shape the key IS the registry id — that is what the spawner takes
 * (registry/robots.py builds every entry's `name` from exactly that key). The value is free to be a
 * description string, and it used to become the id: `{ so101: 'SO-101 6-DOF arm' }` produced
 * name: 'SO-101 6-DOF arm', so the picker looked right and the spawn asked for a robot that does not
 * exist. A description now lands in the LABEL, where a description belongs.
 */
function entryToRobot(value: unknown, keyName?: string): RegistryRobot | null {
  if (typeof value === 'string') {
    const text = value.trim()
    if (keyName) return { name: keyName, label: text ? `${keyName} — ${text}` : keyName }
    return text ? { name: text, label: text } : null
  }
  if (!value || typeof value !== 'object') {
    return keyName ? { name: keyName, label: keyName } : null
  }
  const o = value as Record<string, unknown>
  const inner = typeof o.name === 'string' && o.name.trim() ? o.name.trim() : undefined
  // The key wins for the id; a differing inner name is a display name, not a spawn target.
  const name = keyName ?? inner
  if (!name) return null

  // A 72-entry dropdown of bare ids is hard to pick from; category and DOF are
  // already in the payload and say which arm this is.
  const bits: string[] = []
  if (keyName && inner && inner !== keyName) bits.push(inner)
  if (typeof o.category === 'string' && o.category) bits.push(o.category)
  if (typeof o.joints === 'number' && Number.isFinite(o.joints)) bits.push(`${o.joints} joints`)
  if (o.has_real === false && o.has_sim === true) bits.push('sim only')
  return { name, label: bits.length ? `${name} — ${bits.join(', ')}` : name }
}

/**
 * One name, one option. A registry that answers the same robot twice (a user registry shadowing a
 * built-in, the same list concatenated by a proxy) rendered two identical <option>s with the same
 * value AND the same React key — and a duplicate key means React reuses one element for both, so a
 * later change to one row can render on the other. First wins: the earlier entry is the more
 * specific one in every shape this endpoint has answered.
 */
function dedupe(rows: RegistryRobot[]): RegistryRobot[] {
  const seen = new Set<string>()
  return rows.filter(r => (seen.has(r.name) ? false : (seen.add(r.name), true)))
}

export function normalizeRegistry(robots: unknown): RegistryRobot[] {
  if (Array.isArray(robots)) {
    return dedupe(robots.map(r => entryToRobot(r)).filter((r): r is RegistryRobot => r !== null))
  }
  if (robots && typeof robots === 'object') {
    return dedupe(Object.entries(robots as Record<string, unknown>)
      .map(([k, v]) => entryToRobot(v, k))
      .filter((r): r is RegistryRobot => r !== null))
  }
  return []
}
