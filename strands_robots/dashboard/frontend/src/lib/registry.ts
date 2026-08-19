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

function entryToRobot(value: unknown, fallbackName?: string): RegistryRobot | null {
  if (typeof value === 'string') {
    const name = value.trim()
    return name ? { name, label: name } : null
  }
  if (!value || typeof value !== 'object') {
    return fallbackName ? { name: fallbackName, label: fallbackName } : null
  }
  const o = value as Record<string, unknown>
  const name = typeof o.name === 'string' && o.name.trim() ? o.name.trim() : fallbackName
  if (!name) return null

  // A 72-entry dropdown of bare ids is hard to pick from; category and DOF are
  // already in the payload and say which arm this is.
  const bits: string[] = []
  if (typeof o.category === 'string' && o.category) bits.push(o.category)
  if (typeof o.joints === 'number' && Number.isFinite(o.joints)) bits.push(`${o.joints} joints`)
  if (o.has_real === false && o.has_sim === true) bits.push('sim only')
  return { name, label: bits.length ? `${name} — ${bits.join(', ')}` : name }
}

export function normalizeRegistry(robots: unknown): RegistryRobot[] {
  if (Array.isArray(robots)) {
    return robots.map(r => entryToRobot(r)).filter((r): r is RegistryRobot => r !== null)
  }
  if (robots && typeof robots === 'object') {
    return Object.entries(robots as Record<string, unknown>)
      .map(([k, v]) => entryToRobot(v, k))
      .filter((r): r is RegistryRobot => r !== null)
  }
  return []
}
