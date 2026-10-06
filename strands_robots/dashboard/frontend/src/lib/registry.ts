/**
 * Normaliser for `/api/robots/registry`. The endpoint answers a list of rich entries (`{name,
 * description, category, joints, has_sim, has_real}`), but it has also answered a list of bare
 * names and a `{name: definition}` map.
 */

export type RegistryRobot = {
  /** the id to send back to the spawner */
  name: string
  /** what to show in the dropdown */
  label: string
  /** How `mode=real` reaches it: a servo bus here, or an address on the network. Absent = no driver. */
  realTransport?: RealTransport
}

export type RealTransport = {
  /** 'serial': `port` is a /dev path on this machine; 'address': it is the robot's IP/host/URI. */
  portKind: 'serial' | 'address'
  /** the driver takes the NIC its DDS traffic binds to (G1, Go2) */
  networkInterface: boolean
}

function realTransportOf(v: unknown): RealTransport | undefined {
  if (!v || typeof v !== 'object') return undefined
  const o = v as Record<string, unknown>
  if (o.port_kind !== 'serial' && o.port_kind !== 'address') return undefined
  return { portKind: o.port_kind, networkInterface: o.network_interface === true }
}

/** `keyName` is the MAP KEY, and where it exists it is authoritative for the id. */
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
  const realTransport = realTransportOf(o.real_transport)
  // A native driver can build a robot whose entry declares no hardware: it is not sim only.
  if (o.has_real === false && o.has_sim === true && !realTransport) bits.push('sim only')
  if (realTransport?.portKind === 'address') bits.push('networked')
  const label = bits.length ? `${name} — ${bits.join(', ')}` : name
  return realTransport ? { name, label, realTransport } : { name, label }
}

/** One name, one option. */
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
