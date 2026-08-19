/**
 * Settings metadata: every tunable in the Settings drawer earns its place here
 * with a plain-language explanation, a unit, its safe default, HOW it applies
 * (live vs next turn vs mesh restart), and a validator.
 *
 * Pure module - no React, no fetch - so it can be verified with node directly
 * (this repo's frontend has no test runner; see the esbuild+node technique in
 * the repo notes).
 */

/** When does a change actually take effect? Shown as a chip next to the field. */
export type ApplyMode = 'live' | 'next-turn' | 'mesh-restart' | 'startup'

export const APPLY_LABEL: Record<ApplyMode, string> = {
  live: 'applies immediately',
  'next-turn': 'applies on the next agent turn',
  'mesh-restart': 'needs a mesh restart',
  startup: 'applies at next server start',
}

export interface SettingMeta {
  key: string
  label: string
  /** One line: what happens if I change this. */
  effect: string
  unit?: string
  /** Safe default, as the user would type it. Empty string = provider default. */
  safeDefault: string
  apply: ApplyMode
  /** Returns an error message, or null when the raw input is acceptable. */
  validate: (raw: string) => string | null
}

/** A number that must parse finite - the Q14 family of breakage starts with NaN. */
export function finiteNumber(raw: string, opts: {
  min?: number; max?: number; integer?: boolean; name?: string
} = {}): string | null {
  const s = raw.trim()
  if (s === '') return null // empty means "use the default", always valid
  const n = Number(s)
  if (!Number.isFinite(n)) return `${opts.name ?? 'value'} must be a number`
  if (opts.integer && !Number.isInteger(n)) return `${opts.name ?? 'value'} must be a whole number`
  if (opts.min !== undefined && n < opts.min) return `minimum is ${opts.min}`
  if (opts.max !== undefined && n > opts.max) return `maximum is ${opts.max}`
  return null
}

/** Comma-separated zenoh endpoints, e.g. "tls/robot.lan:7447". */
export function endpointList(raw: string): string | null {
  const s = raw.trim()
  if (s === '') return null
  for (const part of s.split(',')) {
    const ep = part.trim()
    if (ep === '') continue
    // shape: proto/host:port  (proto one of tcp,tls,quic,udp)
    const m = ep.match(/^(tcp|tls|quic|udp)\/([^/:\s]+):(\d{1,5})$/)
    if (!m) return `"${ep}" is not proto/host:port (e.g. tls/robot.lan:7447)`
    const port = Number(m[3])
    if (port < 1 || port > 65535) return `"${ep}" has an out-of-range port`
  }
  return null
}

export const SETTINGS: SettingMeta[] = [
  {
    key: 'agent.temperature',
    label: 'Temperature',
    effect: 'Higher = more varied agent replies; lower = more deterministic. Empty uses the model default.',
    safeDefault: '',
    apply: 'next-turn',
    validate: raw => finiteNumber(raw, { min: 0, max: 2, name: 'temperature' }),
  },
  {
    key: 'agent.max_tokens',
    label: 'Max tokens',
    effect: 'Caps the length of one agent reply. Too low truncates tool use mid-thought.',
    unit: 'tokens',
    safeDefault: '',
    apply: 'next-turn',
    validate: raw => finiteNumber(raw, { min: 1, max: 200000, integer: true, name: 'max tokens' }),
  },
  {
    key: 'agent.model_id',
    label: 'Model id',
    effect: 'Which model answers chat and fleet commands. Empty uses the provider default.',
    safeDefault: '',
    apply: 'next-turn',
    validate: () => null,
  },
  {
    key: 'mesh.port',
    label: 'Mesh port',
    effect: 'UDP/TCP port the zenoh mesh binds. Every robot on the desk must agree on it.',
    unit: 'port',
    safeDefault: '7447',
    apply: 'mesh-restart',
    validate: raw => finiteNumber(raw, { min: 1, max: 65535, integer: true, name: 'port' }),
  },
  {
    key: 'mesh.camera_hz',
    label: 'Camera rate',
    effect: 'How many frames per second each robot publishes. Higher is smoother but costs LAN bandwidth.',
    unit: 'Hz',
    safeDefault: '5',
    apply: 'mesh-restart',
    validate: raw => finiteNumber(raw, { min: 0.1, max: 60, name: 'camera rate' }),
  },
  {
    key: 'mesh.connect',
    label: 'Connect endpoints',
    effect: 'Other mesh routers this dashboard dials out to. Empty relies on multicast discovery.',
    safeDefault: '',
    apply: 'mesh-restart',
    validate: endpointList,
  },
  {
    key: 'mesh.listen',
    label: 'Listen endpoints',
    effect: 'Addresses this dashboard accepts mesh connections on. Empty uses zenoh defaults.',
    safeDefault: '',
    apply: 'mesh-restart',
    validate: endpointList,
  },
]

const byKey = new Map(SETTINGS.map(s => [s.key, s]))
export const settingMeta = (key: string): SettingMeta | undefined => byKey.get(key)

/** Validate one field by key; unknown keys are permissive (never block save on missing metadata). */
export function validateSetting(key: string, raw: string): string | null {
  return byKey.get(key)?.validate(raw) ?? null
}

/** True when every entry in {key: rawValue} passes - the save button's enabled state. */
export function allValid(drafts: Record<string, string>): boolean {
  return Object.entries(drafts).every(([k, v]) => validateSetting(k, v) === null)
}
