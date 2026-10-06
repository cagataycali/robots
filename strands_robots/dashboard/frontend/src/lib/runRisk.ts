/**
 * Is pressing ▶ about to move METAL, or pixels? JOURNEYS.md #3: with a policy selected, typing
 * a sentence enabled ▶, and the 4th click moved a real arm with zero confirmation — no dialog,
 * no mention of the word "physical" anywhere in the app.
 */
import type { Peer, Presence } from '../types'

export type RunRisk = {
  /** True when the run is expected to drive physical hardware. */
  physical: boolean
  /** Short reason, shown to the operator so the judgment is auditable. */
  reason: string
  /** The hardware's own name for itself, when it gave one. */
  device: string | null
}

/** Where the server filed the presence record from, and whether it could check a sim claim. */
export type PresenceProvenance = Pick<Peer, 'presence_source' | 'sim_corroborated'>

/**
 * Errs toward "physical". A peer whose nature we cannot establish gets the confirm sheet: a
 * needless dialog costs one click, a missing one costs a collision.
 *
 * The browser twin of the server's `agent_motion.peer_is_physical`, in its order: hardware
 * first, then the sim claim. A presence record is the peer's own description of itself, so a
 * sim claim the server filed from the wire counts only when the server corroborated it (this
 * dashboard launched that peer as a sim); anyone on the mesh could have written it otherwise.
 */
export function runRisk(presence?: Presence | null, provenance?: PresenceProvenance | null): RunRisk {
  const hw = typeof presence?.hw === 'string' ? presence.hw.trim() : ''
  const type = String(presence?.robot_type ?? '').toLowerCase()
  const simHw = /^(sim|mock|fake|mujoco)/i.test(hw)

  if (hw && !simHw) {
    return { physical: true, reason: `real hardware attached (${hw})`, device: hw }
  }
  if (type === 'sim' || simHw) {
    if (provenance?.presence_source === 'wire' && provenance.sim_corroborated !== true) {
      return {
        physical: true,
        reason: 'it says it is simulated, but this dashboard did not launch it, so the claim cannot be checked',
        device: hw || null,
      }
    }
    return hw
      ? { physical: false, reason: `simulated backend (${hw})`, device: hw }
      : { physical: false, reason: 'simulated robot — nothing physical moves', device: null }
  }
  if (presence?.connected === false) {
    // Online peer, hardware disconnected: the run will fail rather than move.
    // Still not treated as safe — it may reconnect between judgment and click.
    return { physical: true, reason: 'hardware is not connected right now', device: null }
  }
  return { physical: true, reason: 'this peer did not say whether it is real', device: null }
}
