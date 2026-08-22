/** How a mesh event becomes the fleet's peer map — as pure functions. */
import { frameProvesLiveness } from './liveness'
import type { Peer } from '../types'

/** How a mesh event becomes the fleet's peer map — as pure functions. */

/** Seconds without an event before a peer's card is marked stale. Matches the server's
 *  PEER_STALE_S: two different thresholds would let a card contradict /api/health. */
export const PEER_STALE_S = 15

export function sweepStale(peers: Record<string, Peer>, nowS: number): Record<string, Peer> {
  let changed = false
  const next = { ...peers }
  for (const [id, peer] of Object.entries(next)) {
    const stale = nowS - (peer.last_seen ?? 0) > PEER_STALE_S
    if (stale !== peer.stale) { next[id] = { ...peer, stale }; changed = true }
  }
  // Identity is the caller's "nothing happened" signal: returning a fresh object every 5s
  // would re-render every card in the grid forever.
  return changed ? next : peers
}

/**
 * Translate a snapshot's peers from the SERVER's clock into this browser's, preserving each
 * peer's AGE rather than its absolute timestamp.
 */
export function rebaseSnapshotPeers(
  peers: Record<string, Peer>, serverNowS: number | undefined, nowS: number,
): Record<string, Peer> {
  if (!serverNowS) return peers
  const out: Record<string, Peer> = {}
  for (const [id, peer] of Object.entries(peers)) {
    if (peer.last_seen === undefined) { out[id] = peer; continue }
    // A future-dated last_seen (skew the other way, or a peer stamped mid-request) is clamped to
    // "just now" rather than allowed to sit in the future, where it could never go stale.
    const ageS = Math.max(0, serverNowS - peer.last_seen)
    out[id] = { ...peer, last_seen: nowS - ageS }
  }
  return out
}

/**
 * Fold one peer-bearing mesh event into the peer map. Returns the SAME object when the event
 * changes nothing, so the caller can skip a render.
 */
export function mergeMeshEvent(
  peers: Record<string, Peer>, ev: any, nowS: number,
): Record<string, Peer> {
  const id = ev?.peer_id
  switch (ev?.type) {
    case 'snapshot':
      return rebaseSnapshotPeers(ev.peers ?? {}, ev.t, nowS)
    case 'mesh_reconfigured':
      // The session was re-pointed under us: the old peer list belongs to the old mesh, so drop
      // it rather than show ghosts.
      return {}
    case 'presence':
    case 'state':
    case 'stream': {
      if (!id) return peers
      const key = ev.type === 'presence' ? 'presence' : ev.type === 'state' ? 'state' : 'stream'
      return { ...peers, [id]: { ...peers[id], peer_id: id, [key]: ev.data, last_seen: nowS, stale: false } }
    }
    case 'camera_meta': {
      if (!id) return peers
      const peer = peers[id]
      if (!peer) return peers
      const cameras = { ...peer.cameras, [ev.cam]: ev.data }
      const fresh = frameProvesLiveness({ frameT: ev.data?.t, nowS })
      return {
        ...peers,
        [id]: fresh ? { ...peer, cameras, last_seen: nowS, stale: false } : { ...peer, cameras },
      }
    }
    default:
      return peers
  }
}
