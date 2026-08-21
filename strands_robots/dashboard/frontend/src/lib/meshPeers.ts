import { frameProvesLiveness } from './liveness'
import type { Peer } from '../types'

/**
 * How a mesh event becomes the fleet's peer map — as pure functions.
 *
 * This lived inside useMesh's socket handler, so the rules that decide whether a
 * robot's card is GREEN or shown as dead could only be exercised by opening a
 * websocket in a browser. Nothing tested them.
 */

/** Seconds without an event before a peer's card is marked stale. Matches the server's
 *  PEER_STALE_S: two different thresholds would let a card contradict /api/health. */
export const PEER_STALE_S = 15

/**
 * A peer's staleness is measured from ARRIVAL on this socket, by design: the bridge sends an
 * unchanged event as a liveness tick precisely so the client can do that (suppressing repeats
 * outright would paint an idle peer — one that only publishes presence — as dead).
 */
export function sweepStale(peers: Record<string, Peer>, nowS: number): Record<string, Peer> {
  let changed = false
  const next = { ...peers }
  for (const [id, peer] of Object.entries(next)) {
    // A MISSING TIMESTAMP STAYS STALE, deliberately (considered and rejected in Q94): `stale` is a
    // boolean, so the only alternative to "stale" is "live", and rendering a peer whose age is
    // unknown as a green card is the worse error for a machine that can move. `?? 0` is not a
    // measurement, but it lands on the safe side of one. After Q94 the only way a peer reaches this
    // sweep without a timestamp is a snapshot the server sent that way.
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
 *
 * Every other path stamps `last_seen` with Date.now(), but a snapshot's `last_seen` values are
 * the server's `time.time()`. Comparing those against a browser clock is comparing two
 * unsynchronised clocks: a browser 30s ahead of the Mac (an ordinary amount of skew for a
 * laptop on the other side of the tunnel) marks the ENTIRE fleet stale the moment it connects,
 * and a browser behind vouches for peers that are already dead. Live peers hide it — their next
 * presence tick (~1 Hz) rewrites last_seen in browser time — so what remains visible is exactly
 * the quiet peer and the genuinely dead one, i.e. the two cases the badge exists to tell apart.
 *
 * Age is clock-independent, so it is the only thing worth carrying across the boundary. When the
 * snapshot has no server timestamp (an older bridge), ages are left alone: inventing one would be
 * a guess, and the sweep's 15s window plus the next event corrects it either way.
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
 *
 * `presence`/`state`/`stream` vouch for the peer: they only exist because the peer published.
 * `camera_meta` does NOT, on its own — the bridge replays a camera's last cached frame to every
 * new subscriber, so mounting a tile used to resurrect a peer that died hours ago (last_seen
 * refreshed, stale cleared, card green). The frame's own capture time settles it.
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
      // AN ANNOTATION MUST NEVER CONJURE A PEER INTO THE FLEET (Q94) - the same rule mesh_bridge.py
      // states for its own server-side annotations. A camera frame is not evidence that a robot
      // exists: the bridge replays each camera's last cached frame to every new subscriber, so a
      // frame for a peer this client has never heard of (or one just cleared by mesh_reconfigured)
      // used to CREATE a card - with no last_seen, which the sweep then read as ancient and rendered
      // as "no heartbeat, treat the arm as unpredictable". A robot invented out of a cached JPEG.
      // A peer that really is live announces itself on presence within ~1s and the next frame lands.
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
