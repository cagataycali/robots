import assert from 'node:assert/strict'

const { mergeMeshEvent, sweepStale, rebaseSnapshotPeers, PEER_STALE_S } = await import('/tmp/meshPeers.mjs')

const NOW = 1_700_000_000

// ── 1. presence/state/stream vouch for a peer: they exist only because it published ──
for (const type of ['presence', 'state', 'stream']) {
  const out = mergeMeshEvent({}, { type, peer_id: 'arm-1', data: { x: 1 } }, NOW)
  assert.equal(out['arm-1'].peer_id, 'arm-1', `${type} creates an unknown peer`)
  assert.deepEqual(out['arm-1'][type], { x: 1 }, `${type} lands under its own key`)
  assert.equal(out['arm-1'].last_seen, NOW, `${type} refreshes last_seen`)
  assert.equal(out['arm-1'].stale, false, `${type} clears stale`)
}
// The other fields of a known peer survive: a state frame must not erase its presence block.
const kept = mergeMeshEvent(
  { 'arm-1': { peer_id: 'arm-1', presence: { role: 'follower' }, cameras: { main: { t: 1 } } } },
  { type: 'state', peer_id: 'arm-1', data: { joints: {} } }, NOW)
assert.deepEqual(kept['arm-1'].presence, { role: 'follower' }, 'presence survives a state event')
assert.deepEqual(kept['arm-1'].cameras, { main: { t: 1 } }, 'cameras survive a state event')

// ── 2.
const dead = { 'arm-1': { peer_id: 'arm-1', last_seen: NOW - 3600, stale: true } }
const replayed = mergeMeshEvent(dead, { type: 'camera_meta', peer_id: 'arm-1', cam: 'main', data: { t: NOW - 3600 } }, NOW)
assert.equal(replayed['arm-1'].last_seen, NOW - 3600, 'an HOUR-OLD cached frame does not refresh last_seen')
assert.equal(replayed['arm-1'].stale, true, 'and it does not un-stale the card — the peer is still dead')
assert.deepEqual(replayed['arm-1'].cameras.main, { t: NOW - 3600 }, 'the frame is still shown; it just proves nothing')

const live = mergeMeshEvent(dead, { type: 'camera_meta', peer_id: 'arm-1', cam: 'main', data: { t: NOW - 0.2 } }, NOW)
assert.equal(live['arm-1'].last_seen, NOW, 'a FRESH capture does vouch for the peer')
assert.equal(live['arm-1'].stale, false, 'and clears the badge')

// ── 3. THE CLOCK BOUNDARY (fixed here): a snapshot's last_seen is the SERVER's time ── Every
// other path stamps Date.now(); only the snapshot carries server timestamps.
const snapshot = {
  type: 'snapshot', t: 5_000,                       // server clock
  peers: { quiet: { peer_id: 'quiet', last_seen: 4_997 }, gone: { peer_id: 'gone', last_seen: 4_900 } },
}
// A browser 30s AHEAD of the Mac — ordinary skew for a laptop on the other side of the tunnel.
const skewed = mergeMeshEvent({}, snapshot, 5_030)
assert.equal(skewed.quiet.last_seen, 5_027, 'AGE is preserved (3s old), not the absolute timestamp')
assert.equal(skewed.gone.last_seen, 4_930, 'a 100s-old peer stays 100s old')
const sweptSkewed = sweepStale(skewed, 5_030)
assert.equal(sweptSkewed.quiet.stale, false,
             'a peer 3s old must NOT be marked stale just because this browser runs 30s ahead of the ' +
             'server — without the rebase the whole fleet went stale the moment a remote browser connected')
assert.equal(sweptSkewed.gone.stale, true, 'and a genuinely 100s-old peer is still called out')

// Skew the other way must not vouch for the dead: a peer whose last_seen would land in the FUTURE.
const behind = rebaseSnapshotPeers({ x: { peer_id: 'x', last_seen: 5_050 } }, 5_000, 9_000)
assert.equal(behind.x.last_seen, 9_000, 'a future-dated last_seen is clamped to now, never left in the future')
assert.equal(sweepStale(behind, 9_100).x.stale, true, 'so it can still go stale — a future timestamp never could')

// An older bridge sends no `t`. Guessing an age would be inventing evidence.
const noT = rebaseSnapshotPeers({ x: { peer_id: 'x', last_seen: 4_000 } }, undefined, 9_000)
assert.equal(noT.x.last_seen, 4_000, 'without a server clock the values are left exactly as sent')
assert.equal(rebaseSnapshotPeers({ x: { peer_id: 'x' } }, 5_000, 9_000).x.last_seen, undefined,
             'a peer with no last_seen is not given one')

// ── 4. a snapshot REPLACES the fleet; a re-pointed mesh empties it ──
const replaced = mergeMeshEvent({ old: { peer_id: 'old' } }, { type: 'snapshot', t: 5_000, peers: { neu: { peer_id: 'neu' } } }, NOW)
assert.deepEqual(Object.keys(replaced), ['neu'], 'the server snapshot is the truth, not a merge onto local memory')
assert.deepEqual(mergeMeshEvent({ old: { peer_id: 'old' } }, { type: 'mesh_reconfigured' }, NOW), {},
                 'a re-pointed session drops the old peer list rather than showing ghosts')

// ── 5. events that carry no peer, and unknown types, change nothing (identity, so no re-render) ──
const before = { 'arm-1': { peer_id: 'arm-1', last_seen: NOW, stale: false } }
assert.equal(mergeMeshEvent(before, { type: 'state', data: {} }, NOW), before, 'no peer_id = no change')
assert.equal(mergeMeshEvent(before, { type: 'safety', kind: 'estop' }, NOW), before, 'safety is not a peer event')
assert.equal(mergeMeshEvent(before, undefined, NOW), before, 'a malformed frame does not throw')

assert.equal(sweepStale(before, NOW + 1), before,
             'nothing changed = the SAME object, or every card in the grid re-renders every 5s forever')
// A peer whose `stale` key is absent (fresh from a snapshot) DOES change on the first sweep:
// undefined becomes an explicit boolean.
const unset = { x: { peer_id: 'x', last_seen: NOW } }
const normalised = sweepStale(unset, NOW + 1)
assert.notEqual(normalised, unset, 'an absent stale key is filled in once')
assert.equal(normalised.x.stale, false, 'and the filled value is measured, not assumed')
assert.equal(sweepStale(normalised, NOW + 2), normalised, 'the sweep after that is a no-op')
assert.equal(sweepStale(before, NOW + PEER_STALE_S + 1)['arm-1'].stale, true, `past ${PEER_STALE_S}s it is stale`)
assert.equal(sweepStale(before, NOW + PEER_STALE_S - 1)['arm-1'].stale, false, 'just inside the window it is not')
// A peer that has never been seen at all is stale, not fresh: `?? 0` must not read as "now".
assert.equal(sweepStale({ x: { peer_id: 'x' } }, NOW).x.stale, true, 'no last_seen at all = stale')
// Recovery is symmetric — a stale peer that publishes again goes green.
assert.equal(sweepStale({ x: { peer_id: 'x', last_seen: NOW, stale: true } }, NOW).x.stale, false, 'staleness clears')

console.log('meshPeers.test.mjs: all assertions passed')

const empty = {}
assert.equal(mergeMeshEvent(empty, { type: 'camera_meta', peer_id: 'ghost-1', cam: 'main', data: { t: NOW - 3600 } }, NOW),
             empty, 'an unknown peer is not created by a cached frame — and identity is returned, so no render')

assert.equal(mergeMeshEvent(empty, { type: 'camera_meta', peer_id: 'ghost-1', cam: 'main', data: { t: NOW - 0.1 } }, NOW),
             empty, 'nor by a fresh one — a peer that is really live announces itself on presence within ~1s')

// THE REACHABLE PATH: mesh_reconfigured empties the map on purpose, and cached frames keep
// arriving for the peers of the mesh we just left.
const cleared = mergeMeshEvent({ 'old-arm': { peer_id: 'old-arm', last_seen: NOW } }, { type: 'mesh_reconfigured' }, NOW)
assert.deepEqual(cleared, {}, 'the old mesh is gone')
assert.deepEqual(mergeMeshEvent(cleared, { type: 'camera_meta', peer_id: 'old-arm', cam: 'main', data: { t: NOW } }, NOW),
                 {}, 'and a replayed frame from it cannot bring a card back')

// ...while annotating a peer that DOES exist still works exactly as before.
const known = { 'arm-1': { peer_id: 'arm-1', last_seen: NOW - 1, stale: false } }
const annotated = mergeMeshEvent(known, { type: 'camera_meta', peer_id: 'arm-1', cam: 'wrist', data: { t: NOW } }, NOW)
assert.deepEqual(annotated['arm-1'].cameras.wrist, { t: NOW }, 'the camera still lands on a real peer')

assert.equal(sweepStale({ x: { peer_id: 'x' } }, NOW).x.stale, true,
             'unknown age keeps the conservative verdict, not a green card')

console.log('meshPeers: Q94 conjured-peer assertions ok')
