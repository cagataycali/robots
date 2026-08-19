import assert from 'node:assert/strict'
import { frameProvesLiveness, FRAME_LIVENESS_MAX_AGE_S } from '/tmp/liveness.mjs'

const NOW = 1_787_180_000

// --- THE DEFECT ------------------------------------------------------------
// The dashboard replays a camera's last cached frame to every new subscriber, so
// mounting a tile delivered a camera event for a frame captured 6.8h earlier
// (measured on so101-arm-1's wrist). That event refreshed last_seen and cleared
// `stale`, i.e. opening a page could make a robot that died hours ago read
// "seen 0s ago" with a live dot - the one number an operator uses to decide
// whether anything they see on the card is worth believing.
assert.equal(frameProvesLiveness({ frameT: NOW - 24_307, nowS: NOW }), false,
  'a 6.8h-old capture must not vouch for the peer')

// A genuinely fresh frame still does, so a working camera keeps a peer alive
// between state messages.
assert.equal(frameProvesLiveness({ frameT: NOW - 0.2, nowS: NOW }), true)
assert.equal(frameProvesLiveness({ frameT: NOW, nowS: NOW }), true)

// The boundary is the stale window itself: inside it a frame is as good as a
// heartbeat, one second past it is not.
assert.equal(frameProvesLiveness({ frameT: NOW - (FRAME_LIVENESS_MAX_AGE_S - 1), nowS: NOW }), true)
assert.equal(frameProvesLiveness({ frameT: NOW - FRAME_LIVENESS_MAX_AGE_S, nowS: NOW }), true)
assert.equal(frameProvesLiveness({ frameT: NOW - (FRAME_LIVENESS_MAX_AGE_S + 1), nowS: NOW }), false)
assert.equal(frameProvesLiveness({ frameT: NOW - 5, nowS: NOW, maxAgeS: 2 }), false, 'window is injectable')

// No capture time: unknowable, so it does not get to vouch. Presence and state
// events refresh liveness on their own, so this loses nothing real.
for (const bad of [undefined, null, 0, -1, NaN, Infinity]) {
  assert.equal(frameProvesLiveness({ frameT: bad, nowS: NOW }), false, `frameT=${bad}`)
}

// A capture stamped in the FUTURE is skew between two machines, not freshness -
// and not evidence of death either, so it is counted neither way.
assert.equal(frameProvesLiveness({ frameT: NOW + 600, nowS: NOW }), false)

// The function is a judgment about ONE frame and says nothing about the peer's
// other rails: it returns a plain boolean, never mutates its input.
const input = { frameT: NOW - 1, nowS: NOW }
frameProvesLiveness(input)
assert.deepEqual(input, { frameT: NOW - 1, nowS: NOW })

console.log('liveness: 16 assertions ok')
