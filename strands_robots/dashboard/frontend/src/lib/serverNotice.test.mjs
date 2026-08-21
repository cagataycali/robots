import assert from 'node:assert/strict'
import { serverNotice } from '/tmp/serverNotice.mjs'

// Silence is the default, and most of this file is about WHEN NOT to speak.
assert.equal(serverNotice(undefined).text, '', 'no block (a clean server) says nothing')
assert.equal(serverNotice(null).text, '')
assert.equal(serverNotice({}).text, '')

// A handful of refusals is somebody signing in. A banner for that teaches the operator to
// ignore banners — and this dashboard's banners guard an e-stop.
assert.equal(
  serverNotice({ total: 3, recent: 3, storm: false, worst: { client: '10.0.0.5', path: '/ws/mesh', count: 3 } }).text,
  '',
  'not a storm = not news',
)

// "It stopped" is the server's answer to "did my fix work?", not a thing to interrupt for.
assert.equal(serverNotice({ total: 400, recent: 0, storm: true }).text, '', 'a stale storm flag is trusted less than the count')

// The real case: a loop, named. The server's sentence is reused rather than re-derived — two
// sources of truth for one number is how they drift.
const storm = serverNotice({
  total: 812, recent: 240, window_s: 300, clients: 1, storm: true,
  worst: { client: '192.168.1.44', path: '/ws/camera/so101-leader/main', kind: 'credential', count: 240 },
  text: '192.168.1.44 is retrying /ws/camera/so101-leader/main and being refused (credential) 240 times in the last 5 minutes, ~48/min. It will not recover by itself: that page is holding an expired or wrong sign-in - reload it and sign in again. Nothing is wrong with the robots.',
})
assert.match(storm.text, /192\.168\.1\.44/)
assert.match(storm.text, /reload it and sign in again/)
assert.equal(storm.vague, false)
// What this layer adds is the one thing the server cannot know: who is reading it.
assert.match(storm.text, /Your own session is fine/, 'the reader is signed in — it must not read as "you are locked out"')
assert.match(storm.text, /another client/)
// And it must not blame the fleet: the whole point of Q88.
assert.match(storm.text, /Nothing is wrong with the robots/)

// An unauthenticated read of /api/health has the counts and no identities (dd658b47). Say the
// true, smaller thing rather than nothing.
const vague = serverNotice({ total: 40, recent: 40, window_s: 300, storm: true })
assert.match(vague.text, /Something is being refused/)
assert.match(vague.text, /40 handshake\(s\) in the last 5 minutes/)
assert.match(vague.text, /Your own session is fine/)
assert.equal(vague.vague, true)
// It cannot invent a culprit it was not told about.
assert.ok(!/192\.168|\/ws\//.test(vague.text))

// A storm whose text the server did not send (older backend, new frontend) still says something
// useful instead of rendering "undefined".
const noText = serverNotice({ recent: 99, storm: true, worst: { client: '10.1.1.7', count: 99 } })
assert.match(noText.text, /10\.1\.1\.7/)
assert.ok(!/undefined/.test(noText.text), 'a missing field must never reach the screen as the word "undefined"')

console.log('serverNotice: 6 groups ok — a loop is named, a handful and a stopped storm stay silent, and the reader is never told they are locked out')

// --- which build is answering (ec5aabb4), and the silence around it -----------------------------
// ADDED as a section, not a new file: serverNotice.ts already had tests and a sibling life once
// lost 31 assertions to a `cat >` on a file it assumed was empty (Q106).
import { serverPredatesBuildStamp, fleetFieldGaps, staleServerNotice } from '/tmp/serverNotice.mjs'

const OLD = { status: 'ok', t: 1787350000, peers: 4 } // the live Aug-19 server: no `build` key
const NEW = { status: 'ok', t: 1787350000, build: { commit: 'a2d7da05733f', version: null, started: 1 } }

// Not asked yet is not evidence. This is the assertion that keeps the banner off screen during
// every page load, which is how a truthful notice becomes a nag.
assert.equal(serverPredatesBuildStamp(undefined), false, 'not fetched yet says nothing')
assert.equal(serverPredatesBuildStamp(null), false)
assert.equal(serverPredatesBuildStamp({}), false, 'an empty object is not a health payload')
assert.equal(serverPredatesBuildStamp([]), false)
assert.equal(serverPredatesBuildStamp('ok'), false)
assert.equal(serverPredatesBuildStamp(OLD), true, 'a real health payload with no build IS old')
assert.equal(serverPredatesBuildStamp(NEW), false)

// A gap needs the WHOLE fleet to be silent: one annotated peer proves the server can annotate,
// so the others are just robots that differ.
assert.deepEqual(fleetFieldGaps([]), [], 'an empty fleet cannot be evidence')
assert.deepEqual(fleetFieldGaps(undefined), [])
assert.deepEqual(fleetFieldGaps([{ peer_id: 'a', origin: null }]), [],
  'present-but-null still means the server SENT the field')
assert.deepEqual(fleetFieldGaps([{ peer_id: 'a', origin: 'external' }, { peer_id: 'b' }]), [],
  'origin on ONE peer proves the server can annotate: the others just differ')
assert.deepEqual(fleetFieldGaps({ a: { peer_id: 'a' }, b: { peer_id: 'b' } }), [
  'which robots this dashboard started itself',
], 'a peers MAP is accepted, not just an array (App holds a record)')
// Measured on the live Aug-19 server: it OMITS the key entirely (not origin:null), which is what
// makes this evidence rather than a guess. A device-row field like `remembered` was deliberately
// NOT added to the table — peers never carry it on any version, so its absence proves nothing.

// The whole point: staleness alone is silent, a gap alone is silent, the two together speak once.
assert.equal(staleServerNotice(OLD, []).text, '', 'an old server with nothing missing is not news')
assert.equal(staleServerNotice(NEW, ['which robots this dashboard started itself']).text, '',
  'a CURRENT server missing a field must never be blamed on staleness — wrong remedy')
assert.equal(staleServerNotice(undefined, ['x']).text, '')
const spoke = staleServerNotice(OLD, fleetFieldGaps([{ peer_id: 'a' }, { peer_id: 'b' }]))
assert.match(spoke.text, /older than the code you are looking at/)
assert.match(spoke.text, /Restart the dashboard from a terminal/, 'one remedy, named')
assert.match(spoke.text, /can show which robots this dashboard started itself, but/,
  "the gap in the operator's words, not the field name")
assert.match(staleServerNotice(OLD, ['a', 'b', 'c']).text, /a, b and c/, 'a list reads as a list')
assert.equal(spoke.vague, false)
