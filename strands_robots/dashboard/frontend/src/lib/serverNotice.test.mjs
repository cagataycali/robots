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
