import assert from 'node:assert/strict'
import { shouldCheckForUpdate, bundleAgeText, SW_UPDATE_INTERVAL_MS, reloadImpact } from '/tmp/swUpdate.mjs'

const NOW = 1_787_200_000_000
const base = { nowMs: NOW, online: true, visible: true, reason: 'interval' }

// --- the incident: a page open for 11 hours had never re-checked ---------------
assert.equal(shouldCheckForUpdate({ ...base, lastCheckedAt: NOW - 11 * 3600_000 }), true)
assert.equal(shouldCheckForUpdate({ ...base, lastCheckedAt: null }), true,
  'never checked since load is exactly the state that stranded the phone')

// --- but not a request per tick -----------------------------------------------
assert.equal(shouldCheckForUpdate({ ...base, lastCheckedAt: NOW - 60_000 }), false)
assert.equal(shouldCheckForUpdate({ ...base, lastCheckedAt: NOW - SW_UPDATE_INTERVAL_MS + 1 }), false)
assert.equal(shouldCheckForUpdate({ ...base, lastCheckedAt: NOW - SW_UPDATE_INTERVAL_MS }), true)

// --- foregrounding is a reason to ask, but does not bypass the interval --------
assert.equal(shouldCheckForUpdate({ ...base, reason: 'visible', lastCheckedAt: NOW - 30_000 }), false,
  'app-switching on a phone must not become a request per switch')
assert.equal(shouldCheckForUpdate({ ...base, reason: 'visible', lastCheckedAt: NOW - 3600_000 }), true)

// --- a phone's realities ------------------------------------------------------
assert.equal(shouldCheckForUpdate({ ...base, online: false, lastCheckedAt: null }), false,
  'a check that cannot succeed is battery and noise')
assert.equal(shouldCheckForUpdate({ ...base, visible: false, lastCheckedAt: null }), false,
  'hidden tabs are throttled; asking there just queues work for later')

// --- registration always checks (it is the baseline) --------------------------
assert.equal(shouldCheckForUpdate({ ...base, reason: 'registered', lastCheckedAt: NOW, online: false, visible: false }), true)

// --- a clock that moved backwards cannot wedge this either way ----------------
assert.equal(shouldCheckForUpdate({ ...base, lastCheckedAt: NOW + 3600_000 }), true,
  'a future last-check must not block updates forever')

// --- how old the running bundle is, in human words ---------------------------
assert.equal(bundleAgeText(NOW - 5_000, NOW), 'just now')
assert.equal(bundleAgeText(NOW - 600_000, NOW), '10m ago')
assert.equal(bundleAgeText(NOW - 11 * 3600_000, NOW), '11.0h ago')
assert.equal(bundleAgeText(NOW - 3 * 86400_000, NOW), '3d ago')
assert.equal(bundleAgeText(null, NOW), null, 'unknown stays unknown, never "just now"')
assert.equal(bundleAgeText(NOW + 10_000, NOW), null)

console.log('swUpdate: all assertions passed')

// ── Q97: the toast must say what reloading costs RIGHT NOW ──
// Auto-update is refused because a reload mid-task tears down camera sockets and the run form of a
// moving robot. The manual prompt inherited that hazard and described it with ONE STATIC SENTENCE,
// printed identically at the safest moment and the worst one — so the operator, who was deliberately
// given the decision, was given nothing to decide with.
const quiet = reloadImpact([])
assert.equal(quiet.busy, false)
assert.match(quiet.text, /good moment/, 'when nothing runs, say so — that is the useful news')
assert.doesNotMatch(quiet.text, /keeps running/, 'nothing is running, so nothing can keep running')

const one = reloadImpact(['so101-follower'])
assert.equal(one.busy, true)
assert.match(one.text, /^so101-follower is running/, 'name it, and agree with itself on number')
assert.match(one.text, /keeps running on the robot/, 'the task survives — it runs in the robot process')
assert.match(one.text, /camera streams and anything typed into a form/, 'what actually dies says so')
assert.doesNotMatch(one.text, /good moment/, 'and it never calls this a good moment')

assert.match(reloadImpact(['a', 'b']).text, /^a and b are running/, 'two are named in full')
assert.match(reloadImpact(['a', 'b', 'c', 'd']).text, /^a, b and 2 more are running/,
             'past two it counts, so the toast cannot grow taller than the screen')

// Peer ids arrive from a busy MAP whose values were toggled, so blanks and repeats are ordinary.
assert.equal(reloadImpact(['so101-follower', 'so101-follower']).text, one.text, 'deduped')
assert.equal(reloadImpact(['', '  ', null, undefined]).busy, false,
             'a map of empty keys is not a running fleet — it must not claim a robot is moving')
assert.match(reloadImpact([' so101-leader ']).text, /^so101-leader is running/, 'trimmed')

console.log('swUpdate: Q97 reload-impact assertions ok')
