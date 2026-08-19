import assert from 'node:assert/strict'
import { trainingFreshness, TRAINING_POLL_S, TRAINING_STALE_POLLS } from '/tmp/trainingFreshness.mjs'

const NOW = 1_787_180_000
const window = TRAINING_POLL_S * TRAINING_STALE_POLLS

// --- THE DEFECT: an unbounded `catch { /* transient */ }` --------------------
// The status poll's failures were swallowed forever, so a job whose provider
// process died kept rendering "running", a bar at 4.7k/10k and a loss sparkline
// unchanged for hours. Training is watched by someone who is NOT at a console:
// a frozen number reads as a healthy one.
const dead = trainingFreshness({ polledAtS: NOW - 600, nowS: NOW, failures: 118, error: 'HTTP 500', state: 'running' })
assert.equal(dead.stale, true)
assert.match(dead.note, /these numbers are 10m old/)
assert.match(dead.note, /the status feed stopped/)
assert.match(dead.note, /118 failed polls: HTTP 500/)
assert.match(dead.note, /may have died, finished, or moved on/)
assert.match(dead.title, /status read 10m ago/)

// Healthy feed: nothing new on screen, but the chip still carries its age.
const live = trainingFreshness({ polledAtS: NOW - 3, nowS: NOW, state: 'running' })
assert.equal(live.stale, false)
assert.equal(live.note, '', 'a working feed must not add noise')
assert.match(live.title, /status read 3s ago/)

// The boundary is three missed polls, not one: a single blip is genuinely
// transient and must not cry wolf.
assert.equal(trainingFreshness({ polledAtS: NOW - (window - 1), nowS: NOW, state: 'running' }).stale, false)
assert.equal(trainingFreshness({ polledAtS: NOW - (window + 1), nowS: NOW, state: 'running' }).stale, true)
assert.equal(trainingFreshness({ polledAtS: NOW - 60, nowS: NOW, state: 'running', staleAfterS: 300 }).stale, false,
  'window is injectable')

// A SETTLED job is not supposed to keep updating: success/failed/cancelled is a
// final answer, so its age is not a fault and raises no alarm hours later.
for (const state of ['success', 'failed', 'cancelled', 'SUCCESS']) {
  const done = trainingFreshness({ polledAtS: NOW - 7200, nowS: NOW, state })
  assert.equal(done.stale, false, `${state} must not be called stale`)
  assert.equal(done.note, '')
  assert.match(done.title, /final status, read 120m ago/)
}

// Never read at all: says so, and only alarms if reads have actually FAILED
// (a job submitted a second ago has simply not been polled yet).
const never = trainingFreshness({ polledAtS: null, nowS: NOW, state: 'running' })
assert.equal(never.ageS, null)
assert.equal(never.stale, false)
assert.equal(never.note, '')
assert.match(never.title, /no status has been read/)
const neverFailing = trainingFreshness({ polledAtS: null, nowS: NOW, failures: 4, error: 'not found', state: 'running' })
assert.equal(neverFailing.stale, true)
assert.match(neverFailing.note, /never read this job's status/)
assert.match(neverFailing.note, /4 status polls failed: not found/)
assert.match(trainingFreshness({ polledAtS: null, nowS: NOW, failures: 1, error: 'x' }).note, /1 status poll failed/)

// Junk inputs cannot produce NaN prose or a negative age (this renders in a
// long-lived tab, and a clock can jump).
for (const bad of [0, -5, NaN, Infinity, undefined]) {
  const v = trainingFreshness({ polledAtS: bad, nowS: NOW, state: 'running' })
  assert.equal(v.ageS, null, `polledAtS=${bad}`)
  assert.doesNotMatch(v.note + v.title, /NaN|Infinity|undefined/)
}
assert.equal(trainingFreshness({ polledAtS: NOW + 30, nowS: NOW, state: 'running' }).ageS, 0, 'clock skew clamps to 0')
assert.equal(trainingFreshness({ polledAtS: NOW + 30, nowS: NOW, state: 'running' }).stale, false)

console.log('trainingFreshness: 33 assertions ok')
