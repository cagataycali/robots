/**
 * The refusals a record session can be continued past — and the ones it cannot.
 *
 * Every camera gate at /api/record/open is continuable by design; this pins that
 * the continuation is REACHABLE (the flag the server names becomes a tick) and
 * that the allowlist keeps a bypass from appearing where nobody designed one.
 */
import assert from 'node:assert/strict'
import { overrideOffered, overrideBody } from '/tmp/recordRefusal.mjs'

// Real server sentences, not paraphrases: this module reads what the backend writes.
const STALE =
  "so101-arm-1: 1 configured camera stopped publishing - wrist (last frame 10.4h ago). " +
  "Pass ignore_dead_cameras to record anyway."
const GONE =
  "so101-arm-1: 1 configured camera is not listed by this machine at all - wrist (index 1). " +
  "Pass ignore_missing_cameras to record without it anyway."
const DRIFT =
  "so101-arm-1: 1 configured camera changed hands - wrist index 0 was USB2.0_CAM1, now Logi 4K Pro " +
  "(USB2.0_CAM1 is at index 1 now). Point the camera at its new index, or pass " +
  "ignore_camera_identity to record with the index as it stands."

for (const [msg, flag] of [
  [STALE, 'ignore_dead_cameras'],
  [GONE, 'ignore_missing_cameras'],
  [DRIFT, 'ignore_camera_identity'],
]) {
  const o = overrideOffered(msg)
  assert.equal(o?.flag, flag, `the refusal naming ${flag} must offer it`)
  assert.ok(o.label.length > 10 && o.cost.length > 10, 'a tick states the claim AND the cost')
}

// The identity override is the one that permits something VISIBLY working, so its
// cost line must say the consequence out loud rather than reassure.
assert.match(overrideOffered(DRIFT).cost, /WRONG view/)

// No flag named -> no tick. A bypass must never appear on a message nobody designed.
for (const msg of [
  'a recording session is already open - close it first',
  'no teleoperator type known for leader robot',
  'Failed to fetch',
  '',
  null,
  undefined,
  42,
  { detail: 'ignore_dead_cameras' },
]) {
  assert.equal(overrideOffered(msg), null)
}

// A word that merely looks like one is not one.
assert.equal(overrideOffered('pass ignore_dead_camera to proceed'), null)
assert.equal(overrideOffered('ignoring dead cameras is not an option'), null)

// TWO faults refused at once are two admissions; one box would collect consent
// for the one the operator did not read.
assert.equal(overrideOffered(`${STALE} ${DRIFT}`), null)

// The flag travels only with a deliberate tick, and never by default.
const drift = overrideOffered(DRIFT)
assert.deepEqual(overrideBody(drift, true), { ignore_camera_identity: true })
assert.deepEqual(overrideBody(drift, false), {})
assert.deepEqual(overrideBody(null, true), {})

console.log('recordRefusal: ok')
