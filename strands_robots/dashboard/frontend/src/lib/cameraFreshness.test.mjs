import assert from 'node:assert/strict'
import {
  captureAge, stoppedCameras, cameraWarning, agoText, CAMERA_STOPPED_AGE_S,
} from '/tmp/cameraFreshness.mjs'

const NOW = 1_787_200_000 // seconds

// --- the incident, in the numbers it was measured with -----------------------
// so101-arm-1: `top` publishing at 4fps, `wrist` captured 37327s (10.4h) earlier.
{
  const stopped = stoppedCameras({ top: { t: NOW - 0.2 }, wrist: { t: NOW - 37327 } }, NOW)
  assert.equal(stopped.length, 1)
  assert.equal(stopped[0].camera, 'wrist')
  const msg = cameraWarning(stopped, { peerId: 'so101-arm-1' })
  assert.match(msg, /so101-arm-1/)
  assert.match(msg, /wrist \(last frame 10\.4h ago\)/)
  assert.match(msg, /frozen or missing/, 'state the consequence, not just the fact')
  assert.match(msg, /training time/)
}

// --- silence never accuses ---------------------------------------------------
assert.deepEqual(stoppedCameras({ top: {}, wrist: { t: null } }, NOW), [],
  'no capture time is not death: nothing may have subscribed yet')
assert.deepEqual(stoppedCameras(undefined, NOW), [])
assert.deepEqual(stoppedCameras({}, NOW), [])
assert.equal(cameraWarning([]), null, 'nothing to say means say nothing')

// --- a normal gap between frames is not death -------------------------------
assert.deepEqual(stoppedCameras({ top: { t: NOW - 30 } }, NOW), [])
assert.deepEqual(stoppedCameras({ top: { t: NOW - CAMERA_STOPPED_AGE_S } }, NOW), [],
  'the threshold itself is still alive')
assert.equal(stoppedCameras({ top: { t: NOW - CAMERA_STOPPED_AGE_S - 1 } }, NOW).length, 1)

// --- a capture from the future is clock skew, not freshness ------------------
assert.equal(captureAge({ t: NOW + 900 }, NOW), null)
assert.deepEqual(stoppedCameras({ top: { t: NOW + 900 } }, NOW), [],
  'two machines disagreeing about the time must not condemn a camera')
assert.equal(captureAge({ t: 0 }, NOW), null)
assert.equal(captureAge({ t: NaN }, NOW), null)
assert.equal(captureAge({ t: 'yesterday' }, NOW), null)

// --- the worst offender leads ----------------------------------------------
{
  const stopped = stoppedCameras({ a: { t: NOW - 200 }, b: { t: NOW - 9000 } }, NOW)
  assert.deepEqual(stopped.map(c => c.camera), ['b', 'a'])
  const msg = cameraWarning(stopped)
  assert.match(msg, /2 cameras have stopped/)
  assert.ok(msg.indexOf('b (') < msg.indexOf('a ('))
  assert.ok(!msg.startsWith(':'), 'no peer id means no orphaned colon')
}

// --- ages read the way a human says them -----------------------------------
assert.equal(agoText(6), '6s ago')
assert.equal(agoText(600), '10m ago')
assert.equal(agoText(37327), '10.4h ago')

console.log('cameraFreshness: all assertions passed')
