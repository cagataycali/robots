import assert from 'node:assert/strict'
import { cameraEvidence } from '/tmp/cameraEvidence.mjs'

// --- frames present: nothing to explain ---------------------------------------
{
  const v = cameraEvidence('so101-arm-1', ['top', 'wrist'], ['top', 'wrist'])
  assert.equal(v.kind, 'ok')
  assert.deepEqual(v.cams, ['top', 'wrist'])
}

// --- THE INCIDENT (BUGS.md Q25): announced two, macOS blocked every frame -----
{
  const v = cameraEvidence('so101-arm-1', ['top', 'wrist'], [])
  assert.equal(v.kind, 'mute')
  assert.match(v.message, /announces 2 cameras \(top, wrist\)/, 'never deny what presence announced')
  assert.match(v.message, /no frames have arrived/)
  assert.match(v.message, /blocked by macOS|another process|unplugged/, 'name the causes, pick none')
  assert.match(v.message, /devices › logs/, 'point at the place that answers')
  assert.doesNotMatch(v.message, /no cameras announced/, 'the old lie must not come back')
  assert.doesNotMatch(v.message, /will (capture|have)/, 'a recording has not started: stay conditional')
}

// --- singular reads as English ------------------------------------------------
{
  const v = cameraEvidence('so101-arm-2', ['wrist'], undefined)
  assert.match(v.message, /announces 1 camera \(wrist\)/)
}

// --- the robot itself lists none: still not proof of intent -------------------
{
  const v = cameraEvidence('sim-a', [], [])
  assert.equal(v.kind, 'unannounced')
  assert.match(v.message, /lists no cameras/)
  assert.match(v.message, /deliberately joints-only|dropped when it connected/)
  assert.match(v.message, /indistinguishable/, 'admit the ambiguity instead of choosing')
  assert.doesNotMatch(v.message, /will have joints only/)
}

// --- absent fields are not a third state --------------------------------------
{
  assert.equal(cameraEvidence('p', undefined, undefined).kind, 'unannounced')
  assert.equal(cameraEvidence('p', ['top'], ['top']).kind, 'ok')
  // a frame from a camera presence never mentioned is still a frame: draw it.
  const v = cameraEvidence('p', [], ['side'])
  assert.equal(v.kind, 'ok')
  assert.deepEqual(v.cams, ['side'])
}

console.log('cameraEvidence: all assertions passed')
