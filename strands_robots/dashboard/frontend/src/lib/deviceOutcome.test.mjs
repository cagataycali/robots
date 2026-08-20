// Assertions for what the devices drawer may claim when spawn/despawn fails
// (lib/deviceOutcome.ts).
// Run: npx esbuild src/lib/deviceOutcome.ts --bundle --format=esm --outfile=/tmp/deviceOutcome.mjs \
//        && node src/lib/deviceOutcome.test.mjs
import assert from 'node:assert/strict'

const { deviceActionFailure } = await import('/tmp/deviceOutcome.mjs')

// THE BUG (spawn): "⚠ <message>" read as "nothing started", so the operator
// spawns again — a second process on the same servo bus.
{
  const v = deviceActionFailure({ kind: 'spawn', status: 0, message: 'cannot reach robots.cagatay.my' })
  assert.equal(v.ambiguous, true)
  assert.match(v.text, /MAY have started/)
  assert.match(v.text, /MAY already hold that port/)
  assert.match(v.text, /Refreshing the device list/)
  assert.match(v.text, /if it appears there, it started/)
  assert.match(v.text, /Port is in use/)
}

// THE BUG (despawn): the opposite lie — the robot may already be dead, and if it
// was recording, the take is gone.
{
  const v = deviceActionFailure({ kind: 'despawn', status: 0, message: 'timeout' })
  assert.equal(v.ambiguous, true)
  assert.match(v.text, /MAY already have been killed/)
  assert.match(v.text, /mid-episode, that take is gone/)
  assert.match(v.text, /if it disappears from it, the despawn landed/)
  assert.doesNotMatch(v.text, /Port is in use/)
}

// A 5xx ran the handler: still ambiguous, and the code stays visible.
for (const kind of ['spawn', 'despawn']) {
  const v = deviceActionFailure({ kind, status: 500, message: 'boom' })
  assert.equal(v.ambiguous, true)
  assert.match(v.text, /failed mid-request \(500: boom\)/)
}

// Only a pre-handler refusal may claim the world is untouched — and for despawn
// that claim has to be the ALARMING one: the robot is still running.
for (const status of [400, 401, 403, 404, 422, 429]) {
  const s = deviceActionFailure({ kind: 'spawn', status, message: 'needs consent' })
  assert.equal(s.ambiguous, false)
  assert.match(s.text, /no process was started/)
  assert.match(s.text, /nothing new is holding the serial port/)

  const d = deviceActionFailure({ kind: 'despawn', status, message: 'needs consent' })
  assert.equal(d.ambiguous, false)
  assert.match(d.text, /NOT stopped/)
  assert.match(d.text, /still running/)
  assert.match(d.text, /still recording, if it was/)
}

// spawn and despawn are different sentences, and each world differs within a kind.
for (const kind of ['spawn', 'despawn']) {
  assert.notEqual(
    deviceActionFailure({ kind, status: 0, message: 'x' }).text,
    deviceActionFailure({ kind, status: 401, message: 'x' }).text,
  )
}
assert.notEqual(
  deviceActionFailure({ kind: 'spawn', status: 0, message: 'x' }).text,
  deviceActionFailure({ kind: 'despawn', status: 0, message: 'x' }).text,
)

// No "undefined", no empty parens, whatever is missing.
for (const kind of ['spawn', 'despawn']) {
  for (const f of [{ kind, status: 0 }, { kind, status: 401 }, { kind, status: 503, message: '  ' }]) {
    const v = deviceActionFailure(f)
    assert.doesNotMatch(v.text, /undefined|null/)
    assert.doesNotMatch(v.text, /\(\)/)
    assert.match(v.text, /no detail|x|boom|\S/)
  }
}

// An unmodelled 4xx is not assumed inert (one classifier, four screens).
assert.equal(deviceActionFailure({ kind: 'spawn', status: 418, message: 'x' }).ambiguous, true)

console.log('deviceOutcome: all assertions passed')
