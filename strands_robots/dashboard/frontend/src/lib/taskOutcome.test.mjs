// Assertions for what a card may claim when a run/stop request to a real robot fails
// (lib/taskOutcome.ts). Run: npx esbuild src/lib/taskOutcome.ts --bundle --format=esm
// --outfile=/tmp/taskOutcome.mjs \ && node src/lib/taskOutcome.test.mjs
import assert from 'node:assert/strict'

const { runFailure, stopFailure } = await import('/tmp/taskOutcome.mjs')

// THE BUG: a lost answer on POST /api/robots/<peer>/task became "failed", and a
// hand reaches into a workspace where the arm may be under policy control.
{
  const v = runFailure({ status: 0, message: 'cannot reach robots.cagatay.my: Load failed' })
  assert.equal(v.ambiguous, true)
  assert.match(v.text, /MAY have started/)
  assert.match(v.text, /this arm can be moving/)
  assert.match(v.text, /Keep hands clear/)
  // Hand off to the observer that actually knows: the peer's own status.
  assert.match(v.text, /watch this card/)
  assert.match(v.text, /“running”/)
  // And name the cost of the reflex press.
  assert.match(v.text, /second task/)
  assert.equal(v.detail, 'no answer — delivery unknown')
}

// A 5xx means the handler ran: same ambiguity, and the detail keeps the code.
{
  const v = runFailure({ status: 502, message: 'bad gateway' })
  assert.equal(v.ambiguous, true)
  assert.equal(v.detail, 'HTTP 502')
  assert.match(v.text, /MAY have started/)
}

// Only a pre-handler refusal may say the arm was never told anything — and it
// MUST, or a refused run leaves the operator treating a parked arm as live.
for (const status of [400, 401, 403, 404, 422, 429]) {
  const v = runFailure({ status, message: 'needs consent' })
  assert.equal(v.ambiguous, false)
  assert.match(v.text, /nothing was sent to the arm/)
  assert.match(v.text, /NOT running/)
  assert.doesNotMatch(v.text, /MAY have started/)
}

{
  const v = stopFailure({ status: 0, message: 'Load failed' })
  assert.equal(v.ambiguous, true)
  assert.match(v.text, /stop may NOT have been delivered/)
  assert.match(v.text, /Assume the arm is still moving/)
  assert.match(v.text, /STOP ALL \(press \.\)/)
  assert.match(v.text, /power switch/)
}
{
  const v = stopFailure({ status: 403, message: 'not authenticated' })
  assert.equal(v.ambiguous, false)
  assert.match(v.text, /never reached the robot/)
  assert.match(v.text, /still doing whatever it was doing/)
  // Even a provably-undelivered stop must still route the operator to a brake.
  assert.match(v.text, /power switch/)
}

// run and stop are different sentences, and both worlds differ within each.
assert.notEqual(runFailure({ status: 0, message: 'x' }).text, stopFailure({ status: 0, message: 'x' }).text)
assert.notEqual(runFailure({ status: 0, message: 'x' }).text, runFailure({ status: 401, message: 'x' }).text)
assert.notEqual(stopFailure({ status: 0, message: 'x' }).text, stopFailure({ status: 401, message: 'x' }).text)

// Missing/blank detail never renders as "undefined" or an empty paren pair.
for (const f of [{ status: 0 }, { status: 401 }, { status: 500, message: '   ' }]) {
  for (const v of [runFailure(f), stopFailure(f)]) {
    assert.doesNotMatch(v.text, /undefined|null/)
    assert.doesNotMatch(v.text, /\(\)/)
  }
}

// An unmodelled 4xx is not assumed inert (shared classifier with estop/training).
assert.equal(runFailure({ status: 418, message: 'x' }).ambiguous, true)

console.log('taskOutcome: all assertions passed')
