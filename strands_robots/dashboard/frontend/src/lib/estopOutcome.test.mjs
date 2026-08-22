// Assertions for what the sheet may claim when a STOP ALL request fails (lib/estopOutcome.ts).
// Run: npx esbuild src/lib/estopOutcome.ts --bundle --format=esm
// --outfile=/tmp/estopOutcome.mjs \ && node src/lib/estopOutcome.test.mjs
import assert from 'node:assert/strict'

const { estopFailureVerdict, resumeFailureVerdict, refusedBeforeActing } =
  await import('/tmp/estopOutcome.mjs')

// THE BUG: fetch rejects (HttpError status 0) both for a request that never left the machine
// and for one that ran and lost its answer.
{
  const v = estopFailureVerdict({ status: 0, message: 'cannot reach robots.cagatay.my: Load failed' })
  assert.equal(v.delivered, 'unknown')
  assert.match(v.headline, /MAY already have reached the fleet/)
  assert.match(v.headline, /cannot tell/)
  assert.doesNotMatch(v.headline + v.advice, /[Nn]othing was sent/)
  assert.match(v.advice, /Assume the robots are still moving/)
  assert.match(v.advice, /power switch/)
  // The follow-on symptom is pre-explained, so a working lockout is not misread
  // as a new fault five seconds later.
  assert.match(v.advice, /refuse commands until you resume/)
  assert.equal(v.retryRepeats, true)
}

// A 5xx is the same story from the server's side: the handler RAN.
{
  const v = estopFailureVerdict({ status: 500, message: 'zenoh publisher closed' })
  assert.equal(v.delivered, 'unknown')
  assert.match(v.headline, /failed mid-stop/)
  assert.match(v.headline, /lockout MAY already have been signalled/)
  assert.doesNotMatch(v.headline, /nothing/i)
}

// Only a pre-handler refusal is provably inert — and that one MUST say so,
// because "may have fired" would send the operator hunting a phantom lockout.
for (const status of [400, 401, 403, 404, 405, 422, 429]) {
  const v = estopFailureVerdict({ status, message: 'not authenticated' })
  assert.equal(v.delivered, 'no', `status ${status} is a refusal before any handler ran`)
  assert.match(v.headline, /nothing was sent, no robot was told to stop/)
  assert.equal(v.retryRepeats, false)
  // Even a refused stop leaves the arms moving: the hardware advice never drops.
  assert.match(v.advice, /power switch/)
}

// The classifier itself, at the edges.
assert.equal(refusedBeforeActing(401), true)
assert.equal(refusedBeforeActing(0), false, 'transport failure is unknowable, never "nothing ran"')
assert.equal(refusedBeforeActing(500), false)
assert.equal(refusedBeforeActing(503), false)
assert.equal(refusedBeforeActing(418), false, 'an unmodelled 4xx is not assumed inert')
assert.equal(refusedBeforeActing(null), false)
assert.equal(refusedBeforeActing(undefined), false)
assert.equal(refusedBeforeActing(NaN), false)

// A missing message never renders as "undefined".
for (const f of [{ status: 0 }, { status: 401 }, { status: 500, message: '  ' }]) {
  const v = estopFailureVerdict(f)
  assert.doesNotMatch(v.headline, /undefined|null/)
  assert.match(v.headline, /no detail|not authenticated|—/)
}

// The two worlds are two different sentences.
assert.notEqual(
  estopFailureVerdict({ status: 0, message: 'x' }).headline,
  estopFailureVerdict({ status: 401, message: 'x' }).headline,
)

// RESUME, same asymmetry: a lost answer may mean the lockout DID clear, so
// "still locked" is not sayable.
{
  const v = resumeFailureVerdict({ status: 0, message: 'Load failed' })
  assert.equal(v.delivered, 'unknown')
  assert.match(v.text, /may or may not have cleared/)
  assert.match(v.text, /Check whether a robot accepts a command/)
  assert.doesNotMatch(v.text, /still in place/)
}
{
  const v = resumeFailureVerdict({ status: 403, message: 'override rejected' })
  assert.equal(v.delivered, 'no')
  assert.match(v.text, /still in place/)
  assert.match(v.text, /cooldown/)
}

console.log('estopOutcome: all assertions passed')
