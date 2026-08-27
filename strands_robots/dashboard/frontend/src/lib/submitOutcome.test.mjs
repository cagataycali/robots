// Assertions for what the training tab may claim when a start-something request fails
// (lib/submitOutcome.ts). Run: npx esbuild src/lib/submitOutcome.ts --bundle --format=esm
// --outfile=/tmp/submitOutcome.mjs \ && node src/lib/submitOutcome.test.mjs
import assert from 'node:assert/strict'

const { sideEffectVerdict } = await import('/tmp/submitOutcome.mjs')

// THE BUG: `⚠ <message>` for a lost answer reads as "it did not happen", and the
// operator presses train again — two multi-hour runs into one output_dir.
{
  const v = sideEffectVerdict({ kind: 'training', status: 0, message: 'cannot reach robots.cagatay.my: Load failed' })
  assert.equal(v.delivered, 'unknown')
  assert.match(v.text, /the training job MAY have started/)
  assert.match(v.text, /cannot tell/)
  assert.match(v.text, /SECOND run on the same GPU/)
  assert.match(v.text, /same output_dir/)
  assert.match(v.text, /job list below/)
  assert.equal(v.doubleRunRisk, true)
}

// A replay spawns a peer that DRIVES AN ARM: the duplicate is physical.
{
  const v = sideEffectVerdict({ kind: 'replay', status: 0, message: 'timeout' })
  assert.match(v.text, /the replay MAY have started/)
  assert.match(v.text, /SECOND peer driving the same arm/)
  assert.match(v.text, /fleet grid/)
  assert.equal(v.doubleRunRisk, true)
}

// Collect: a second recorder appending to the same dataset is silent corruption.
{
  const v = sideEffectVerdict({ kind: 'collect', status: 502, message: 'bad gateway' })
  assert.equal(v.delivered, 'unknown')
  assert.match(v.text, /failed mid-request \(502/)
  assert.match(v.text, /SECOND recorder appending to the same dataset/)
}

// Export is the one action where retrying is cheap — say that instead of a
// duplicate warning that does not apply.
{
  const v = sideEffectVerdict({ kind: 'export', status: 0, message: 'x' })
  assert.equal(v.doubleRunRisk, false)
  assert.match(v.text, /safe to retry an export/)
  assert.doesNotMatch(v.text, /SECOND/)
}

// Only a pre-handler refusal may say nothing started (shared classifier with the
// estop path, so the two screens cannot disagree about what 401 means).
for (const status of [400, 401, 403, 404, 422, 429]) {
  const v = sideEffectVerdict({ kind: 'training', status, message: 'not authenticated' })
  assert.equal(v.delivered, 'no')
  assert.match(v.text, /NOT started, nothing is running/)
  assert.equal(v.doubleRunRisk, false)
}
// ...and a 5xx is never that.
assert.equal(sideEffectVerdict({ kind: 'training', status: 500, message: 'x' }).delivered, 'unknown')
assert.equal(sideEffectVerdict({ kind: 'training', status: 418, message: 'x' }).delivered, 'unknown')

// The two worlds are different sentences for every kind.
for (const kind of ['training', 'collect', 'replay', 'export']) {
  const a = sideEffectVerdict({ kind, status: 0, message: 'x' }).text
  const b = sideEffectVerdict({ kind, status: 401, message: 'x' }).text
  assert.notEqual(a, b)
  assert.doesNotMatch(a, /undefined|null/)
  assert.doesNotMatch(b, /undefined|null/)
}

// Missing detail never renders as "undefined", and never as an empty paren pair.
{
  const v = sideEffectVerdict({ kind: 'training', status: 0 })
  assert.match(v.text, /no detail/)
  assert.doesNotMatch(v.text, /\(\)/)
}

console.log('submitOutcome: all assertions passed')
