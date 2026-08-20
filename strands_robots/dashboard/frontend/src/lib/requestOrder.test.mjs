// Assertions for response ordering (lib/requestOrder.ts) — and a pin on the
// damage a stale answer does once it reaches lossTrace.
// Run: npx esbuild src/lib/requestOrder.ts --bundle --format=esm --outfile=/tmp/requestOrder.mjs \
//        && npx esbuild src/lib/lossTrace.ts --bundle --format=esm --outfile=/tmp/lossTrace.mjs \
//        && node src/lib/requestOrder.test.mjs
import assert from 'node:assert/strict'

const { isLatestRequest, newerThanApplied } = await import('/tmp/requestOrder.mjs')
const { pushLoss } = await import('/tmp/lossTrace.mjs')

// The newest request owns the screen.
assert.equal(isLatestRequest(4, 4), true)
assert.equal(isLatestRequest(3, 4), false)
assert.equal(isLatestRequest(5, 4), false, 'a seq ahead of latest is impossible; refuse it too')

// Per-key state: newer only, and a repeat of the same request is NOT newer (it
// carries no new information and must not re-stamp the freshness clock).
assert.equal(newerThanApplied(1, undefined), true)
assert.equal(newerThanApplied(2, 1), true)
assert.equal(newerThanApplied(1, 2), false)
assert.equal(newerThanApplied(2, 2), false)

// WHY the ordering is enforced BEFORE interpretation: lossTrace reads a step
// lower than the last as a restart and drops the entire curve. So one late answer
// from a superseded poll erases a healthy run's history.
{
  let trace = []
  trace = pushLoss(trace, 1000, 2.0)
  trace = pushLoss(trace, 2000, 1.5)
  trace = pushLoss(trace, 3000, 1.2)
  assert.equal(trace.length, 3)
  const wiped = pushLoss(trace, 1500, 1.9)   // an older tick landing late
  assert.deepEqual(wiped, [{ step: 1500, loss: 1.9 }], 'the curve is GONE — this is the damage')
  // Guarded by ordering, the same late answer never gets there.
  const applied = 7
  assert.equal(newerThanApplied(5, applied), false)
  assert.deepEqual(trace.length, 3)
}

// Interleaving is realistic in both directions: fast-then-slow and slow-then-fast.
{
  const order = []
  for (const [seq, latest] of [[1, 3], [3, 3], [2, 3]]) {
    if (isLatestRequest(seq, latest)) order.push(seq)
  }
  assert.deepEqual(order, [3], 'only the newest request painted anything')
}

console.log('requestOrder: all assertions passed')
