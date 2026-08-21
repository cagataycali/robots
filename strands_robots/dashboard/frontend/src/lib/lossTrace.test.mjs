// Run: npx esbuild src/lib/lossTrace.ts --bundle --format=esm --outfile=/tmp/lossTrace.mjs && node src/lib/lossTrace.test.mjs
//
// The loss curve is BUILT BY THE DASHBOARD from polled snapshots — there is no history API, each poll
// contributes at most one point. Every rule here exists because a plausible-looking curve made of
// witnessed points can lie in a specific way, and none of them were pinned by a test.
import assert from 'node:assert/strict'
import { pushLoss, lossPath, lossBand, fmtStep } from '/tmp/lossTrace.mjs'

// ── accumulation ──
let t = pushLoss([], 100, 2.5)
assert.deepEqual(t, [{ step: 100, loss: 2.5 }])
t = pushLoss(t, 200, 2.1)
assert.equal(t.length, 2)

// polls outpace log lines: the same step twice is ONE point, not a flat segment
t = pushLoss(t, 200, 2.1)
assert.equal(t.length, 2, 'an identical repeat changes nothing')
t = pushLoss(t, 200, 2.0)
assert.equal(t.length, 2, 'a fresher loss for the same step REPLACES it')
assert.equal(t[1].loss, 2.0)

// a lower step means the job restarted: the old curve belongs to a run that no longer exists
t = pushLoss(t, 10, 4.0)
assert.deepEqual(t, [{ step: 10, loss: 4.0 }], 'a restart resets rather than drawing a sawtooth')

// junk never enters the trace (a NaN loss in the log is diagnostic, not plottable)
const before = pushLoss([], 5, 1.0)
for (const [s, l] of [[NaN, 1], [5, NaN], [Infinity, 1], [5, Infinity], ['6', 1], [6, '1'], [null, null], [undefined, 2]]) {
  assert.equal(pushLoss(before, s, l), before, `dropped: step=${String(s)} loss=${String(l)}`)
}
// …and the identity is preserved, so React does not re-render on a rejected poll
assert.equal(pushLoss(before, NaN, NaN), before)
// a loss of 0, and a negative loss, are REAL values
assert.equal(pushLoss(before, 6, 0).length, 2)
assert.equal(pushLoss(before, 6, -0.4)[1].loss, -0.4)

// ── the cap thins the OLDER half and never loses the newest ──
let big = []
for (let i = 1; i <= 300; i++) big = pushLoss(big, i * 10, 1 / i, 240)
assert.ok(big.length <= 240, `stayed under the cap: ${big.length}`)
assert.equal(big[big.length - 1].step, 3000, 'the latest reading survives every thinning')
// steps stay strictly increasing, or lossPath would draw backwards
for (let i = 1; i < big.length; i++) assert.ok(big[i].step > big[i - 1].step, 'monotonic after decimation')
// early shape is kept, not truncated: the first point is still the first reading
assert.equal(big[0].step, 10)

// ── THE BAND: a flat run must not be magnified into a mountain ──
// A run stuck at 2.5000 ± 0.0004 used to fill the full canvas height, which reads as progress.
const stuck = [{ step: 1, loss: 2.5001 }, { step: 2, loss: 2.4999 }, { step: 3, loss: 2.5000 }]
const band = lossBand(stuck)
assert.equal(band.flat, true, 'noise below 2% of magnitude is flat, and says so')
assert.ok(band.hi - band.lo >= 2.5 * 0.02, 'the band is padded to a floor the noise cannot fill')
const flatPath = lossPath(stuck, 100, 34, 2)
const ys = flatPath.map(([, y]) => y)
assert.ok(Math.max(...ys) - Math.min(...ys) < 1, `drawn flat, not magnified: spread ${Math.max(...ys) - Math.min(...ys)}`)
assert.ok(Math.min(...ys) > 34 * 0.25 && Math.max(...ys) < 34 * 0.75,
  'a flat curve runs through the MIDDLE — pinned to the bottom edge reads as "converged to its best"')

// real variation is untouched: the band is exactly min..max and fills the height
const real = [{ step: 1, loss: 2.5 }, { step: 2, loss: 1.4 }, { step: 3, loss: 0.8 }]
assert.deepEqual(lossBand(real), { lo: 0.8, hi: 2.5, flat: false })
const path = lossPath(real, 100, 34, 2)
assert.equal(path.length, 3)
assert.equal(path[0][0], 2, 'first point at the left pad')
assert.equal(path[2][0], 98, 'last point at the right pad')
// LOW LOSS SITS LOW — the orientation every practitioner expects
assert.equal(path[0][1], 2, 'the worst loss is at the top')
assert.equal(path[2][1], 32, 'the best loss is at the bottom')
// x uses the STEP value, so decimated (unevenly spaced) points stay honest
const uneven = lossPath([{ step: 0, loss: 1 }, { step: 90, loss: 0.5 }, { step: 100, loss: 0.4 }], 102, 10, 1)
assert.equal(uneven[1][0], 1 + 0.9 * 100, 'x is proportional to step, not to index')

// all-zero losses have no magnitude to take a percentage of — an absolute floor, still centred
const zeros = lossBand([{ step: 1, loss: 0 }, { step: 2, loss: 0 }])
assert.equal(zeros.flat, true)
assert.ok(zeros.lo < 0 && zeros.hi > 0, 'centred on zero rather than collapsing')

// nothing drawable, no throw
assert.deepEqual(lossPath([], 100, 34), [])
assert.deepEqual(lossPath([{ step: 1, loss: 1 }], 100, 34), [], 'one point is not a curve')
assert.deepEqual(lossPath(real, 0, 34), [], 'an unmeasured canvas draws nothing')
assert.deepEqual(lossPath(real, 100, 0), [])
// every coordinate is finite: a NaN reaching a canvas path silently draws nothing at all
for (const [x, y] of lossPath(real, 100, 34)) assert.ok(Number.isFinite(x) && Number.isFinite(y))

// ── step labels match the trainer's own log formatting ──
assert.equal(fmtStep(999), '999')
assert.equal(fmtStep(1000), '1.0k')
assert.equal(fmtStep(12_345), '12.3k')
assert.equal(fmtStep(1_000_000), '1.0M')
assert.equal(fmtStep(2_500_000), '2.5M')
assert.equal(fmtStep(0), '0')
assert.equal(fmtStep(NaN), '?', 'a missing step never renders as a number')
assert.equal(fmtStep(Infinity), '?')

console.log('lossTrace: all assertions passed')
