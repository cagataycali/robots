// Assertions for the joint strip's own age note (lib/jointFreshness.ts).
// Run: npx esbuild src/lib/jointFreshness.ts --bundle --format=esm --outfile=/tmp/jointFreshness.mjs \
//        && node src/lib/jointFreshness.test.mjs
import assert from 'node:assert/strict'

const { jointAgeNote, LAGGING_MS, FROZEN_MS } = await import('/tmp/jointFreshness.mjs')

// Fresh: a 10Hz stream with jitter must not decorate every card with a warning.
for (const ms of [0, 50, 333, 1999]) {
  const v = jointAgeNote(ms)
  assert.equal(v.level, 'live')
  assert.equal(v.text, null)
  assert.equal(v.dim, false)
}

// Lagging: say the age, do not yet disown the numbers.
{
  const v = jointAgeNote(4200)
  assert.equal(v.level, 'lagging')
  assert.match(v.text, /4\.2s old/)
  assert.match(v.text, /lagging/)
  assert.equal(v.dim, false, 'a 4s-old value is still roughly right; dimming it cries wolf')
}

// THE BUG: past the presence window this is a dead stream, and the strip used to
// keep presenting it as the arm's position — with the operator's hands on the arm.
{
  const v = jointAgeNote(23_000)
  assert.equal(v.level, 'frozen')
  assert.match(v.text, /frozen 23s ago/)
  // It must name the WRONG CONCLUSION, not just the fact.
  assert.match(v.text, /not where the arm is now/)
  assert.equal(v.dim, true)
}

// Never received a frame is not "stale": the empty state explains itself.
for (const nothing of [null, undefined, NaN, Infinity]) {
  const v = jointAgeNote(nothing)
  assert.equal(v.level, 'unknown')
  assert.equal(v.text, null)
  assert.equal(v.dim, false)
}

// Boundaries are exact, and the two thresholds are ordered.
assert.ok(LAGGING_MS < FROZEN_MS)
assert.equal(jointAgeNote(LAGGING_MS - 1).level, 'live')
assert.equal(jointAgeNote(LAGGING_MS).level, 'lagging')
assert.equal(jointAgeNote(FROZEN_MS - 1).level, 'lagging')
assert.equal(jointAgeNote(FROZEN_MS).level, 'frozen')

// A negative age (clock skew between a phone and the Mac) is not a lie about
// freshness in the dangerous direction: treat it as now, never as frozen.
assert.equal(jointAgeNote(-5000).level, 'live')

// Ages read as seconds a human can compare with a stopwatch, and never as
// "0.0s" once we have decided to complain about them.
assert.match(jointAgeNote(2_500).text, /2\.5s/)
assert.match(jointAgeNote(9_900).text, /9\.9s/)
assert.match(jointAgeNote(61_000).text, /61s/)
for (const ms of [2_000, 5_000, 60_000, 600_000]) {
  const t = jointAgeNote(ms).text
  assert.doesNotMatch(t, /undefined|NaN/)
  assert.doesNotMatch(t, /\b0\.0s\b/)
}

// Only the frozen level dims: dimming is a claim ("do not read this as now").
assert.deepEqual(
  [0, 3_000, 30_000].map(ms => jointAgeNote(ms).dim),
  [false, false, true],
)

console.log('jointFreshness: all assertions passed')
