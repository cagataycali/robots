import assert from 'node:assert/strict'

const { holdout } = await import('/tmp/holdout.mjs')

// Empty is a real answer: train on everything. It must NOT be reported as a problem, or the
// form would nag every operator who does not want a split.
{
  const h = holdout('')
  assert.equal(h.send, null)
  assert.equal(h.problem, null)
  assert.match(h.say, /every episode/)
  assert.match(h.say, /memoris/, 'the cost of no split is stated, since that is the default')
  assert.deepEqual(holdout('   '), h, 'whitespace is empty')
}

// A good value is sent and described.
{
  const h = holdout('2')
  assert.equal(h.send, 2)
  assert.equal(h.problem, null)
  assert.match(h.say, /last 2 episodes/)
  assert.match(h.say, /eval loss/)
}
assert.match(holdout('1').say, /last 1 episode\b/, 'singular reads as English')

// With the count known, the say does the arithmetic the operator would do in their head.
assert.match(holdout('2', 20).say, /18 to train on/)

// 0 and negatives produce NO split in the backend — accepted, they would show a holdout that
// does not exist.
for (const bad of ['0', '-3']) {
  const h = holdout(bad)
  assert.equal(h.send, null, `${bad} must not be sent`)
  assert.match(h.problem, /reserves no episodes/)
}

// Fractions are refused with the number they would ACTUALLY reserve (ceiling), not rounded quietly.
{
  const h = holdout('2.7')
  assert.equal(h.send, null)
  assert.match(h.problem, /reserve 3 episodes, not 2/)
}

// Junk.
assert.match(holdout('two').problem, /not a number/)

// The dataset's own count is the bound: nothing left to train on is refused here rather than
// after the submit round trip.
{
  const h = holdout('20', 20)
  assert.equal(h.send, null)
  assert.match(h.problem, /has 20 episodes/)
  assert.match(h.problem, /leaves 0 to train on/)
}
assert.equal(holdout('19', 20).send, 19, 'the last legal holdout is count - 1')

// Unknown count must not block a legitimate run: a Hub dataset with no local root has no
// episode count in the picker, and guessing a bound would refuse something valid.
for (const count of [undefined, null, 0, NaN]) {
  assert.equal(holdout('5', count).send, 5, `count ${count} must not veto a plain value`)
}

console.log('holdout: all assertions passed')
