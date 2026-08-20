import assert from 'node:assert/strict'
import { numField } from '/tmp/numField.mjs'

const STEPS = { what: 'steps', min: 1, max: 2_000_000, remedy: 'submit a shorter run' }
const EPISODES = { what: 'episodes', min: 1, max: 500, remedy: 'collect in batches' }
const SECONDS = { what: 'seconds per episode', min: 1, max: 600 }

// THE DEFECT: `Number(raw) || 10000` treats a cleared field as consent to a 10k-step run…
assert.match(numField('', STEPS).problem, /how many steps/)
assert.equal(numField('', STEPS).value, 0)
// …and lets a minus sign through untouched, because `||` only catches 0 and NaN.
assert.match(numField('-100', STEPS).problem, /cannot be negative/)
assert.match(numField('-3', EPISODES).problem, /cannot be negative/)
assert.equal(numField('-3', EPISODES).value, 0)
assert.match(numField('0', EPISODES).problem, /below the minimum of 1/)

// the ordinary paths stay silent
assert.deepEqual(numField('10000', STEPS), { value: 10000, problem: null, note: null })
assert.deepEqual(numField(' 5 ', EPISODES), { value: 5, problem: null, note: null })
assert.deepEqual(numField('10', SECONDS), { value: 10, problem: null, note: null })

// a correction we make is stated
const frac = numField('4.8', EPISODES)
assert.equal(frac.value, 4)
assert.match(frac.note, /using 4 episodes/)
// …and a fraction below the floor is refused by the minimum, not floored to zero
assert.match(numField('0.4', EPISODES).problem, /below the minimum of 1/)
// (the floored-below-min guard only bites on a fractional minimum, which no caller has yet)
assert.match(numField('0.7', { what: 'hz', min: 0.5, max: 10 }).problem, /cannot be less than 0.5/)

// the ceiling names the remedy instead of clamping into a week-long job
assert.equal(numField('2000000', STEPS).problem, null)
assert.match(numField('2000001', STEPS).problem, /submit a shorter run/)
assert.match(numField('501', EPISODES).problem, /collect in batches/)
assert.match(numField('900', SECONDS).problem, /max 600 seconds per episode/)

// junk the DOM can produce never throws and never yields a usable number
for (const junk of ['abc', 'Infinity', '1e999', '--1', '1,5', undefined, null]) {
  const r = numField(junk, EPISODES)
  assert.ok(r.problem !== null && r.value === 0, `junk ${junk}`)
}
console.log('numField: 22 assertions ok')
