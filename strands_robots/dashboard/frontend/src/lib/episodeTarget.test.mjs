import assert from 'node:assert/strict'
import { episodeTarget, EPISODE_MAX } from '/tmp/episodeTarget.mjs'

// the bug: a typo silently became the default 20
const typo = episodeTarget('3o')
assert.equal(typo.problem, '“3o” is not a number')
assert.equal(typo.value, 0, 'a typo must not resolve to a usable count')

// the other two silent corrections
assert.match(episodeTarget('0').problem, /record nothing/)
assert.match(episodeTarget('-5').problem, /negative/)

// empty is a question, not an error shout
assert.match(episodeTarget('').problem, /how many/)
assert.match(episodeTarget('   ').problem, /how many/)

// the ordinary path stays ordinary
assert.deepEqual(episodeTarget('20'), { value: 20, problem: null, note: null })
assert.deepEqual(episodeTarget(' 7 '), { value: 7, problem: null, note: null })
assert.equal(episodeTarget('1').value, 1)

// a correction we DO make is admitted rather than hidden
const frac = episodeTarget('3.7')
assert.equal(frac.value, 3)
assert.equal(frac.problem, null)
assert.match(frac.note, /recording 3 episodes/)

// an absurd count is refused with the remedy, not clamped into a 6-hour session
assert.equal(episodeTarget(String(EPISODE_MAX)).problem, null)
assert.match(episodeTarget(String(EPISODE_MAX + 1)).problem, /record in batches/)

// nothing here throws on junk the DOM can produce
for (const junk of ['NaN', 'Infinity', '1e999', '--1', '1,5', undefined]) {
  const r = episodeTarget(junk)
  assert.ok(typeof r.value === 'number' && (r.problem === null ? r.value > 0 : r.value === 0), `junk ${junk}`)
}
// Infinity is not finite, so it is refused as "not a number" rather than as a too-large count —
// which is the honest wording for someone who typed a word, and 1e999 lands in the same place.
assert.match(episodeTarget('Infinity').problem, /is not a number/)
assert.match(episodeTarget('1e999').problem, /is not a number/)

console.log('episodeTarget: 18 assertions ok')
