import assert from 'node:assert/strict'
import { motionChip } from '/tmp/motionChip.mjs'

// --- THE DEFECT: a tri-state rendered through a ternary ---------------------- `moving ?
const noJoints = motionChip(null, { jointsSeen: false })
assert.equal(noJoints.tone, 'unknown')
assert.notEqual(noJoints.tone, 'still', 'unmeasurable must not borrow the measured-still styling')
assert.equal(noJoints.label, 'motion unknown')
assert.match(noJoints.title, /publishes no joint positions/)
assert.match(noJoints.title, /treat the arm as able to move/)
assert.match(noJoints.aria, /motion unknown/)

// The transient reason is the same tone but a different sentence: "measuring" is
// a state that ends by itself, "motion unknown" is one that does not.
const warming = motionChip(null, { jointsSeen: true })
assert.equal(warming.tone, 'unknown')
assert.equal(warming.label, 'measuring')
assert.match(warming.title, /not enough state samples yet/)
assert.notEqual(warming.label, noJoints.label, 'the two reasons must read differently')

// Unknown-yet (nothing heard at all) behaves like the transient case: absence of
// a fact is not a fact about the peer.
for (const js of [null, undefined]) {
  const u = motionChip(null, { jointsSeen: js })
  assert.equal(u.label, 'measuring', `jointsSeen=${js} must not accuse the peer`)
}
assert.equal(motionChip(undefined).label, 'measuring')
assert.equal(motionChip(null).tone, 'unknown')

// --- the two real measurements are unchanged --------------------------------
const still = motionChip(false, { jointsSeen: true })
assert.equal(still.tone, 'still')
assert.equal(still.label, 'still')
assert.match(still.aria, /measured still/)

const moving = motionChip(true, { jointsSeen: true })
assert.equal(moving.tone, 'moving')
assert.equal(moving.label, 'moving')
assert.match(moving.title, /keep hands clear/)
assert.match(moving.aria, /keep hands clear/)

assert.equal(motionChip(true, { jointsSeen: false }).tone, 'moving')

// Every state announces itself to a screen reader - a coloured dot is not an
// announcement - and no state is silent or duplicated.
const all = [noJoints, warming, still, moving]
for (const c of all) {
  assert.ok(c.aria.length > 8, `aria too thin: ${JSON.stringify(c)}`)
  assert.ok(c.title.length > 20)
  assert.ok(['still', 'moving', 'unknown'].includes(c.tone))
  assert.doesNotMatch(c.label, /undefined|null|NaN/)
}
assert.equal(new Set(all.map(c => c.label)).size, 4, 'four distinct words for four distinct states')
// Only a real measurement may use the words "still"/"not changing" at all.
for (const c of [noJoints, warming]) {
  assert.doesNotMatch(c.label, /^still$/)
  assert.doesNotMatch(c.title, /joints are not changing/)
}

console.log('motionChip: 30 assertions ok')
