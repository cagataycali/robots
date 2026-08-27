import assert from 'node:assert/strict'
import { openActionCopy } from '/tmp/recordAction.mjs'

const real = openActionCopy(false)
const mock = openActionCopy(true)
const unknown = openActionCopy(null)

// --- the label must name the ACTION, not a data structure ------------------
for (const a of [real, mock, unknown]) {
  assert.doesNotMatch(a.label, /^open session$/, 'still describing a data structure')
  assert.ok(a.hint.length > 40, 'the consequence must be stated, not hinted at')
  assert.ok(a.aria.length > a.label.length, 'the accessible name must carry the consequence too')
}

// --- the real thing says what happens to the ROOM --------------------------
assert.match(real.label, /arms/)
assert.match(real.hint, /despawned/)
assert.match(real.hint, /energised to hold position/)
assert.match(real.hint, /Nothing is written until you start an episode/)
assert.equal(real.cls, '')
assert.match(real.aria, /despawns both peers/)

// --- rehearsal admits itself ON the button ---------------------------------
assert.match(mock.label, /rehearsal/)
assert.equal(mock.cls, 'rehearsal')
assert.match(mock.hint, /no dataset is written/)
assert.match(mock.hint, /no arm is touched/)
assert.match(mock.aria, /nothing is written/)
// and it must NOT claim the real consequences, which would be a lie
assert.doesNotMatch(mock.hint, /despawned/)
assert.doesNotMatch(mock.label, /^open the arms/)

// --- unknown claims neither recorder, but still names the action -----------
assert.equal(unknown.label, real.label)
assert.doesNotMatch(unknown.label, /rehearsal/, 'guessed a rehearsal from silence')
assert.deepEqual(openActionCopy(undefined), unknown)

// only the literal true means rehearsal
for (const weird of ['true', 1, {}, 'mock']) {
  assert.doesNotMatch(openActionCopy(weird).label, /rehearsal/, `truthy ${JSON.stringify(weird)} claimed rehearsal`)
}
console.log('openActionCopy: all assertions passed')
