import assert from 'node:assert/strict'
import { recordNavFlag } from '/tmp/rehearsalNav.mjs'

// --- SILENCE IS NOT EVIDENCE: an unprobed backend flags nothing -------------
for (const unknown of [null, undefined]) {
  const f = recordNavFlag(unknown)
  assert.equal(f.flagged, false)
  assert.equal(f.suffix, '')
  assert.equal(f.cls, '')
  assert.doesNotMatch(f.title, /rehearsal/i, 'guessed a rehearsal from silence')
  assert.doesNotMatch(f.aria, /rehearsal/i)
}

// a real backend is left completely alone
const real = recordNavFlag(false)
assert.deepEqual(real, recordNavFlag(null), 'a real backend must look like the default')
assert.equal(real.title, 'Record teleop episodes into a dataset')

// --- a known rehearsal is unmissable BEFORE the click ----------------------
const mock = recordNavFlag(true)
assert.equal(mock.flagged, true)
assert.match(mock.suffix, /rehearsal/)
assert.equal(mock.cls, 'rehearsal')
assert.match(mock.title, /nothing is written to disk/)
assert.match(mock.title, /no \/api\/record/)
// the warning must not be colour- or glyph-only: it is in the accessible name
assert.match(mock.aria, /rehearsal/)
assert.match(mock.aria, /nothing is written/)
// ...and it still says what the button DOES — a warning that replaces the label
// leaves the operator guessing what they lost
assert.match(mock.title, /Record teleop episodes/)

// the base description is caller-supplied and always survives
const custom = recordNavFlag(true, 'Collect a dataset')
assert.match(custom.title, /^Collect a dataset\. REHEARSAL/)
assert.equal(recordNavFlag(false, 'Collect a dataset').title, 'Collect a dataset')

// only the literal boolean true means rehearsal — no truthiness surprises from
// a probe that resolves to a string or an object
for (const weird of ['true', 1, {}, 'mock']) {
  assert.equal(recordNavFlag(weird).flagged, false, `truthy ${JSON.stringify(weird)} claimed rehearsal`)
}

console.log('recordNavFlag: all assertions passed')
