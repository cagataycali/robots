// npx esbuild src/lib/peerName.ts --bundle --format=esm --outfile=/tmp/pn.mjs && node src/lib/peerName.test.mjs
import assert from 'node:assert/strict'
const { peerNameField, sanitizePeerName, PEER_NAME_RE } = await import('/tmp/pn.mjs')

// Empty is a legitimate choice, never a refusal — but it says what the generated name will be, so
// "so101-real-4718" stops being a surprise the operator meets on the card.
const blank = peerNameField('', { robotName: 'so101', mode: 'real' })
assert.equal(blank.problem, null)
assert.equal(blank.value, null)
assert.match(blank.note, /so101-real-<clock>/)
assert.match(blank.note, /cannot be renamed/)
assert.equal(peerNameField('   ').problem, null)

// A good name travels verbatim: this field must never rewrite what they typed.
assert.deepEqual(peerNameField('left-arm', { existing: ['right-arm'] }), {
  value: 'left-arm', problem: null, note: null, suggestion: null,
})
assert.equal(peerNameField('so101.wrist_2:a-B9').value, 'so101.wrist_2:a-B9')

// The charset refusal happens HERE, before a process exists, and names the consequence.
const star = peerNameField('left*')
assert.equal(star.value, null)
assert.match(star.problem, /only letters, digits and \. _ : -/)
assert.match(star.problem, /key space/)
const slash = peerNameField('arms/left')
assert.match(slash.problem, /key space/)
assert.equal(slash.suggestion, 'armsleft')
// A space is the likeliest thing a human types: offer the dashed form to tap.
assert.equal(peerNameField('left arm').suggestion, 'left-arm')
// A refusal with nothing salvageable offers nothing rather than an empty suggestion.
assert.equal(peerNameField('***').suggestion, null)

// Too long is reported with the real length, and the suggestion is the truncated legal form.
const long = peerNameField('a'.repeat(70))
assert.match(long.problem, /70 characters/)
assert.equal(long.suggestion.length, 64)

// A collision is the backend's 409, moved to before the button. The stake is named: the peer that
// is already running must not be shadowed.
const dup = peerNameField('left-arm', { existing: ['left-arm', 'left-arm-2'] })
assert.equal(dup.value, null)
assert.match(dup.problem, /already exists/)
// ...and the offered alternative skips the one that is also taken.
assert.equal(dup.suggestion, 'left-arm-3')

// No evidence is not evidence of a problem (the origin-badge / arm-role posture): with no listing,
// any legal name is accepted rather than guessed at.
assert.equal(peerNameField('left-arm', { existing: [] }).problem, null)
assert.equal(peerNameField('left-arm').problem, null)

// sanitize is pure and idempotent — pressing the suggestion twice cannot walk anywhere.
assert.equal(sanitizePeerName('  left  arm  '), 'left-arm')
assert.equal(sanitizePeerName('left-arm'), 'left-arm')
assert.equal(sanitizePeerName('--a--b--'), 'a-b')
assert.equal(sanitizePeerName(''), null)
assert.ok(PEER_NAME_RE.test('a'), 'the mirrored regex is exported for the sync test')

console.log('peerName: all assertions passed')
