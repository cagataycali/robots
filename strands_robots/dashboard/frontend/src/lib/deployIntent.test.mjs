// node assertions over the bundled module (esbuild first — see the sibling
// calibrateCommand.test.mjs for the same pattern).
//
// A DEPLOY INTENT is the note the Training tab leaves for a robot's run form: "this checkpoint,
// carried across, for a human to press Run on". Untested until now, and it guards a physical act —
// the thing it must never do is reappear later on a DIFFERENT robot's form.
import assert from 'node:assert/strict'

// A sessionStorage the module can use: it is read at call time, so defining it before the import is enough.
const store = new Map()
globalThis.sessionStorage = {
  getItem: k => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => { store.set(k, String(v)) },
  removeItem: k => { store.delete(k) },
}

const { setDeployIntent, peekDeployIntent, clearDeployIntent } = await import('/tmp/deployIntent.mjs')

const intent = { checkpoint: 'HashtagRobotics/smolvla-tic-tac-toe', policy_type: 'smolvla', source: 'job 41' }

// the round trip, and the banner's material
setDeployIntent(intent)
const got = peekDeployIntent()
assert.equal(got.checkpoint, intent.checkpoint)
assert.equal(got.source, 'job 41', 'the prefill must be able to explain where it came from')
assert.equal(typeof got.at, 'number')

// peek does NOT consume: the form reads it, then clears it deliberately after applying
assert.ok(peekDeployIntent(), 'a second peek still sees it')
clearDeployIntent()
assert.equal(peekDeployIntent(), null, 'consumed means gone')

// EXPIRY: 10 minutes. A forgotten click cannot ambush a form later.
setDeployIntent(intent)
const t = peekDeployIntent().at
assert.ok(peekDeployIntent(t + 9 * 60 * 1000), 'still valid at 9 minutes')
assert.equal(peekDeployIntent(t + 11 * 60 * 1000), null, 'expired at 11 minutes')
assert.equal(peekDeployIntent(), null, 'an expired intent is REMOVED, not just hidden')

// A CLOCK THAT JUMPED BACK (sleep/resume, NTP correction, VM snapshot) used to make an intent
// immortal: `now - at > TTL` is never true when the age is negative. It must read as untrustworthy.
store.set('strands.deployIntent', JSON.stringify({ ...intent, at: 5_000_000 }))
assert.equal(peekDeployIntent(1_000_000), null, 'a future stamp is expired, not fresh')
assert.equal(store.size, 0, 'and it is cleared, so it cannot come back when the clock catches up')

// small backwards drift is tolerated — a 30s skew is not a reason to lose a legitimate click
store.set('strands.deployIntent', JSON.stringify({ ...intent, at: 1_030_000 }))
assert.ok(peekDeployIntent(1_000_000), '30s of drift still applies')

// GARBAGE never becomes a prefill: half a note is not a note
for (const bad of ['not json', '5', 'null', JSON.stringify({ checkpoint: '' }), JSON.stringify({ policy_type: 'x', at: Date.now() })]) {
  store.set('strands.deployIntent', bad)
  assert.equal(peekDeployIntent(), null, `refused: ${bad}`)
}
// a note with no timestamp is refused too — an undated intent cannot be expired, so it must not be honoured
store.set('strands.deployIntent', JSON.stringify({ checkpoint: 'a/b', source: 's' }))
assert.equal(peekDeployIntent(), null, 'no `at` means no expiry, so no prefill')

// storage that refuses to write must not throw into the click handler
globalThis.sessionStorage = { getItem: () => { throw new Error('blocked') }, setItem: () => { throw new Error('blocked') }, removeItem: () => {} }
assert.doesNotThrow(() => setDeployIntent(intent), 'a blocked storage still leaves the button harmless')
assert.equal(peekDeployIntent(), null, 'and reads degrade to "no intent"')

console.log('deployIntent: all assertions passed')
