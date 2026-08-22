import assert from 'node:assert/strict'
import { validationScope, changedKeys } from '/tmp/validationScope.mjs'

const V = { provider: 'lerobot_local', config: { pretrained_name_or_path: 'HashtagRobotics/smolvla-a', device: 'mps' } }

// --- THE DEFECT ------------------------------------------------------------- validate()
// vouches for the config in the form; the verdict was cleared on ONE event only (changing the
// provider).
const swapped = validationScope(V, { provider: 'lerobot_local', config: { pretrained_name_or_path: 'someone/typo-model', device: 'mps' } })
assert.equal(swapped.applies, false)
assert.deepEqual(swapped.changed, ['pretrained_name_or_path'])
// It names WHAT moved: "something changed" would send them hunting.
assert.match(swapped.note, /before pretrained_name_or_path changed/)
assert.match(swapped.note, /does not describe the form as it is now/)
assert.match(swapped.note, /Validate again/)

// Unchanged input: the verdict still stands and adds no noise.
const same = validationScope(V, { provider: 'lerobot_local', config: { device: 'mps', pretrained_name_or_path: 'HashtagRobotics/smolvla-a' } })
assert.equal(same.applies, true, 'key ORDER is not a change')
assert.deepEqual(same.changed, [])
assert.equal(same.note, '')

// Provider changes are listed first, because they invalidate everything else.
const prov = validationScope(V, { provider: 'lerobot_async', config: { ...V.config, device: 'cuda' } })
assert.equal(prov.applies, false)
assert.equal(prov.changed[0], 'provider')
assert.ok(prov.changed.includes('device'))

// Added and removed keys both count.
assert.deepEqual(validationScope(V, { provider: V.provider, config: { ...V.config, extra: 1 } }).changed, ['extra'])
assert.deepEqual(validationScope(V, { provider: V.provider, config: { pretrained_name_or_path: V.config.pretrained_name_or_path } }).changed, ['device'])

// An absent key and an empty one are the same input (the backend skips blanks),
// so clearing an already-empty box must not invalidate a verdict.
assert.equal(validationScope({ provider: 'p', config: { a: 1 } }, { provider: 'p', config: { a: 1, b: '' } }).applies, true)
assert.equal(validationScope({ provider: 'p', config: { a: 1, b: null } }, { provider: 'p', config: { a: 1 } }).applies, true)
// ...but a real value appearing in that box is a change.
assert.equal(validationScope({ provider: 'p', config: { a: 1, b: '' } }, { provider: 'p', config: { a: 1, b: 'x' } }).applies, false)

// Nested config compares by VALUE, not by reference or key order.
const nested = { provider: 'p', config: { opts: { fps: 30, size: [640, 480] } } }
assert.equal(validationScope(nested, { provider: 'p', config: { opts: { size: [640, 480], fps: 30 } } }).applies, true)
assert.equal(validationScope(nested, { provider: 'p', config: { opts: { fps: 15, size: [640, 480] } } }).applies, false)

// "1" from a text input and 1 from a number input are the same wire value.
assert.equal(validationScope({ provider: 'p', config: { n: 1 } }, { provider: 'p', config: { n: '1' } }).applies, true)

// No verdict recorded yet -> nothing to invalidate (the banner is whatever the
// server just said).
assert.equal(validationScope(null, { provider: 'p', config: {} }).applies, true)
assert.equal(validationScope(undefined, { provider: 'p', config: { a: 1 } }).note, '')

// Long change lists stay readable instead of dumping a form.
const many = validationScope({ provider: 'p', config: { a: 1, b: 2, c: 3, d: 4, e: 5 } }, { provider: 'p', config: { a: 9, b: 9, c: 9, d: 9, e: 9 } })
assert.equal(many.changed.length, 5)
assert.match(many.note, /\+2 more/)

// Malformed configs cannot crash the banner (this renders next to ▶ Run).
for (const bad of [null, undefined]) {
  assert.equal(validationScope({ provider: 'p', config: bad }, { provider: 'p', config: bad }).applies, true)
}
assert.deepEqual(changedKeys({}, {}), [])
assert.deepEqual(changedKeys({ z: 1, a: 2 }, { z: 2, a: 3 }), ['a', 'z'], 'sorted for a stable sentence')

console.log('validationScope: 30 assertions ok')
