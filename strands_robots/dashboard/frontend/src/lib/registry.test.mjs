// Run: npx esbuild src/lib/registry.ts --bundle --format=esm --outfile=/tmp/registry.mjs && node src/lib/registry.test.mjs
//
// This normaliser exists because rendering a registry entry straight into an <option> IS React error
// #31 ("objects are not valid as a React child"), which throws during render and takes the whole
// dashboard down — not just the picker. So it must survive every shape /api/robots/registry has ever
// answered, and the id it produces must be a real spawn target.
import assert from 'node:assert/strict'
import { normalizeRegistry } from '/tmp/registry.mjs'

// ── the canonical shape (registry/robots.py list_robots) ──
const real = normalizeRegistry([
  { name: 'so101', description: 'SO-101 arm', category: 'arm', joints: 6, has_sim: true, has_real: true },
  { name: 'g1', description: '', category: 'humanoid', joints: 43, has_sim: true, has_real: false },
  { name: 'mystery', description: '', category: '', joints: null, has_sim: false, has_real: false },
])
assert.deepEqual(real.map(r => r.name), ['so101', 'g1', 'mystery'], 'the id is the spawn target')
assert.equal(real[0].label, 'so101 — arm, 6 joints')
assert.equal(real[1].label, 'g1 — humanoid, 43 joints, sim only')
assert.equal(real[2].label, 'mystery', 'nothing to add, so no dangling em dash')
// joints is `info.get("joints")` server-side and CAN be null — it must not become "null joints"
for (const r of real) assert.doesNotMatch(r.label, /null|undefined|NaN/)

// ── bare names, the older shape ──
assert.deepEqual(normalizeRegistry(['so101', 'g1']), [
  { name: 'so101', label: 'so101' }, { name: 'g1', label: 'g1' },
])
assert.deepEqual(normalizeRegistry([' so101 ']), [{ name: 'so101', label: 'so101' }], 'trimmed')
assert.deepEqual(normalizeRegistry(['', '   ']), [], 'a blank name is not a robot')

// ── THE MAP SHAPE: the KEY is the id ──
// A value that is a description used to BECOME the id, so the picker looked right and the spawn asked
// for a robot that does not exist.
assert.deepEqual(normalizeRegistry({ so101: 'SO-101 6-DOF arm' }), [
  { name: 'so101', label: 'so101 — SO-101 6-DOF arm' },
])
assert.deepEqual(normalizeRegistry({ so101: {} }), [{ name: 'so101', label: 'so101' }])
assert.deepEqual(normalizeRegistry({ so101: null }), [{ name: 'so101', label: 'so101' }],
  'a definition that failed to load still leaves a spawnable id')
// an inner name that DISAGREES with the key is a display name, never the spawn target
const disagree = normalizeRegistry({ so101: { name: 'SO-101 (Feetech)', category: 'arm', joints: 6 } })
assert.equal(disagree[0].name, 'so101')
assert.match(disagree[0].label, /SO-101 \(Feetech\)/, 'but it is still shown')
assert.match(disagree[0].label, /6 joints/)
// the map's own definitions normally carry the same name — no doubling then
assert.equal(normalizeRegistry({ so101: { name: 'so101', category: 'arm' } })[0].label, 'so101 — arm')

// ── one name, one option (a duplicate React key lets one row render as another) ──
assert.deepEqual(
  normalizeRegistry([{ name: 'so101', category: 'arm' }, 'so101', { name: 'so101' }]).map(r => r.label),
  ['so101 — arm'], 'first wins, and it is the more specific one')

// ── junk can never throw, and never reaches the DOM as an object ──
for (const junk of [null, undefined, 42, 'so101', true, NaN, () => {}]) {
  assert.deepEqual(normalizeRegistry(junk), [], `${String(junk)} yields nothing, not a crash`)
}
assert.deepEqual(normalizeRegistry([]), [])
assert.deepEqual(normalizeRegistry({}), [])
// a mixed, half-broken list keeps every usable row instead of failing whole
assert.deepEqual(
  normalizeRegistry(['so101', null, 42, { name: '   ' }, { name: 'g1' }]).map(r => r.name),
  ['so101', 'g1'])
// every field the UI puts in the DOM is a string
for (const r of normalizeRegistry([{ name: 'so101', category: 'arm', joints: 6 }])) {
  assert.equal(typeof r.name, 'string')
  assert.equal(typeof r.label, 'string')
}

console.log('registry: all assertions passed')
