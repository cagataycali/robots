// npx esbuild src/lib/recordFps.ts --bundle --format=esm --outfile=/tmp/rf.mjs && node src/lib/recordFps.test.mjs
import assert from 'node:assert/strict'
const { fpsField, fpsSuggestion, DEFAULT_FPS } = await import('/tmp/rf.mjs')

// Empty = the documented backend default, no complaint.
assert.deepEqual(fpsField(''), { value: DEFAULT_FPS, problem: null, note: null })
assert.deepEqual(fpsField('  '), { value: DEFAULT_FPS, problem: null, note: null })

// A typo is reported, never silently turned into 30 (episodeTarget's lesson).
assert.match(fpsField('3o').problem, /not a number/)
assert.match(fpsField('0').problem, /between 1 and 60/)
assert.match(fpsField('120').problem, /between 1 and 60/)

// A correction we make must be admitted.
const r = fpsField('29.6')
assert.equal(r.value, 30); assert.equal(r.problem, null); assert.match(r.note, /whole number/)
assert.deepEqual(fpsField('4'), { value: 4, problem: null, note: null })

// The suggestion turns the warning into an action.
const s = fpsSuggestion({ declared_fps: 30, measured_fps: 4.2, ratio: 7.1, slower: true, detail: 'x' })
assert.equal(s.fps, '4')
assert.match(s.label, /use 4 fps next session/)
assert.match(s.why, /cannot be re-declared/, 'must not imply this session gets fixed')

// Nothing to suggest: no notice, an unusable measurement, or agreement.
assert.equal(fpsSuggestion(null), null)
assert.equal(fpsSuggestion({ declared_fps: 30, measured_fps: 0.4, ratio: 75, slower: true, detail: '' }), null)
assert.equal(fpsSuggestion({ declared_fps: 30, measured_fps: 29.7, ratio: 1.01, slower: true, detail: '' }), null)
assert.equal(fpsSuggestion({ declared_fps: 30, measured_fps: NaN, ratio: 1, slower: true, detail: '' }), null)

// Faster-than-declared gets its own sentence rather than the slower one.
const f = fpsSuggestion({ declared_fps: 5, measured_fps: 12, ratio: 2.4, slower: false, detail: '' })
assert.match(f.why, /faster than declared/)

console.log('recordFps: ok — 15 assertions')
