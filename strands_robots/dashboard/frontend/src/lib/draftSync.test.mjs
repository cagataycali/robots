// Q76: a value the operator has touched belongs to the operator. Run:
//   npx esbuild src/lib/draftSync.ts --bundle --format=esm --outfile=/tmp/draftSync.mjs && node src/lib/draftSync.test.mjs
import assert from 'node:assert/strict'
const { syncDrafts, dirtyFields, unsavedSummary } = await import('/tmp/draftSync.mjs')

// THE BUG: type a new prompt, then save the mesh tab -> reload -> the prompt used to revert.
{
  const server = { prompt: 'old prompt', meshPort: '7447' }
  const current = { prompt: 'MY LONG NEW PROMPT', meshPort: '9000' }
  const r = syncDrafts(current, server, { prompt: 'old prompt', meshPort: '9000' })
  assert.equal(r.next.prompt, 'MY LONG NEW PROMPT', 'the untouched-by-server prompt draft must survive')
  assert.deepEqual(r.kept, ['prompt'])
  assert.deepEqual(r.conflicts, [], 'the save landing on meshPort is not a conflict with itself')
  // The field that WAS saved now matches the server, so it is no longer dirty.
  assert.equal(r.next.meshPort, '9000')
  assert.deepEqual(dirtyFields(r.next, { prompt: 'old prompt', meshPort: '9000' }), ['prompt'])
}

// An untouched field still tracks the server: another operator's change must show up.
{
  const r = syncDrafts({ a: 'x' }, { a: 'x' }, { a: 'y' })
  assert.equal(r.next.a, 'y')
  assert.deepEqual(r.adopted, ['a'])
  assert.deepEqual(r.kept, [])
}

// Touched AND the server moved = a real conflict. The draft is STILL kept (never overwrite typing),
// but it is reported, because only the human can resolve it.
{
  const r = syncDrafts({ a: 'mine' }, { a: 'orig' }, { a: 'theirs' })
  assert.equal(r.next.a, 'mine')
  assert.deepEqual(r.conflicts, ['a'])
  assert.deepEqual(r.kept, [])
}

// Typing something and then typing it BACK is not dirty: the operator ended where the server is.
assert.deepEqual(dirtyFields({ a: 'x' }, { a: 'x' }), [])
// Empty string is a real value, distinct from absent (Q75's lesson applied to drafts).
assert.deepEqual(dirtyFields({ a: '' }, { a: 'x' }), ['a'])
assert.deepEqual(dirtyFields({}, { a: 'x' }), [], 'a field not yet rendered is not "dirty"')

// First load: nothing was seeded yet, so the server wins - except for a draft already typed
// (the drawer can be opened and typed in before the first config arrives).
{
  const fresh = syncDrafts({}, {}, { a: 'server' })
  assert.equal(fresh.next.a, 'server')
  assert.deepEqual(fresh.adopted, ['a'])
  const typedEarly = syncDrafts({ a: 'typed' }, {}, { a: 'server' })
  assert.equal(typedEarly.next.a, 'typed', 'typing before the first load must not be swallowed')
  assert.deepEqual(typedEarly.kept, ['a'])
}

// Keys the server no longer reports are left alone rather than deleted underneath the operator.
{
  const r = syncDrafts({ gone: 'draft' }, { gone: 'draft' }, {})
  assert.equal(r.next.gone, 'draft')
}

// The close guard's sentence: names the fields, in English, or says nothing at all.
assert.equal(unsavedSummary([]), '')
assert.equal(unsavedSummary(['prompt'], { prompt: 'the system prompt' }), 'Unsaved change to the system prompt')
assert.equal(
  unsavedSummary(['prompt', 'temperature'], { prompt: 'the system prompt', temperature: 'temperature' }),
  'Unsaved changes to the system prompt and temperature',
)
assert.equal(
  unsavedSummary(['a', 'b', 'c']),
  'Unsaved changes to a, b and c',
)
// An unlabelled key still produces a usable sentence rather than "undefined".
assert.match(unsavedSummary(['weird_key']), /weird_key/)

console.log('draftSync: all assertions passed')
