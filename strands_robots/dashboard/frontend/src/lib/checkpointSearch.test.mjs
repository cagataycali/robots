// Assertions for the checkpoint type-ahead's honest empty state + stale-response
// guard (lib/checkpointSearch.ts).
// Run: npx esbuild src/lib/checkpointSearch.ts --bundle --format=esm --outfile=/tmp/checkpointSearch.mjs \
//        && node src/lib/checkpointSearch.test.mjs
import assert from 'node:assert/strict'

const { emptyNote, isCurrent } = await import('/tmp/checkpointSearch.mjs')

const HUB_DOWN = 'Hub search unavailable (timeout) - showing local cache only'

// Everything answered and nothing matched: the claim may cover the catalogue,
// and must SAY that both halves were searched (otherwise a user cannot tell
// this apart from the local-only case).
{
  const t = emptyNote({ query: 'smolvla-xyz' })
  assert.match(t, /no checkpoints match “smolvla-xyz”/)
  assert.match(t, /local cache \+ Hub/)
}

// THE FIX: with the Hub down, "no checkpoints match" is a claim nothing measured.
{
  const t = emptyNote({ query: 'smolvla-xyz', hubProblem: HUB_DOWN })
  assert.match(t, /already on this machine/, 'the claim must be scoped to what was consulted')
  assert.match(t, /Hub was not searched/)
  assert.match(t, /timeout/, 'the reason is repeated where it is being read')
  assert.match(t, /not "it does not exist"/, 'the wrong conclusion is named and refused')
  assert.doesNotMatch(t, /^no checkpoints match/)
}

// The two worlds are different sentences (the whole point).
assert.notEqual(emptyNote({ query: 'act' }), emptyNote({ query: 'act', hubProblem: HUB_DOWN }))

// An empty query (focus opens the menu before typing) invites input instead of
// reporting a nonexistent failure.
{
  const t = emptyNote({ query: '' })
  assert.match(t, /type part of a checkpoint name/)
  assert.doesNotMatch(t, /no checkpoints match/)
}
{
  const t = emptyNote({ query: '   ', hubProblem: HUB_DOWN })
  assert.match(t, /local cache/)
  assert.match(t, /catalogue is unknown from here/)
}

// Whitespace-only hubProblem is not a problem, and cannot invent a caveat.
assert.doesNotMatch(emptyNote({ query: 'act', hubProblem: '   ' }), /not searched/)
assert.doesNotMatch(emptyNote({ query: 'act', hubProblem: null }), /not searched/)

// Junk inputs cannot throw or produce a claim.
for (const bad of [undefined, null, 0, {}]) {
  assert.equal(typeof emptyNote({ query: bad, hubProblem: bad }), 'string')
}

// Stale-response guard: only the newest request may paint.
assert.equal(isCurrent(7, 7), true)
assert.equal(isCurrent(6, 7), false, 'a slow older search must not overwrite a newer one')
assert.equal(isCurrent(8, 7), false, 'a seq from the future is not current either')
assert.equal(isCurrent(NaN, NaN), false, 'unreadable sequence never paints')

console.log('checkpointSearch: all assertions passed')
