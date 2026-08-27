// Assertions for the dataset picker's hint (lib/datasetHint.ts).
// Run: npx esbuild src/lib/datasetHint.ts --bundle --format=esm --outfile=/tmp/datasetHint.mjs \
//        && node src/lib/datasetHint.test.mjs
import assert from 'node:assert/strict'

const { datasetHint, isCurrentResponse } = await import('/tmp/datasetHint.mjs')

// A settled search with rows: the list speaks for itself.
{
  const h = datasetHint({ query: 'so101', shownQuery: 'so101', count: 4 })
  assert.equal(h.text, null)
  assert.equal(h.auth, null)
}

// A settled search with nothing: "no match" is a real ANSWER and must not wear
// an outage's clothes.
{
  const h = datasetHint({ query: 'so101', shownQuery: 'so101', count: 0 })
  assert.equal(h.tone, 'info')
  assert.match(h.text, /nothing here or on the Hub matches “so101”/)
  assert.match(h.text, /the Hub answered/)
}

{
  const h = datasetHint({ query: 'so101', shownQuery: 'so101', count: 0, problem: 'hub search failed: 503' })
  assert.equal(h.tone, 'warn')
  assert.match(h.text, /hub search failed: 503/)
  assert.doesNotMatch(h.text, /no match/)
}

{
  const h = datasetHint({ query: 'so101', shownQuery: 'so', count: 0 })
  assert.equal(h.tone, 'pending')
  assert.match(h.text, /searching for “so101”/)
  assert.doesNotMatch(h.text, /no match/, 'nobody has asked the Hub about so101 yet')
}

// THE BUG (2): rows from the previous query are still on screen — say so, rather
// than letting them look like the answer to the question being typed.
{
  const h = datasetHint({ query: 'so101', shownQuery: 'pusht', count: 3 })
  assert.equal(h.tone, 'pending')
  assert.match(h.text, /searching for “so101”/)
  assert.match(h.text, /still the results for “pusht”/)
}

// A stale FAILURE is not reported as this query's failure either.
{
  const h = datasetHint({ query: 'so101', shownQuery: 'so', count: 0, problem: 'hub search failed' })
  assert.equal(h.tone, 'pending')
  assert.doesNotMatch(h.text, /failed/)
}

// Never searched yet, box empty: nothing to say (the panel's own empty state
// explains where datasets come from).
{
  const h = datasetHint({ query: '', shownQuery: null, count: 0 })
  assert.equal(h.text, null)
  assert.equal(h.tone, 'pending')
}
// Box emptied after a search: no claim about "" is worth making.
assert.equal(datasetHint({ query: '', shownQuery: 'so101', count: 2 }).text, null)

// Whitespace is not a different question ("so101" vs "so101 " must settle).
{
  const h = datasetHint({ query: 'so101 ', shownQuery: 'so101', count: 0 })
  assert.equal(h.tone, 'info')
  assert.match(h.text, /matches “so101”/)
}

// Anonymous Hub search changes what "no match" MEANS, so it rides along in every
// state where the user is asking the Hub something — including a pending one.
for (const shown of ['so101', 'so', null]) {
  const h = datasetHint({ query: 'so101', shownQuery: shown, count: 0, anonymous: true, authDetail: 'no HF_TOKEN' })
  assert.match(h.auth, /public only \(no HF_TOKEN\)/)
  assert.match(h.auth, /gated dataset will look like “no match”/)
}
// ...but not when nothing has been typed, and not when we ARE authenticated.
assert.equal(datasetHint({ query: '', shownQuery: null, count: 0, anonymous: true }).auth, null)
assert.equal(datasetHint({ query: 'so101', shownQuery: 'so101', count: 0, anonymous: false }).auth, null)
// Missing detail must not render an empty paren pair.
assert.doesNotMatch(datasetHint({ query: 'x', shownQuery: 'x', count: 0, anonymous: true }).auth, /\(\)/)

// The last REQUEST wins, not the last response: a slow answer for a query the
// user has moved past may not repopulate the list.
assert.equal(isCurrentResponse(7, 7), true)
assert.equal(isCurrentResponse(6, 7), false)
assert.equal(isCurrentResponse(8, 7), false)

console.log('datasetHint: all assertions passed')
