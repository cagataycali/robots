// Dataset name derived from the task sentence (lib/datasetNameSuggest.ts).
// Run: npx esbuild src/lib/datasetNameSuggest.ts --bundle --format=esm --outfile=/tmp/datasetNameSuggest.mjs && node src/lib/datasetNameSuggest.test.mjs
import assert from 'node:assert/strict'

const { suggestDatasetName } = await import('/tmp/datasetNameSuggest.mjs')

// a task sentence becomes a recognisable kebab name, capped at 4 words
assert.equal(
  suggestDatasetName('pick up the red cube and place it in the bin', '', []),
  'pick-up-the-red',
)
// punctuation and case are flattened, not preserved
assert.equal(suggestDatasetName('Stack Cubes!', '', []), 'stack-cubes')
// THE LAW: any operator text in the field suppresses the offer
assert.equal(suggestDatasetName('stack cubes', 'my-name', []), null)
assert.equal(suggestDatasetName('stack cubes', '  x', []), null)
// a local collision offers the next free -N variant, not the taken name
assert.equal(
  suggestDatasetName('stack cubes', '', [{ repo_id: 'stack-cubes', local: true }]),
  'stack-cubes-2',
)
// a HUB dataset of the same name is not a local collision (matches nameVerdict's law)
assert.equal(
  suggestDatasetName('stack cubes', '', [{ repo_id: 'stack-cubes', local: false }]),
  'stack-cubes',
)
// nothing to derive from
assert.equal(suggestDatasetName('', '', []), null)
assert.equal(suggestDatasetName('!!!', '', []), null)
// null known list = no collision knowledge, plain slug
assert.equal(suggestDatasetName('wave hello', '', null), 'wave-hello')

console.log('datasetNameSuggest: all assertions passed')
