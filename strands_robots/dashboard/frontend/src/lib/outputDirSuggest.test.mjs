// Output-dir suggestion from the picked dataset (lib/outputDirSuggest.ts).
// Run: npx esbuild src/lib/outputDirSuggest.ts --bundle --format=esm --outfile=/tmp/outputDirSuggest.mjs && node src/lib/outputDirSuggest.test.mjs
import assert from 'node:assert/strict'

const { suggestOutputDir } = await import('/tmp/outputDirSuggest.mjs')

// a local dataset suggests from its folder name
assert.equal(
  suggestOutputDir({ dataset_root: '/data/lerobot/cagatay/pick-cube' }, ''),
  '/tmp/train_pick-cube',
)
// a Hub dataset suggests from the repo name, not the owner
assert.equal(
  suggestOutputDir({ dataset_repo_id: 'lerobot/pusht' }, ''),
  '/tmp/train_pusht',
)
// trailing slashes don't produce an empty name
assert.equal(
  suggestOutputDir({ dataset_root: '/data/demos/' }, ''),
  '/tmp/train_demos',
)
// unsafe characters become underscores, edges trimmed
assert.equal(
  suggestOutputDir({ dataset_repo_id: 'org/my set (v2)' }, ''),
  '/tmp/train_my_set_v2',
)
// THE LAW: the operator's own text is never fought — any non-blank value suppresses the offer
assert.equal(suggestOutputDir({ dataset_repo_id: 'lerobot/pusht' }, '/my/dir'), null)
assert.equal(suggestOutputDir({ dataset_repo_id: 'lerobot/pusht' }, '  x'), null)
// nothing picked, nothing offered
assert.equal(suggestOutputDir({}, ''), null)
assert.equal(suggestOutputDir({ dataset_root: '///' }, ''), null)

console.log('outputDirSuggest: all assertions passed')
