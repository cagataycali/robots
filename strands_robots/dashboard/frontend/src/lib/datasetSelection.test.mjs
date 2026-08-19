// node assertions over the bundled module (esbuild first — see the sibling
// calibrateCommand.test.mjs for the same pattern).
import assert from 'node:assert/strict'
import { datasetKey, selectDataset, selectionKey, replayable } from '/tmp/datasetSelection.mjs'

const local = { root: '/data/mine', repo_id: 'me/mine', local: true, total_episodes: 4 }
const hub = { repo_id: 'lerobot/pusht', local: false, downloads: 1200 }

// exactly one field, never both
const a = selectDataset([local, hub], datasetKey(local))
assert.equal(a.dataset_root, '/data/mine')
assert.equal(a.dataset_repo_id, '', 'a local pick must NOT also send a repo id')
assert.equal(a.fromHub, false)
assert.match(a.label, /4 eps/)

const b = selectDataset([local, hub], datasetKey(hub))
assert.equal(b.dataset_repo_id, 'lerobot/pusht')
assert.equal(b.dataset_root, '', 'a Hub pick must NOT also send a path')
assert.equal(b.fromHub, true)
assert.match(b.label, /downloaded from the Hub/)

// a hub row and a local row can share a repo_id; the key keeps them apart
assert.notEqual(datasetKey({ repo_id: 'x/y', root: '/p' }), datasetKey({ repo_id: 'x/y' }))

// unknown / empty key clears rather than guesses
for (const k of ['', 'hub:gone/away', '/deleted/path']) {
  const s = selectDataset([local, hub], k)
  assert.equal(s.dataset_root, '')
  assert.equal(s.dataset_repo_id, '')
  assert.equal(s.label, '')
}

// round trip: form state -> select value -> same row
assert.equal(selectionKey(a), datasetKey(local))
assert.equal(selectionKey(b), datasetKey(hub))
assert.equal(selectionKey({ dataset_root: '', dataset_repo_id: '' }), '')
assert.equal(selectDataset([local, hub], selectionKey(b)).dataset_repo_id, 'lerobot/pusht')

// replay is offered only for what is actually on this disk
assert.equal(replayable(local).ok, true)
assert.equal(replayable(hub).ok, false)
assert.match(replayable(hub).reason, /not on this machine/)

console.log('datasetSelection: all assertions passed')
