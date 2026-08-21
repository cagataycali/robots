// node assertions over the bundled module (esbuild first — see the sibling
// calibrateCommand.test.mjs for the same pattern).
import assert from 'node:assert/strict'
import { datasetKey, selectDataset, selectionKey, replayable, trainable, selectedRow, datasetMark } from '/tmp/datasetSelection.mjs'

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

// ---- Q37: a directory an abandoned recording left behind must not be replayed or trained on.
// The server writes the verdict onto the row (dataset_check.dataset_verdict); these assertions
// pin what the UI is allowed to conclude from it.
{
  const abandoned = {
    root: '/data/local/sim_recording', repo_id: 'local/sim_recording', total_episodes: 0, fps: 30,
    usable: false, reason: 'no_episodes',
    problem: '0 episodes. meta/info.json is written when a recording session OPENS, before the first episode is captured.',
  }

  // Replay reads EPISODE 0, which does not exist here - so the button goes dead with the
  // server's own sentence. Not re-worded locally: the server knows which failure mode this is.
  const r = replayable(abandoned)
  assert.equal(r.ok, false)
  assert.match(r.reason, /OPENS/)

  // Training too - and it is the more expensive click: it fails after the environment setup,
  // the base-model download and the dataset scan.
  const tr = trainable(abandoned)
  assert.equal(tr.ok, false)
  assert.match(tr.reason, /0 episodes/)

  // A row that is FINE is untouched by any of this.
  const good = { root: '/data/good', repo_id: 'org/good', total_episodes: 30, fps: 10, usable: true }
  assert.equal(replayable(good).ok, true)
  // The sentence deliberately names NO episode number: since the replay button gained an episode box,
  // the number comes from episodeChoice and the tooltip composes both — a hardcoded "episode 0" here
  // would let the two disagree in front of the operator. What must survive is the promise about WHERE
  // it happens, because that is why the operator looks at the fleet grid afterwards.
  assert.match(replayable(good).reason, /live mesh sim/)
  assert.doesNotMatch(replayable(good).reason, /episode \d/, 'the index belongs to episodeChoice now')
  assert.equal(trainable(good).ok, true)

  // AN OLDER SERVER SENDS NO VERDICT. A missing verdict is not a bad verdict: behave exactly as
  // before, or every dataset on a dashboard that has not been restarted becomes unusable.
  const unknown = { root: '/data/legacy', repo_id: 'org/legacy', total_episodes: 12, fps: 30 }
  assert.equal(replayable(unknown).ok, true)
  assert.equal(trainable(unknown).ok, true)

  // Hub rows: still refused for replay for the ORIGINAL reason (not here yet), never for Q37 -
  // nothing on this machine can inspect a dataset it has not downloaded.
  const hub = { repo_id: 'lerobot/pusht', local: false, downloads: 900 }
  assert.equal(replayable(hub).ok, false)
  assert.match(replayable(hub).reason, /not on this machine/)
  assert.equal(trainable(hub).ok, true)

  // Nothing picked yet is not a refusal to start - the form has its own required-field message.
  assert.equal(trainable(null).ok, true)

  // A verdict with no sentence still refuses, with something a human can act on.
  assert.equal(replayable({ root: '/d', repo_id: 'a/b', usable: false }).ok, false)
  assert.match(trainable({ root: '/d', repo_id: 'a/b', usable: false }).reason, /no episodes/)

  // selectedRow: the round trip the submit gate depends on. A selection that is no longer in
  // the list resolves to null (the list changed under the operator), and null must not refuse.
  const rows = [abandoned, good, hub]
  assert.equal(selectedRow(rows, { dataset_root: '/data/local/sim_recording', dataset_repo_id: '' })?.repo_id, 'local/sim_recording')
  assert.equal(selectedRow(rows, { dataset_root: '', dataset_repo_id: 'lerobot/pusht' })?.repo_id, 'lerobot/pusht')
  assert.equal(selectedRow(rows, { dataset_root: '/data/gone', dataset_repo_id: '' }), null)
  assert.equal(selectedRow(rows, { dataset_root: '', dataset_repo_id: '' }), null)
}

// ---- Q38: a dataset being recorded into RIGHT NOW is a different thing from an abandoned one.
{
  const live = {
    root: '/data/local/sim_recording', repo_id: 'local/sim_recording', total_episodes: 0, fps: 30,
    usable: false, recording: true, reason: 'recording_in_progress',
    problem: 'a recording session is writing into this dataset right now - 2 episode(s) captured so far. '
      + 'Wait for the session to close; do NOT delete the folder.',
  }
  const abandoned = {
    root: '/data/local/old', repo_id: 'local/old', total_episodes: 0, fps: 30,
    usable: false, reason: 'no_episodes', problem: '0 episodes. ... Record into it, or delete it.',
  }

  // The two must not wear the same glyph: ⚠ on a recording the operator is deliberately making
  // says "something is wrong here", which is a lie about the one thing that is going right.
  assert.equal(datasetMark(live).glyph, '⏺ ')
  assert.equal(datasetMark(live).kind, 'recording')
  assert.equal(datasetMark(abandoned).glyph, '⚠ ')
  assert.equal(datasetMark(abandoned).kind, 'problem')
  assert.equal(datasetMark({ root: '/d', repo_id: 'a/b', usable: true }).glyph, '')
  // No verdict at all (older server) is not a problem to announce.
  assert.equal(datasetMark({ root: '/d', repo_id: 'a/b' }).kind, 'ok')

  // Both verbs still refuse it - a growing dataset is not trainable and replay would race the
  // writer - but with the sentence that tells the operator to WAIT, not to delete.
  assert.equal(trainable(live).ok, false)
  assert.match(trainable(live).reason, /do NOT delete the folder/)
  assert.equal(replayable(live).ok, false)
  assert.match(replayable(live).reason, /right now/)

  // recording wins over the abandoned reading even though both carry usable:false.
  assert.equal(datasetMark({ ...abandoned, recording: true }).kind, 'recording')
}

// ── an EMPTY dataset row, from a server too old to judge it (measured on this machine) ──
// /Users/cagatay/.cache/huggingface/lerobot/local/sim_recording as the RUNNING dashboard reports it:
// no `usable`, no `problem`, and a count of zero. Source-side dataset_verdict already calls this
// unusable; the page must reach the same answer without it.
const emptyRow = {
  root: '/Users/cagatay/.cache/huggingface/lerobot/local/sim_recording',
  repo_id: 'local/sim_recording', total_episodes: 0, total_frames: 0, fps: 30, robot_type: 'unknown',
}
assert.ok(!trainable(emptyRow).ok, 'training on an empty folder must be refused by the page too')
assert.match(trainable(emptyRow).reason, /abandoned session/, 'the reason explains why the folder exists')
assert.ok(!replayable(emptyRow).ok, 'there is no episode 0 to replay')
assert.equal(datasetMark(emptyRow).kind, 'problem', 'a row both buttons refuse must not look normal')

// A server that DOES judge it wins: its sentence knows which failure mode this is.
const judged = { ...emptyRow, usable: false, problem: 'meta/info.json lists 3 episodes but data/ is empty' }
assert.match(trainable(judged).reason, /data\/ is empty/, "the server's own wording is not overwritten")

// NO EVIDENCE stays allowed: a row without counts is not a row known to be empty.
assert.ok(trainable({ root: '/x', repo_id: 'a/b' }).ok, 'an unknown count blocks nothing')
assert.ok(trainable({ root: '/x', repo_id: 'a/b', total_episodes: null }).ok, 'null is not zero')
assert.ok(trainable({ root: '/x', repo_id: 'a/b', total_episodes: 30, total_frames: 14595 }).ok)
// A row a recorder is writing into RIGHT NOW starts at zero episodes and is NOT a mistake.
assert.equal(datasetMark({ ...emptyRow, recording: true }).kind, 'recording')
