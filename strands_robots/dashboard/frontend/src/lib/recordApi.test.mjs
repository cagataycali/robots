// Run: npx esbuild src/lib/recordApi.ts --bundle --format=esm --outfile=/tmp/recordApi.mjs && node src/lib/recordApi.test.mjs
//
// Two things are tested here, and both decide whether a recording session is REAL:
//   1. the probe that chooses the live backend vs the in-browser rehearsal — a wrong answer either
//      hides a working recorder or pretends a dataset is being written;
//   2. the mock's fidelity to /api/record, because the rehearsal is where the UI's behaviour is
//      declared correct. A mock that succeeds where the backend refuses is how an error path gets
//      deleted as "unreachable".
// The probe is exercised through the REAL lib/endpoints with fetch stubbed, so this also covers the
// status → HttpError mapping that the 404 rule depends on.
import assert from 'node:assert/strict'

const store = new Map()
globalThis.localStorage = {
  getItem: k => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)), removeItem: k => store.delete(k),
}
globalThis.location = { search: '', host: 'dash.local', origin: 'http://dash.local' }
let reply = () => new Response('{}', { status: 200, headers: { 'content-type': 'application/json' } })
globalThis.fetch = async () => reply()

const mod = await import('/tmp/recordApi.mjs')

// ── the mock's state machine, which is what the rehearsal teaches the UI ──
const mock = mod.__mockForTests ? mod.__mockForTests() : null
// no test hook is exported, so drive the mock the way the app does: a 404 probe selects it.
reply = () => new Response(JSON.stringify({ detail: 'not found' }), { status: 404, headers: { 'content-type': 'application/json' } })
const api = await mod.getRecordApi()
assert.equal(api.mock, true, 'a genuinely missing route selects the rehearsal')

let s = await api.session()
assert.equal(s.dataset, null, 'no session is open before open()')
assert.equal(s.phase, 'idle')

// starting an episode with no session must refuse, exactly like the backend
await assert.rejects(() => api.startEpisode(), /no open session/)

s = await api.open({ dataset: 'me/so101-pick', task: 'pick the cube', leader: 'arm-1', follower: 'arm-2', target_episodes: 3 })
assert.equal(s.dataset, 'me/so101-pick')
assert.equal(s.fps, 30, 'an omitted fps means the backend default, never NaN')
const s60 = await api.open({ dataset: 'd', task: 't', leader: 'a', follower: 'b', target_episodes: 1, fps: 60 })
assert.equal(s60.fps, 60, 'a declared fps is honoured (Q54)')

s = await api.open({ dataset: 'me/so101-pick', task: 'pick the cube', leader: 'arm-1', follower: 'arm-2', target_episodes: 3 })
s = await api.startEpisode()
assert.equal(s.phase, 'recording')
assert.equal(s.episodes.length, 1, 'the in-flight take is listed while it records, as the real one is')
assert.equal(s.episodes[0].index, 0)
// a second start while recording is a no-op, not a second take
s = await api.startEpisode()
assert.equal(s.episodes.length, 1)
s = await api.stopEpisode()
assert.equal(s.phase, 'idle')
assert.ok(s.episodes[0].frames >= 1, 'a kept take has at least one frame')
// stop again: nothing to stop, nothing invented
s = await api.stopEpisode()
assert.equal(s.episodes.length, 1)

// redo drops the in-flight take, and the NEXT take reuses its index
s = await api.startEpisode()
assert.equal(s.episodes.length, 2)
s = await api.redoEpisode()
assert.equal(s.episodes.length, 1, 'a redone take leaves no trace')
s = await api.startEpisode()
assert.equal(s.episodes[s.episodes.length - 1].index, 1)
await api.stopEpisode()

// discard MARKS, never removes — the episode numbering in the dataset is the backend's, and a row
// that vanished would make the operator think a take was renumbered
s = await api.discard(0)
assert.equal(s.episodes.length, 2)
assert.equal(s.episodes[0].discarded, true)
assert.notEqual(s.episodes[1].discarded, true)

// THE FIDELITY FIX: the real route 404s on an unknown index (record_worker.discard raises KeyError)
// and refuses when no session is open (_require_open). The rehearsal used to succeed at both.
await assert.rejects(() => api.discard(99), /no saved episode with index 99/)
await api.close()
await assert.rejects(() => api.discard(0), /no open session/)

// a closed session is empty again, and the fresh session does NOT share the previous episodes array
s = await api.session()
assert.equal(s.dataset, null)
assert.deepEqual(s.episodes, [], 'no phantom takes carried across a close')

// the rehearsal never implies it can publish
const pre = await api.uploadPreflight()
assert.equal(pre.ok, false)
assert.equal(pre.state, 'no_credential')
assert.match(pre.detail, /nothing is written and nothing can be published/)

// ── the probe rule: ONLY a 404 selects the rehearsal ──
// (getRecordApi caches per page load, so each case needs a fresh module instance)
const fresh = async status => {
  reply = () => (status === 0
    ? Promise.reject(new TypeError('network down'))
    : new Response(JSON.stringify({ detail: 'x' }), { status, headers: { 'content-type': 'application/json' } }))
  const m = await import(`/tmp/recordApi.mjs?case=${status}`)
  return (await m.getRecordApi()).mock
}
assert.equal(await fresh(404), true, '404 = the route genuinely does not exist')
assert.equal(await fresh(401), false, 'a 401 means the backend IS there — the auth gate will sort the token out')
assert.equal(await fresh(403), false)
assert.equal(await fresh(500), false, 'a broken backend is still the real recorder')
assert.equal(await fresh(0), false, 'a network blip must never swap a real recorder for a mock that pretends to write datasets')
assert.equal(await fresh(200), false)

console.log('recordApi: all assertions passed')
