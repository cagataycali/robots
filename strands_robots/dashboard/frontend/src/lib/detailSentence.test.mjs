import assert from 'node:assert/strict'
import { detailSentence } from '/tmp/detailSentence.mjs'

// ── REAL bodies, measured against the running dashboard on 2026-08-21 ──
// GET /api/devices/logs/no-such-peer. Someone wrote that hint and listed the three peers that WOULD
// have worked; the screen used to show JSON.stringify of it.
const LOGS_404 = {
  error: 'unknown peer no-such-peer',
  hint: 'only locally spawned robots keep a log ring buffer',
  managed_peers: ['so101-follower', 'so101-follower-twin', 'so101-leader'],
}
assert.equal(detailSentence(LOGS_404),
  'unknown peer no-such-peer — only locally spawned robots keep a log ring buffer '
  + '(managed peers: so101-follower, so101-follower-twin, so101-leader)')

// server.py:705 — a command aimed at a peer that is not on the mesh.
assert.equal(detailSentence({
  error: "no peer 'so101-arm-9' in the fleet",
  hint: 'GET /api/fleet lists the peers that can be commanded',
  known_peers: ['so101-follower', 'so101-leader'],
}), "no peer 'so101-arm-9' in the fleet — GET /api/fleet lists the peers that can be commanded "
  + '(known peers: so101-follower, so101-leader)')

// auth.py:346 — error AND detail AND hint: the general refusal, the specific cause, the remedy.
assert.equal(detailSentence({
  error: 'this host cannot be used for a passkey ceremony',
  detail: 'rp_id 1.0.0.0.ip6.arpa is not a registrable domain',
  hint: 'reach the dashboard on its enrolled hostname',
}), 'this host cannot be used for a passkey ceremony — rp_id 1.0.0.0.ip6.arpa is not a registrable '
  + 'domain — reach the dashboard on its enrolled hostname')

// server.py:1983 — the alternatives are OBJECTS. Naming them beats dumping them.
assert.equal(detailSentence({
  error: "'leader' exists 2 times - say which with ?device_type=&device_model=",
  candidates: [{ name: 'leader', device_type: 'robots' }, { name: 'leader', device_type: 'teleoperators' }],
}), "'leader' exists 2 times - say which with ?device_type=&device_model= (candidates: leader, leader)")
// ...and an unnameable list counts instead of dumping: the count IS the actionable part.
assert.equal(detailSentence({ error: 'ambiguous', candidates: [{ a: 1 }, { b: 2 }] }),
  'ambiguous (candidates: 2)')

// A long list never grows past the toast, and says how many it is holding back.
const many = { error: 'unknown peer x', known_peers: ['a','b','c','d','e','f','g','h'] }
assert.equal(detailSentence(many), 'unknown peer x (known peers: a, b, c, d, e, f and 2 more)')

// A FastAPI validation error is a LIST of {loc,msg}: "body" is plumbing, the last segment is the field.
assert.equal(detailSentence([
  { type: 'missing', loc: ['body', 'dataset'], msg: 'Field required' },
  { type: 'int_parsing', loc: ['body', 'target_episodes'], msg: 'Input should be a valid integer' },
]), 'dataset: Field required; target_episodes: Input should be a valid integer')
// A whole-body error has no field to name: "body: Field required" is plumbing talking to the operator.
// (This case was missing, and a surviving mutation proved it: the filter only bites here.)
assert.equal(detailSentence([{ type: 'missing', loc: ['body'], msg: 'Field required' }]), 'Field required')

// ── the ordinary case stays untouched ──
assert.equal(detailSentence('leader peer_id is required'), 'leader peer_id is required')
assert.equal(detailSentence('  padded  '), 'padded')
assert.equal(detailSentence(['just', 'strings']), 'just; strings')

// ── NEVER LOSSY: an unrecognised shape shows its JSON rather than a tidy summary that drops a field ──
assert.equal(detailSentence({ weird: 'shape', code: 7 }), '{"weird":"shape","code":7}')
assert.equal(detailSentence([{ no: 'msg' }]), '[{"no":"msg"}]')
assert.equal(detailSentence({ error: '', hint: '' }), '{"error":"","hint":""}',
  'an object whose fields are all blank has told us nothing - show the truth, not an empty toast')
assert.equal(detailSentence(null), '')
assert.equal(detailSentence(undefined), '')
// A hint with no error still helps; an empty list adds no noise.
assert.equal(detailSentence({ hint: 'restart it from a terminal', known_peers: [] }),
  'restart it from a terminal')

console.log('detailSentence: real measured bodies read as sentences')
