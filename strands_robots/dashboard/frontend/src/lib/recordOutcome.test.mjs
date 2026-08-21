// Assertions for what the collect panel may claim when a record action fails
// (lib/recordOutcome.ts).
// Run: npx esbuild src/lib/recordOutcome.ts --bundle --format=esm --outfile=/tmp/recordOutcome.mjs \
//        && node src/lib/recordOutcome.test.mjs
import assert from 'node:assert/strict'

const { recordFailure } = await import('/tmp/recordOutcome.mjs')

const KINDS = ['open', 'start', 'stop', 'redo', 'discard', 'close']

// THE BUG: every lost answer printed the raw error, reading as "the click did
// nothing". Each action's ambiguity has its own physical consequence.
{
  const v = recordFailure({ kind: 'open', status: 0, message: 'Load failed' })
  assert.equal(v.ambiguous, true)
  assert.match(v.text, /session MAY be open/)
  assert.match(v.text, /follower energised and stiff/)   // the arms already moved
  assert.match(v.text, /Re-reading the session now/)
}
{
  const v = recordFailure({ kind: 'start', status: 0, message: 'Load failed' })
  assert.match(v.text, /MAY already be recording/)
  // The two ways an operator ruins a take by trusting "it didn't start".
  assert.match(v.text, /only once you know/)
  assert.match(v.text, /do not walk away/)
  assert.equal(v.destructive, false)
}
{
  const v = recordFailure({ kind: 'stop', status: 0, message: 'Load failed' })
  assert.match(v.text, /MAY already be saved, or MAY still be recording/)
  assert.match(v.text, /cannot tell which/)
}
// redo/discard are the irreversible pair: say so, and flag it for the caller.
for (const kind of ['redo', 'discard']) {
  const v = recordFailure({ kind, status: 0, message: 'x' })
  assert.equal(v.destructive, true)
  assert.match(v.text, /MAY already/)
  assert.match(v.text, /cannot be undone/)
}
assert.equal(recordFailure({ kind: 'stop', status: 0, message: 'x' }).destructive, false)
assert.equal(recordFailure({ kind: 'open', status: 0, message: 'x' }).destructive, false)

// close: the ambiguity reaches OUTSIDE this machine when upload was ticked.
{
  const v = recordFailure({ kind: 'close', status: 0, message: 'Load failed' })
  assert.equal(v.ambiguous, true)
  assert.equal(v.destructive, false)
  assert.match(v.text, /MAY already be finished/)
  assert.match(v.text, /if you ticked upload, MAY already be on the Hub/)
}
assert.match(recordFailure({ kind: 'close', status: 401, message: 'no' }).text, /NOT finished/)
assert.match(recordFailure({ kind: 'close', status: 401, message: 'no' }).text, /nothing was uploaded/)

// A 5xx ran the handler: ambiguous, code kept.
{
  const v = recordFailure({ kind: 'stop', status: 503, message: 'unavailable' })
  assert.equal(v.ambiguous, true)
  assert.match(v.text, /failed mid-request \(503: unavailable\)/)
}

// Only a pre-handler refusal may say the recorder was untouched — and for stop
// that claim must be the alarming one (it is still recording).
for (const status of [400, 401, 403, 404, 422, 429]) {
  assert.match(recordFailure({ kind: 'open', status, message: 'no' }).text, /no session was opened/)
  assert.match(recordFailure({ kind: 'start', status, message: 'no' }).text, /nothing is being recorded/)
  const stop = recordFailure({ kind: 'stop', status, message: 'no' })
  assert.match(stop.text, /NOT stopped/)
  assert.match(stop.text, /it still is/)
  assert.match(recordFailure({ kind: 'redo', status, message: 'no' }).text, /still there/)
  for (const kind of KINDS) {
    const v = recordFailure({ kind, status, message: 'no' })
    assert.equal(v.ambiguous, false)
    assert.equal(v.destructive, false)
    // A refusal must never send the operator hunting in the episode list.
    assert.doesNotMatch(v.text, /MAY/)
  }
}

// Every kind says something different in each world, and the two worlds differ.
const seen = new Set()
for (const kind of KINDS) {
  const a = recordFailure({ kind, status: 0, message: 'x' }).text
  const b = recordFailure({ kind, status: 401, message: 'x' }).text
  assert.notEqual(a, b)
  seen.add(a); seen.add(b)
  assert.doesNotMatch(a, /undefined|null/)
  assert.doesNotMatch(b, /undefined|null/)
}
assert.equal(seen.size, KINDS.length * 2)

// Missing detail never renders as "undefined" or an empty paren pair.
for (const kind of KINDS) {
  for (const f of [{ kind, status: 0 }, { kind, status: 401 }, { kind, status: 500, message: '   ' }]) {
    const v = recordFailure(f)
    assert.match(v.text, /no detail/)
    assert.doesNotMatch(v.text, /\(\)/)
  }
}

// An unmodelled 4xx is not assumed inert (one classifier, five screens).
assert.equal(recordFailure({ kind: 'redo', status: 418, message: 'x' }).ambiguous, true)

console.log('recordOutcome: all assertions passed')

// ── Q101: the inert verdict for `open` must not promise a fleet it cannot see ──
// A refusal can be raised AFTER both arms were parked, and the respawn can fail; the server names the
// arm that did not come back in the same message. A blanket "the arms are untouched" contradicted it in
// one toast, and the reassurance was the wrong half to keep.
const refusedOpen = recordFailure({ kind: 'open', status: 422, message: 'fps must be positive' })
assert.equal(refusedOpen.ambiguous, false)
assert.doesNotMatch(refusedOpen.text, /untouched/)
assert.match(refusedOpen.text, /unless the message above says otherwise/,
  'it defers to the server\'s own sentence about the fleet instead of overruling it')
assert.match(refusedOpen.text, /no session was opened/, 'what IS certain is still stated plainly')

console.log('recordOutcome: Q101 inert-open wording ok')
