// Q72: the arming rule of the record screen's "upload to the Hub" tick, as node assertions.
//
// The rule lives in RecordPanel (uploadBlocked / armedUpload) and cannot be imported without React,
// so it is restated here in one line and pinned. If the component's version drifts from this, the
// component is wrong: these are the cases that cost a whole recording session when they go wrong.
import assert from 'node:assert/strict'

/** exactly RecordPanel's expression */
const arming = (upload, pre, preErr, force) => {
  const blocked = !!upload && (!pre || (!pre.ok && !(pre.needs_force && force)))
  return { blocked, armed: upload && !blocked, preErr: !!preErr }
}

const READY = { ok: true, state: 'ready', needs_force: false, destination: 'me/ds', user: 'me', detail: '' }
const NO_CRED = { ok: false, state: 'no_credential', needs_force: false, destination: null, user: null, detail: '' }
const FOREIGN = { ok: false, state: 'foreign_namespace', needs_force: true, destination: 'org/ds', user: 'me', detail: '' }

// Untouched tick: nothing to check, nothing blocked, nothing armed.
assert.deepEqual(arming(false, null, false, false), { blocked: false, armed: false, preErr: false })

// The happy path arms.
assert.equal(arming(true, READY, false, false).armed, true)

// A certain refusal cannot be forced — ticking the force box (it is not even shown) changes nothing.
assert.equal(arming(true, NO_CRED, false, false).armed, false)
assert.equal(arming(true, NO_CRED, false, true).armed, false, 'no credential is not a matter of will')

// The one honest unknown is forceable, but only DELIBERATELY.
assert.equal(arming(true, FOREIGN, false, false).armed, false)
assert.equal(arming(true, FOREIGN, false, true).armed, true)

// Not asked yet, or the ask failed: stay disarmed. Guessing "probably fine" here spends a whole
// session to find out otherwise, and the session cannot be re-pushed from the dashboard.
assert.equal(arming(true, null, false, false).armed, false, 'in flight is not permission')
assert.equal(arming(true, null, true, false).armed, false, 'a failed check is not permission')
assert.equal(arming(true, null, true, true).armed, false)

console.log('uploadArming: all assertions passed')
