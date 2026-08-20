import assert from 'node:assert/strict'
import { jointAbsence, expectsJoints, STATE_QUIET_S } from '/tmp/jointAbsence.mjs'

// The REAL so101-arm-1 document, copied from /api/fleet on 2026-08-20.
const NOW = 1787195402.7
const ARM1 = {
  presence: { robot_id: 'so101-arm-1', connected: true, hw: 'so_follower', cameras: ['top', 'wrist'] },
  state: { peer_id: 'so101-arm-1', t: 1787195402.368862, task: { status: 'idle' } },
}

// --- the defect: this rendered "no joint data on this peer" -------------------
{
  const n = jointAbsence({ ...ARM1, nowS: NOW })
  assert.equal(n.tone, 'attention')
  assert.match(n.text, /state is arriving, but carries no joint positions/)
  assert.doesNotMatch(n.text, /no joint data on this peer/)
  assert.match(n.hint, /devices → logs/)
  assert.match(n.hint, /lockout/, 'name the two candidates, do not pick one')
  assert.match(n.hint, /bus read/)
}

// --- an arm that has not spoken yet is a DIFFERENT situation ------------------
{
  const n = jointAbsence({ presence: { connected: true, action_keys: ['a', 'b', 'c'] }, state: null, nowS: NOW })
  assert.equal(n.tone, 'waiting')
  assert.match(n.text, /waiting for the first state frame \(3 joints expected\)/)
  assert.equal(n.hint, null)
}

// --- a peer that went quiet blames the process, not the bus ------------------
{
  const n = jointAbsence({ presence: { hw: 'so_leader' }, state: { t: NOW - 600 }, nowS: NOW })
  assert.equal(n.tone, 'attention')
  assert.match(n.text, /state went quiet 10m ago/)
  assert.match(n.hint, /process may have exited/)
}
{
  const n = jointAbsence({ presence: { hw: 'so_leader' }, state: { t: NOW - STATE_QUIET_S + 1 }, nowS: NOW })
  assert.match(n.text, /state is arriving/, 'inside the window is still live')
}

// --- a peer that genuinely has no joints must not be accused ------------------
{
  const cam = jointAbsence({ presence: { connected: true }, state: { t: NOW - 1 }, nowS: NOW })
  assert.equal(cam.tone, 'none')
  assert.match(cam.text, /publishes state without joint positions/)
  assert.equal(cam.hint, null)
  const silent = jointAbsence({ presence: {}, state: null, nowS: NOW })
  assert.equal(silent.text, 'no joint data on this peer', 'the old wording is right for exactly this case')
  assert.equal(silent.tone, 'none')
}

// --- what counts as "expects joints" ----------------------------------------
assert.equal(expectsJoints({ action_keys: ['x'] }), 1)
assert.equal(expectsJoints({ hw: 'so_follower' }), 'yes')
assert.equal(expectsJoints({ action_keys: [] }), 'unknown')
assert.equal(expectsJoints(null), 'unknown')

// --- clock skew must not invent a quiet period -------------------------------
{
  const n = jointAbsence({ presence: { hw: 'so_follower' }, state: { t: NOW + 5 }, nowS: NOW })
  assert.match(n.text, /state is arriving/, 'a frame from the future is skew, not silence')
}

console.log('jointAbsence: all assertions passed')
