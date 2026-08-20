import assert from 'node:assert/strict'
import { jointAbsence, expectsJoints, failingForText, STATE_QUIET_S } from '/tmp/jointAbsence.mjs'

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

// ------------------------------------------------------------------ Q80: the backend knows now
// The module's own docstring says it must not GUESS the cause. Since Q80 the cause can arrive with
// the peer (annotation `joint_problem`, read from the child's log server-side), and then withholding
// it would be the dishonest choice: a held serial port and an uncalibrated board are opposite
// remedies that both used to render as the same "check its log" shrug.
{
  const arriving = { state: { t: 1000 }, presence: { hw: 'so_follower', action_keys: ['a', 'b'] }, nowS: 1000.3 }

  const held = jointAbsence({
    ...arriving,
    problem: {
      kind: 'port_in_use',
      headline: "another process is holding this arm's serial port",
      remedy: 'Find the other owner and stop it - `/usr/sbin/lsof -n | grep usbmodem` names every holder.',
      detail: 'ConnectionError("... [TxRxResult] Port is in use!")',
    },
  })
  assert.equal(held.tone, 'attention')
  assert.match(held.text, /holding this arm's serial port/)
  assert.match(held.hint, /lsof/)
  assert.match(held.detail, /Port is in use/)
  assert.ok(!/both look like this/.test(held.hint), 'the shrug must be gone when evidence exists')

  const uncal = jointAbsence({
    ...arriving,
    problem: { kind: 'uncalibrated', headline: 'this board has no calibration', remedy: 'Calibrate this arm.' },
  })
  assert.match(uncal.text, /no calibration/)
  assert.ok(!/lsof/.test(uncal.hint), 'opposite remedy: never conflate the two')
  assert.equal(uncal.detail, null, 'a missing detail is null, not "undefined" text')

  // Absent verdict => the old honest wording, unchanged. A backend that cannot tell must not make
  // this module quieter than it was.
  const blind = jointAbsence(arriving)
  assert.match(blind.text, /carries no joint positions/)
  assert.match(blind.hint, /check its log/)

  // A verdict with no headline is not a verdict.
  const empty = jointAbsence({ ...arriving, problem: { kind: 'probe_failed' } })
  assert.match(empty.text, /carries no joint positions/)

  // Joints ARRIVING is the bridge's job to gate, but a quiet peer must still be reported as quiet:
  // the process being gone is not a bus fault, whatever the log once said.
  const quiet = jointAbsence({
    state: { t: 1000 }, presence: { hw: 'so_follower' }, nowS: 1120,
    problem: { kind: 'port_in_use', headline: "another process is holding this arm's serial port" },
  })
  assert.match(quiet.text, /went quiet/)
  console.log('  ✓ Q80 backend verdict is used when present, and only when it applies')
}

// --- the fault's AGE and its provenance (Q85/Q86) -------------------------------------------------
// Two of cagatay's arms sat silent for 3.5 hours while their cards said only "no joints". The
// backend now publishes how long and how often, from the robot itself.

assert.equal(failingForText(3), null, 'a 3s-old fault is as likely a transient: say nothing')
assert.equal(failingForText(45), 'for 45s')
assert.equal(failingForText(600), 'for 10m')
assert.equal(failingForText(12600), 'for 3.5h')
assert.equal(failingForText(null), null)
assert.equal(failingForText(Number.NaN), null, 'NaN must not render as "for NaNs"')

const UNCAL = {
  kind: 'uncalibrated',
  headline: 'this board has no calibration, so its positions cannot be read in degrees',
  remedy: 'Calibrate this arm (devices > calibrate).',
  detail: 'RuntimeError: has no calibration registered.',
  source: 'peer',
  failures: 900,
  for_seconds: 12600,
}

{
  const note = jointAbsence({ ...ARM1, problem: UNCAL, nowS: ARM1.state.t + 1 })
  assert.equal(note.text, 'no joint positions for 3.5h — this board has no calibration, so its positions cannot be read in degrees')
  assert.equal(note.tone, 'attention')
  assert.match(note.hint, /Calibrate this arm/)
  // The tooltip carries what the one-line sentence cannot.
  assert.match(note.detail, /900 consecutive failed reads/)
  assert.match(note.detail, /reported by the robot itself/)
  assert.match(note.detail, /RuntimeError: has no calibration registered/)
}

{
  // A log-derived verdict says so, because it CANNOT clear itself: mesh.core logs a failure once
  // and never a recovery, so it may describe a fault that is already over.
  const note = jointAbsence({ ...ARM1, problem: { ...UNCAL, source: undefined, for_seconds: undefined, failures: undefined }, nowS: ARM1.state.t + 1 })
  assert.equal(note.text, 'no joint positions — this board has no calibration, so its positions cannot be read in degrees')
  assert.match(note.detail, /never a recovery/)
  assert.doesNotMatch(note.detail, /consecutive failed reads/, 'a count nobody reported must not be invented')
}

{
  // One failure is not "1 consecutive failed reads".
  const note = jointAbsence({ ...ARM1, problem: { ...UNCAL, failures: 1, for_seconds: 2 }, nowS: ARM1.state.t + 1 })
  assert.doesNotMatch(note.detail, /consecutive failed reads/)
  assert.equal(note.text, 'no joint positions — this board has no calibration, so its positions cannot be read in degrees')
}

{
  // A verdict with a headline but NOTHING to show keeps detail null: an existing contract in this
  // file, and it caught the first version of the provenance clause, which would have rendered a
  // tooltip made only of "read from its log" -- no fact for the operator to weigh.
  const note = jointAbsence({ ...ARM1, problem: { kind: 'probe_failed', headline: 'the joint read failed' }, nowS: ARM1.state.t + 1 })
  assert.equal(note.detail, null)
  assert.equal(note.text, 'no joint positions — the joint read failed')
}
