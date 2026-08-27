import assert from 'node:assert/strict'
import { jointAbsence, expectsJoints, failingForText, STATE_QUIET_S } from '/tmp/jointAbsence.mjs'

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
  // A log-derived verdict says where it came from, because it clears only if the log later
  // records a recovery -- which mesh.core now logs, but a robot running older code never does,
  // so the verdict can still describe a fault that is already over.
  const note = jointAbsence({ ...ARM1, problem: { ...UNCAL, source: undefined, for_seconds: undefined, failures: undefined }, nowS: ARM1.state.t + 1 })
  assert.equal(note.text, 'no joint positions — this board has no calibration, so its positions cannot be read in degrees')
  assert.match(note.detail, /read from this robot's log/)
  assert.match(note.detail, /clears only when the log records a recovery/)
  assert.doesNotMatch(note.detail, /consecutive failed reads/, 'a count nobody reported must not be invented')
}

{
  // One failure is not "1 consecutive failed reads".
  const note = jointAbsence({ ...ARM1, problem: { ...UNCAL, failures: 1, for_seconds: 2 }, nowS: ARM1.state.t + 1 })
  assert.doesNotMatch(note.detail, /consecutive failed reads/)
  assert.equal(note.text, 'no joint positions — this board has no calibration, so its positions cannot be read in degrees')
}

{
  // A verdict with a headline but NOTHING to show keeps detail null: an existing contract in
  // this file, and it caught the first version of the provenance clause, which would have
  // rendered a tooltip made only of "read from its log" -- no fact for the operator to weigh.
  const note = jointAbsence({ ...ARM1, problem: { kind: 'probe_failed', headline: 'the joint read failed' }, nowS: ARM1.state.t + 1 })
  assert.equal(note.detail, null)
  assert.equal(note.text, 'no joint positions — the joint read failed')
}

{
  const problem = {
    kind: 'uncalibrated',
    headline: 'this board has no calibration, so its positions cannot be read in degrees',
    remedy: 'Calibration files DO exist on this machine, so this is probably an id/path mismatch…',
    detail: 'RuntimeError: FeetechMotorsBus(...) has no calibration registered.',
  }
  // NO state document at all — the shape of an arm that failed its probe before publishing anything.
  const n = jointAbsence({ presence: { connected: true, hw: 'so_leader' }, state: null, problem, nowS: NOW })
  assert.match(n.text, /no state frames yet/)
  assert.match(n.text, /no calibration/, 'the headline is the news, not the missing frame')
  assert.equal(n.hint, problem.remedy)
  assert.equal(n.detail, problem.detail)
  assert.equal(n.tone, 'attention')
  // Unknown peer, no verdict: the old shrug is still exactly right.
  const shrug = jointAbsence({ presence: { connected: true }, state: null, nowS: NOW })
  assert.equal(shrug.text, 'no joint data on this peer')
  assert.equal(shrug.hint, null)
}
{
  // A peer gone QUIET, whose log says the bus was contended. The default hint rules the bus out —
  // a guess that this evidence contradicts, so the evidence must win.
  const problem = {
    kind: 'port_in_use',
    headline: 'another process holds this arm’s serial port',
    remedy: 'Find the other owner (devices → bus holders); replugging changes nothing.',
    detail: "[TxRxResult] Port is in use!",
  }
  const quiet = { presence: { connected: true, hw: 'so_follower' }, state: { t: NOW - 600 }, nowS: NOW }
  const withVerdict = jointAbsence({ ...quiet, problem })
  assert.match(withVerdict.text, /state went quiet 10m ago/)
  assert.equal(withVerdict.hint, problem.remedy)
  assert.ok(!/not the servo bus/.test(withVerdict.hint), 'never rule out the bus the log just blamed')
  // With no verdict the guess is the best available answer and stays.
  assert.match(jointAbsence(quiet).hint, /not the servo bus/)
}

{
  const alive = { connected: true, hw: 'so_follower' }
  const nowS = 1_000_000

  const behind = jointAbsence({ state: { t: nowS - 11 }, presence: alive, peerStale: false, nowS })
  assert.equal(behind.tone, 'attention', 'joints are still missing: this is not a neutral state')
  assert.match(behind.text, /11s behind/, 'the lag is still reported, it is a real fact')
  assert.doesNotMatch(behind.hint ?? '', /may have exited/, 'presence proves it did not exit')
  assert.match(behind.hint ?? '', /servo-bus|logs/, 'it points at the bus and the log that holds the reason')

  // A peer the dashboard's own ageing calls stale keeps the original sentence: there the guess is
  // the best available reading, because both rails agree the peer has gone.
  const gone = jointAbsence({ state: { t: nowS - 11 }, presence: alive, peerStale: true, nowS })
  assert.match(gone.text, /went quiet/)
  assert.match(gone.hint ?? '', /may have exited/)

  // UNKNOWN staleness is not "alive". Absent field => unchanged behaviour, so nothing that already
  // renders this component without the new prop silently changes meaning.
  for (const unknown of [undefined, null]) {
    const n = jointAbsence({ state: { t: nowS - 11 }, presence: alive, peerStale: unknown, nowS })
    assert.match(n.hint ?? '', /may have exited/, `peerStale=${unknown} must not be read as alive`)
  }

  // A backend verdict still outranks the new hint, exactly as it outranked the old guess.
  const withVerdict = jointAbsence({
    state: { t: nowS - 11 }, presence: alive, peerStale: false, nowS,
    problem: { headline: 'bus busy', remedy: 'stop the other owner of that port', detail: 'Port is in use!' },
  })
  assert.equal(withVerdict.hint, 'stop the other owner of that port')
  assert.equal(withVerdict.detail, 'Port is in use!')

  // And a state document INSIDE the window is untouched by any of this.
  const fresh = jointAbsence({ state: { t: nowS - 2 }, presence: alive, peerStale: false, nowS })
  assert.doesNotMatch(fresh.text, /behind|went quiet/)
}
