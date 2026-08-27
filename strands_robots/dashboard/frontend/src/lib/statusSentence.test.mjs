import assert from 'node:assert/strict'
import { peerStatusFields, statusSentence, ribbonDetail } from '/tmp/statusSentence.mjs'

const base = { stale: false, lastSeenAgoS: 1, hwConnected: true, taskStatus: 'idle',
  instruction: null, taskDurationS: null, moving: false, stateAgeS: 0.2 }

// the ribbon shows the word as a chip; the sentence must not repeat it
const idle = statusSentence(base)
assert.equal(idle.word, 'idle')
assert.equal(idle.text, 'idle and still — safe to approach')
assert.equal(ribbonDetail(idle), 'and still — safe to approach')
// ...and the underlying sentence is untouched for every other consumer
assert.equal(idle.text, 'idle and still — safe to approach')

// a sentence that does NOT open with the word is passed through verbatim
const moving = statusSentence({ ...base, moving: true })
assert.equal(moving.word, 'moving')
assert.match(moving.text, /keep hands clear/)
assert.equal(ribbonDetail(moving), moving.text, 'must not rewrite a non-duplicate')

// stripping can never empty the detail — a chip alone says too little
assert.equal(ribbonDetail({ severity: 'ok', word: 'idle', text: 'idle' }), 'idle')
assert.equal(ribbonDetail({ severity: 'ok', word: 'idle', text: 'idle — ' }), 'idle — ')
// nor can a missing/odd word break it
assert.equal(ribbonDetail({ severity: 'ok', word: '', text: 'anything' }), 'anything')
assert.equal(ribbonDetail({ severity: 'ok', word: 'no hw', text: 'no hw connected — unplugged' }),
  'connected — unplugged')
// case-insensitive, and the leading punctuation goes with the word
// matching is case-insensitive: a chip reading WEDGED? still deduplicates
assert.equal(ribbonDetail({ severity: 'warn', word: 'Wedged?', text: 'wedged? maybe' }), 'maybe')
// ...but only a WHOLE leading word counts. 'idle' must not eat the front of
// 'idling along' and leave 'ing along' — the bug this assertion caught.
assert.equal(ribbonDetail({ severity: 'ok', word: 'idle', text: 'idling along nicely' }), 'idling along nicely')
assert.equal(ribbonDetail({ severity: 'ok', word: 'still', text: 'stillness is fine' }), 'stillness is fine')
assert.equal(ribbonDetail({ severity: 'warn', word: 'IDLE', text: 'idle: nothing to do' }), 'nothing to do')

// the safety-critical states keep their whole warning
for (const f of [{ ...base, stale: true }, { ...base, hwConnected: false },
                 { ...base, taskStatus: 'running', moving: false, taskDurationS: 9 }]) {
  const l = statusSentence(f)
  assert.ok(ribbonDetail(l).length > 12, `${l.word}: detail got thin: ${ribbonDetail(l)}`)
}
console.log('statusSentence/ribbonDetail: all assertions passed')

const idleBase = {
  stale: false, lastSeenAgoS: 2, hwConnected: true, taskStatus: 'idle',
  instruction: null, taskDurationS: null, moving: false, stateAgeS: 0.3,
}

const noJoints = statusSentence({
  ...idleBase, moving: null, jointsSeen: false,
})
assert.equal(noJoints.severity, 'warn', 'a peer with no joint stream is not a green card')
assert.equal(noJoints.word, 'idle?')
assert.doesNotMatch(noJoints.text, /safe to approach/, 'never claim safety without a measurement')
assert.match(noJoints.text, /publishes no joint positions/)
assert.match(noJoints.text, /treat the arm as able to move/)

// The transient case is DIFFERENT: the ring needs ~1s. Warning on every page
// load would cry wolf, so it stays 'ok' - but it still may not say "safe".
const measuring = statusSentence({ ...idleBase, moving: null, jointsSeen: true })
assert.equal(measuring.severity, 'ok')
assert.equal(measuring.word, 'idle')
assert.doesNotMatch(measuring.text, /safe to approach/)
assert.match(measuring.text, /motion not measured yet/)

// jointsSeen unknown (null/absent - nothing heard yet) must behave like the
// transient case, not like a verdict about the peer.
for (const js of [null, undefined]) {
  const unknown = statusSentence({ ...idleBase, moving: null, jointsSeen: js })
  assert.equal(unknown.severity, 'ok', `jointsSeen=${js} is not evidence of absence`)
  assert.doesNotMatch(unknown.text, /publishes no joint positions/)
  assert.doesNotMatch(unknown.text, /safe to approach/)
}

const measuredStill = statusSentence({ ...idleBase, moving: false, jointsSeen: true })
assert.equal(measuredStill.severity, 'ok')
assert.equal(measuredStill.text, 'idle and still \u2014 safe to approach')
const fabricated = statusSentence({ ...idleBase, moving: false, jointsSeen: false })
assert.equal(fabricated.severity, 'warn', 'stillness derived from an empty stream is not a measurement')
assert.equal(fabricated.word, 'idle?')
assert.doesNotMatch(fabricated.text, /safe to approach/)
// A peer with no joints that somehow reports MOVING still warns about motion
// first: an unexplained movement claim outranks the missing-stream complaint.
assert.equal(statusSentence({ ...idleBase, moving: true, jointsSeen: false }).word, 'moving')

// The higher-severity states are untouched by the new branch: a peer with no
// joints that is also stale/frozen/unplugged still reports the bigger problem.
assert.equal(statusSentence({ ...idleBase, stale: true, moving: null, jointsSeen: false }).word, 'offline')
assert.equal(statusSentence({ ...idleBase, stateAgeS: 30, moving: null, jointsSeen: false }).word, 'frozen')
assert.equal(statusSentence({ ...idleBase, hwConnected: false, moving: null, jointsSeen: false }).word, 'no hw')
assert.equal(statusSentence({ ...idleBase, taskStatus: 'running', moving: null, jointsSeen: false }).word, 'running')

// ribbonDetail must not mangle either new sentence.
assert.match(ribbonDetail(noJoints), /publishes no joint positions/)
assert.match(ribbonDetail(measuring), /motion not measured yet/)

console.log('statusSentence: silence-is-not-stillness assertions ok')

const TASK_STATUSES = ['idle', 'connecting', 'running', 'completed', 'stopped', 'error']

for (const status of TASK_STATUSES) {
  const line = statusSentence({ ...base, taskStatus: status })
  if (status === 'idle' || status === 'completed') {
    assert.equal(line.text, 'idle and still — safe to approach',
                 `${status} IS idle, and with motion measured still it earns the green sentence`)
  } else {
    assert.doesNotMatch(line.text, /safe to approach/, `Q93: "${status}" must never render the safety claim`)
  }
}

// connecting is the WORST possible moment for "safe to approach": the instant before torque engages.
const connecting = statusSentence({ ...base, taskStatus: 'connecting' })
assert.equal(connecting.word, 'starting')
assert.equal(connecting.severity, 'active')
assert.match(connecting.text, /torque can engage.*without warning/, 'and it says why to keep clear')

// a task that ENDED BADLY is not an idle robot: no safety claim, and it says where the arm is.
const failed = statusSentence({ ...base, taskStatus: 'error' })
assert.equal(failed.severity, 'warn')
assert.equal(failed.word, 'failed')
assert.match(failed.text, /stopped wherever it/, 'the arm did not go home; it stopped mid-task')

// `stopped` is deliberately NOT a warning — an operator pressing stop is normal, and an amber card
// after every normal stop is alarm fatigue. It only has to stop claiming safety.
const stopped = statusSentence({ ...base, taskStatus: 'stopped' })
assert.equal(stopped.severity, 'ok', 'pressing stop is not an anomaly')
assert.match(stopped.text, /resume/, 'but a resume moves an arm parked mid-task')

// AN UNRECOGNISED STATUS IS NO EVIDENCE, and no evidence cannot earn the green sentence.
const paused = statusSentence({ ...base, taskStatus: 'paused' })
assert.equal(paused.word, 'unknown')
assert.equal(paused.severity, 'warn')
assert.match(paused.text, /paused/, 'quote the state back, so the operator can look it up')
assert.match(paused.text, /stillness is not confirmed/)

// MOTION STILL WINS over a task-status sentence: a moving arm is the more urgent physical claim.
assert.equal(statusSentence({ ...base, taskStatus: 'error', moving: true }).word, 'moving',
             'an arm moving after a failed task is a MOVING arm first')
assert.equal(statusSentence({ ...base, taskStatus: 'stopped', moving: true }).word, 'moving')
// ...except during connecting, where motion is the robot homing itself, not a stranger commanding it.
const homing = statusSentence({ ...base, taskStatus: 'connecting', moving: true })
assert.equal(homing.word, 'starting')
assert.match(homing.text, /ALREADY MOVING/)
assert.equal(homing.severity, 'warn', 'and that IS worth amber — it moved before anyone asked')

// the dead-peer and hardware sentences still outrank every task status.
assert.equal(statusSentence({ ...base, taskStatus: 'error', stale: true }).word, 'offline')
assert.equal(statusSentence({ ...base, taskStatus: 'connecting', hwConnected: false }).word, 'no hw')
// whitespace and case come off the wire unevenly.
assert.equal(statusSentence({ ...base, taskStatus: ' ERROR ' }).word, 'failed')
assert.equal(statusSentence({ ...base, taskStatus: '' }).text, 'idle and still — safe to approach',
             'an empty status is the same silence as no status at all')

console.log('statusSentence: Q93 task-status assertions ok')

const locked = statusSentence({ ...base, lockout: 'locked' })
assert.equal(locked.word, 'locked')
assert.equal(locked.severity, 'warn')
assert.doesNotMatch(locked.text, /safe to approach/, 'a locked arm never gets the safety claim')
assert.match(locked.text, /commands are refused/, 'it says why it is still')
assert.match(locked.text, /clearing the lockout is what makes it live again/, 'and what would end it')

// A lockout means commands are REFUSED. Joints moving anyway is the worst state on the card: either the
// lockout is not holding, or something outside the mesh is driving the arm.
const escaping = statusSentence({ ...base, lockout: 'locked', moving: true })
assert.equal(escaping.severity, 'danger', 'this is the only card state that outranks warn')
assert.equal(escaping.word, 'locked?!')
assert.match(escaping.text, /lockout is not holding|outside the mesh/)

// The lockout explains a "running" task that is not moving, so it must OUTRANK the wedged
// accusation: a locked arm under a policy is not a wedged policy, and the remedy is the
// lockout, not a restart.
const lockedRunning = statusSentence({ ...base, lockout: 'locked', taskStatus: 'running', moving: false })
assert.equal(lockedRunning.word, 'locked', 'not "wedged?" — the lockout is the reason, and it is fixable')

assert.equal(statusSentence({ ...base, lockout: 'locked', taskStatus: 'error' }).word, 'locked')
assert.equal(statusSentence({ ...base, lockout: 'locked', taskStatus: 'connecting' }).word, 'locked')

// 'unknown' DELIBERATELY says nothing.
for (const state of ['unknown', 'clear', null, undefined, '']) {
  assert.equal(statusSentence({ ...base, lockout: state }).text, 'idle and still — safe to approach',
               `lockout ${JSON.stringify(state)} leaves the sentence to the motion measurement`)
}
// case and whitespace come off the wire unevenly, as everywhere else in this module
assert.equal(statusSentence({ ...base, lockout: ' LOCKED ' }).word, 'locked')

// but a dead peer or unplugged hardware still outranks the lockout: both mean the lockout state on the
// card is itself second-hand, and both have a different first action.
assert.equal(statusSentence({ ...base, lockout: 'locked', stale: true }).word, 'offline')
assert.equal(statusSentence({ ...base, lockout: 'locked', hwConnected: false }).word, 'no hw')

console.log('statusSentence: Q95 lockout assertions ok')

{
  const base = { stale: false, lastSeenAgoS: 2, hwConnected: true, taskStatus: 'idle',
    moving: null, jointsSeen: false, stateAgeS: 1 }

  const host = statusSentence({ ...base, hostsChildren: ['sim-a__so101'] })
  assert(host.severity === 'ok', `a host process must not be a warning, got ${host.severity}`)
  assert(host.word === 'process', `word should name what it is, got ${host.word}`)
  assert(/hosts sim-a__so101/.test(host.text), `the sentence must name the child: ${host.text}`)
  assert(!/able to move/.test(host.text), 'a process must not be described as able to move')

  const many = statusSentence({ ...base, hostsChildren: ['a__x', 'a__y'] })
  assert(/2 robots \(a__x, a__y\)/.test(many.text), `plural form should list them: ${many.text}`)

  // The mute ARM keeps its warning — this is the whole point of not diluting it.
  const arm = statusSentence({ ...base, hostsChildren: [] })
  assert(arm.severity === 'warn', 'a childless jointless peer is still a broken arm')
  assert(/able to move/.test(arm.text), `the true warning must survive: ${arm.text}`)
  const armNull = statusSentence({ ...base, hostsChildren: null })
  assert(armNull.severity === 'warn', 'absent host info must not soften the warning')

  // A parent that DOES publish joints is not silent at all, so this branch never applies:
  // jointsSeen true takes the ordinary path even with children.
  const busyParent = statusSentence({ ...base, jointsSeen: true, moving: false,
    hostsChildren: ['a__x'] })
  assert(busyParent.word !== 'process', 'a peer publishing joints is an arm, whatever it hosts')
}

{
  const peer = {
    last_seen: Date.now() / 1000 - 2, stale: false,
    presence: { connected: true, robot_type: 'robot', task_status: 'idle' },
    state: { task: { status: 'idle' } }, lockout: null,
  }
  const f = peerStatusFields(peer, { moving: false, jointsSeen: true, stateAgeS: 0.4 })
  assert(f.hwConnected === true, 'hardware connection must survive the mapping')
  assert(f.taskStatus === 'idle' && f.moving === false && f.jointsSeen === true, 'facts pass through')
  assert(f.lastSeenAgoS > 1 && f.lastSeenAgoS < 5, `heartbeat age should be seconds, got ${f.lastSeenAgoS}`)
  assert(statusSentence(f).text === 'idle and still — safe to approach', 'the calm case still reads calm')

  // The mute arm reads the SAME on both screens — that is the point of the shared builder.
  const mute = peerStatusFields(peer, { moving: null, jointsSeen: false, stateAgeS: 0.4 })
  assert(statusSentence(mute).severity === 'warn', 'a mute arm warns wherever it is rendered')

  const host = peerStatusFields(peer, { moving: null, jointsSeen: false, stateAgeS: 0.4 }, ['sim__so101'])
  assert(statusSentence(host).word === 'process', 'a host process is a process on both screens')

  // Absent telemetry must not fabricate measurements: undefined becomes null, never false.
  const empty = peerStatusFields(peer, {})
  assert(empty.moving === null && empty.jointsSeen === null && empty.stateAgeS === null,
    'missing telemetry is null, not a measurement')
}
