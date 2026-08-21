// Run: node scripts/run-lib-tests.mjs taskPhase
//
// This table decides whether the operator sees a task as still running — and therefore whether the STOP
// button is on the screen while an arm is moving. It lived in useTask's body, reachable only by rendering
// a card against a live peer, so nothing tested it.
import assert from 'node:assert/strict'

const { reportedTaskStatus, isRunningStatus, nextPhase, deriveTaskFlags } = await import('/tmp/taskPhase.mjs')

// ── 1. where the peer's words come from, and what "no words" looks like ──
assert.equal(reportedTaskStatus({ state: { task: { status: 'running' } } }), 'running', 'state.task.status')
assert.equal(reportedTaskStatus({ presence: { task_status: 'idle' } }), 'idle', 'presence.task_status is the fallback')
assert.equal(reportedTaskStatus({ state: { task: { status: 'running' } }, presence: { task_status: 'idle' } }),
             'running', 'the state topic wins over presence — it is the fresher channel')
assert.equal(reportedTaskStatus({}), undefined, 'a peer that says nothing about tasks')
assert.equal(reportedTaskStatus({ presence: {} }), undefined, 'presence without the field says nothing')
assert.equal(reportedTaskStatus({ presence: { task_status: null } }), undefined,
             'a null is not a status — it must not be compared as a word')

assert.equal(isRunningStatus('running'), true)
assert.equal(isRunningStatus('executing'), true, 'both words the robot uses')
assert.equal(isRunningStatus('idle'), false)
assert.equal(isRunningStatus(undefined), false, 'silence is not running…')

// ── 2. Q87 — THE LAW: absence of a status is NOT a completion ──
// `!reportedRunning` used to be true both when the robot said "idle" and when it said NOTHING, so one
// presence payload that lost its task_status (mesh/core wraps the read in a bare try/except, and some
// peers never report one) flipped the UI to "done" mid-task — which also drops `running`, taking the
// STOP button off the screen while the arm is still executing.
assert.equal(nextPhase('running', undefined), null,
             'Q87: a peer that goes QUIET does not complete its task — hold the phase, keep stop reachable')
assert.equal(nextPhase('running', 'idle'), 'done',
             'a robot that AFFIRMATIVELY says idle has finished — hardware_robot always reports a status ' +
             'value for a finished task, so this is the real completion path')
assert.equal(nextPhase('running', 'completed'), 'done', 'any affirmative non-running word ends our optimism')
assert.equal(nextPhase('running', 'error'), 'done', 'including error — it is still not executing')
assert.equal(nextPhase('running', 'running'), null, 'already running: no state change, no re-render')

// The peer's report also RESCUES the phase: a task started outside this browser shows as running.
assert.equal(nextPhase('idle', 'running'), 'running', 'someone else started it — adopt the robot\'s truth')
assert.equal(nextPhase('failed', 'executing'), 'running', 'a peer that is demonstrably executing overrides "failed"')

// Settled words are not re-opened by a status arriving late.
for (const phase of ['idle', 'starting', 'stopping', 'failed', 'done']) {
  assert.equal(nextPhase(phase, 'idle'), null, `an idle report does not disturb phase "${phase}"`)
  assert.equal(nextPhase(phase, undefined), null, `silence does not disturb phase "${phase}"`)
}

// ── 3. flags: the robot's report wins over our optimism, in the direction of safety ──
const flags = (phase, reported, twinBusy = false) => deriveTaskFlags({ phase, reported, twinBusy })
assert.equal(flags('idle', 'running').running, true, 'the robot says running, so it is running')
assert.equal(flags('starting', undefined).running, true,
             'our own request counts as running BEFORE any report — the stop button must exist in the gap')
assert.equal(flags('running', undefined).running, true, 'and it keeps existing while the peer is quiet')
assert.equal(flags('done', 'idle').running, false, 'both agree it is over')
assert.equal(flags('failed', undefined).running, false,
             'a failed START is not running: physicalFail already said whether the arm may be moving')

assert.equal(flags('starting', undefined).busy, true, 'busy while starting')
assert.equal(flags('stopping', 'running').busy, true, 'busy while stopping')
assert.equal(flags('running', 'running').busy, false, 'a running task is not "busy" — stop must be clickable')
assert.equal(flags('idle', 'idle', true).busy, true, 'a twin toggle in flight blocks the controls')

console.log('taskPhase.test.mjs: all assertions passed')
