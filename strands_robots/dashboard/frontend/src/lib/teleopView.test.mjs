import assert from 'node:assert/strict'
import { teleopView, stopVerdict, startVerdict } from '/tmp/teleopView.mjs'

// UNASKED IS NOT IDLE. Every shape of "no answer" says nothing at all: an arm that IS streaming would
// otherwise render as quiet, which is the exact lie the raw counters told on real hardware.
for (const empty of [null, undefined, {}, { peer_id: 'x' }, { health: null }, { health: 'nope' }])
  assert.equal(teleopView(empty), null, `no health payload says nothing: ${JSON.stringify(empty)}`)

// An ANSWERED request with nothing running may say idle — that is a measurement, not an assumption.
assert.deepEqual(teleopView({ health: { receivers: {}, publishers: {} } }),
  { tone: 'idle', headline: 'no teleop on this arm', streaming: false,
    detail: 'it is neither following another arm nor publishing its own joints' })

const refusing = teleopView({ health: { worst: { state: 'refusing', headline: 'every frame is being refused',
  detail: 'shoulder_lift.pos |46.4| > 12.57', refusal: { kind: 'envelope' } }, publishers: {}, receivers: {} } })
assert.equal(refusing.tone, 'warn')
assert.equal(refusing.consentKind, 'teleop_degree_units', 'the operator is one consent away, and the screen must say which')
assert.equal(refusing.streaming, true, 'frames ARE on the wire — a stop button must be live')
assert.match(refusing.detail, /shoulder_lift/, 'the joint and the numbers survive to the screen')

// A healthy follower, and a refusal is NOT invented when the server did not report one.
const following = teleopView({ health: { worst: { state: 'following', headline: 'following so101-leader at 9.8Hz' } } })
assert.deepEqual({ ...following }, { tone: 'ok', headline: 'following so101-leader at 9.8Hz', detail: null, streaming: true })
assert.equal('consentKind' in following, false, 'no refusal reported → no consent nag')

// unrouted / silent are warnings, stopped is not a fault.
assert.equal(teleopView({ health: { worst: { state: 'unrouted', headline: 'nothing reaches this follower' } } }).tone, 'warn')
assert.equal(teleopView({ health: { worst: { state: 'silent', headline: 'no frames are arriving' } } }).tone, 'warn')
const stopped = teleopView({ health: { worst: { state: 'stopped', headline: 'not following anything' } } })
assert.equal(stopped.tone, 'idle'); assert.equal(stopped.streaming, false)

// An unknown state from a newer server is a WARNING, not silence: the screen must not swallow news it
// cannot classify (a future state would otherwise render as fine).
assert.equal(teleopView({ health: { worst: { state: 'wat', headline: 'something new' } } }).tone, 'warn')

// A LEADER (publishers only, no receiver) is not broken: its rate is the headline and the pairing is
// the operator's next step.
const leader = teleopView({ health: { receivers: {}, publishers: { so101: { state: 'publishing', headline: '176 frames at 9.4Hz', detail: null } } } })
assert.equal(leader.tone, 'ok'); assert.equal(leader.streaming, true)
assert.match(leader.headline, /publishing so101: 176 frames at 9\.4Hz/)
const idleLeader = teleopView({ health: { receivers: {}, publishers: { so101: { state: 'stopped', headline: '0 frames at 0.0Hz' } } } })
assert.equal(idleLeader.tone, 'idle'); assert.equal(idleLeader.streaming, false)

// The server's own worst-first ordering is respected when `worst` is absent but receivers exist.
const fallback = teleopView({ health: { receivers: { 'a': { state: 'refusing', headline: 'refused' } } } })
assert.equal(fallback.tone, 'warn')
// SLICE 2: a stop is only "stopped" when the arm SAYS SO on a re-ask.
const stillOn = stopVerdict(teleopView({ health: { worst: { state: 'following', headline: 'following so101-leader at 9.8Hz' } } }))
assert.equal(stillOn.ok, false, 'frames still on the wire is a FAILED stop, however clean the POST was')
assert.match(stillOn.line, /STILL/)
const refusingAfter = stopVerdict(teleopView({ health: { worst: { state: 'refusing', headline: 'every frame is being refused' } } }))
assert.equal(refusingAfter.ok, false, 'a refusing stream is still a stream — the arm is being commanded')
const gone = stopVerdict(teleopView({ health: { receivers: {}, publishers: {} } }))
assert.equal(gone.ok, true); assert.match(gone.line, /teleop stopped/)
assert.equal(stopVerdict(teleopView({ health: { worst: { state: 'stopped', headline: 'not following anything' } } })).ok, true)
// Silence after a stop is NOT success: the arm may be mid-refusal and unable to answer.
const silent = stopVerdict(null)
assert.equal(silent.ok, false); assert.match(silent.line, /nothing confirms/)

// SLICE 3b: starting is the dangerous direction, and this fleet has already produced the outcome that
// matters — "receive started" with all 176 frames refused (degrees into a radian envelope).
const live = startVerdict(teleopView({ health: { worst: { state: 'following', headline: 'following so101-leader at 9.8Hz' } } }))
assert.equal(live.ok, true); assert.match(live.line, /teleop live/)
const refused = startVerdict(teleopView({ health: { worst: { state: 'refusing', headline: 'every frame is being refused', refusal: 'out of range' } } }))
assert.equal(refused.ok, false, 'a refusing stream is NOT a working teleop session however cleanly it started')
assert.match(refused.line, /REFUSED/)
assert.match(refused.line, /teleop_degree_units/, 'the remedy is named where it is granted')
const nothing = startVerdict(teleopView({ health: { receivers: {}, publishers: {} } }))
assert.equal(nothing.ok, false)
assert.match(nothing.line, /45s/, 'a slow subscriber must not be reported as a failure')
assert.equal(startVerdict(null).ok, false)

console.log('teleopView: all assertions passed')
