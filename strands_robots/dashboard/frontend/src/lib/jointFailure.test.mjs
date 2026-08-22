/** Why an arm reports no joints, read from its own ring buffer.
 *  Subject: npx esbuild src/lib/jointFailure.ts --bundle --format=esm --outfile=/tmp/jointFailure.mjs */
import assert from 'node:assert/strict'
import { jointFailure, jointFailureLine } from '/tmp/jointFailure.mjs'

// VERBATIM from this fleet (GET /api/devices/logs/so101-follower), including the reassuring tail that
// postdates the failure — the reason a dead arm looks healthy on every other surface.
const FOLLOWER = [
  `13:58:49 WARNING:strands_robots.mesh.core:[safety:so101-follower] No emergency-stop resume code set.`,
  `13:58:52 WARNING:strands_robots.mesh.core:[mesh] so101-follower: state probe 'hw_joints' failed, that section of the snapshot is omitted (further failures logged at debug): ConnectionError("Failed to sync read 'Present_Position' on ids=[1,2,3,4,5,6] after 3 tries. [TxRxResult] Port is in use!")`,
  `13:58:52 hardware connected`,
  `13:58:52 so101-follower (real @ /dev/cu.usbmodem5AB01584281) online`,
]
const f = jointFailure(FOLLOWER)
assert.equal(f.kind, 'ConnectionError')
assert.match(f.headline, /holds .*cannot read its own position|something else on this machine holds/)
assert.match(f.remedy, /more than one owner/)
assert.match(f.remedy, /nothing needs unplugging/, 'a bus collision must not send anyone to the cables')
assert.equal(f.tailMisleads, true, 'the log says "hardware connected" AFTER the failure')
assert.match(jointFailureLine(f), /that is the PROCESS, not the joints/)

const LEADER = [
  `13:59:22 WARNING:strands_robots.mesh.core:[mesh] so101-leader: state probe 'hw_joints' failed, that section of the snapshot is omitted (further failures logged at debug): RuntimeError("FeetechMotorsBus(Port '/dev/cu.usbmodem5AB01818061', 6 sts3215 motors) has no calibration registered")`,
  `13:59:23 hardware connected`,
]
const l = jointFailure(LEADER)
assert.equal(l.kind, 'RuntimeError')
assert.match(l.headline, /robot id that has no calibration file/)
assert.match(l.remedy, /calibration\/robots\/<type>\/<robot_id>\.json/, 'the remedy names the path lerobot looks in')
assert.match(l.remedy, /name mismatch, not a hardware fault/, 'nobody should be sent to the hardware for a path bug')

// AN UNKNOWN EXCEPTION IS QUOTED, NEVER GUESSED AT: a wrong remedy for a hardware fault is worse than none.
const odd = jointFailure([`12:00:00 [mesh] x: state probe 'hw_joints' failed, omitted: TimeoutError("no reply from motor 4")`])
assert.equal(odd.kind, 'TimeoutError')
assert.equal(odd.remedy, undefined, 'no remedy is offered for a failure the dashboard does not know')
assert.match(odd.headline, /no advice for/)
assert.match(odd.quote, /no reply from motor 4/, 'the arm\'s own words survive')
assert.equal(odd.tailMisleads, undefined, 'no reassuring tail, no claim of one')

// Respawned and failed differently: the LAST failure is the current one.
const twice = jointFailure([
  `1 [mesh] a: state probe 'hw_joints' failed, omitted: ConnectionError("Port is in use!")`,
  `2 [mesh] a: state probe 'hw_joints' failed, omitted: RuntimeError("has no calibration registered")`,
])
assert.equal(twice.kind, 'RuntimeError')

// Silence is not a diagnosis.
assert.equal(jointFailure([]), null)
assert.equal(jointFailure(null), null)
assert.equal(jointFailure(['13:58:52 hardware connected', '13:58:52 online']), null, 'a healthy log explains nothing')
assert.equal(jointFailure([`x state probe 'hw_joints' failed, omitted: `]), null, 'a failure with no exception text is not a sentence')
assert.equal(jointFailureLine(null), null)
console.log('jointFailure: all assertions passed')
