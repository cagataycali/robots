// R5: assertions for the calibration command builder (lib/calibrateCommand.ts).
// Run: npx esbuild src/lib/calibrateCommand.ts --bundle --format=esm --outfile=/tmp/calibrateCommand.mjs && node src/lib/calibrateCommand.test.mjs
import assert from 'node:assert/strict'

const { calibratePlan, deviceModel, deviceId } = await import('/tmp/calibrateCommand.mjs')

const FOLLOWER = { device: '/dev/cu.usbmodem5AB01584281', serial_number: '5AB0158428', role: 'follower', role_volts: 12.6 }
const LEADER = { device: '/dev/cu.usbmodem5AB01818061', serial_number: '5AB0181806', role: 'leader', role_volts: 7.4 }

// --- the happy path names every flag the operator would have had to guess ---
{
  const p = calibratePlan(FOLLOWER, 'so101')
  assert.ok(p.command, 'a measured follower must produce a command')
  assert.match(p.command, /^lerobot-calibrate /)
  assert.match(p.command, /--device_type=robots\b/)
  assert.match(p.command, /--device_model=so101_follower\b/)
  assert.match(p.command, /--port=\/dev\/cu\.usbmodem5AB01584281\b/)
  assert.match(p.command, /--device_id=follower_5AB0158428\b/)
  assert.match(p.reason, /12\.6V/, 'the reason must cite the measurement it trusted')
}

// --- THE POINT OF THE FILE: a leader is a teleoperator, not a robot ---
{
  const p = calibratePlan(LEADER, 'so101')
  assert.match(p.command, /--device_type=teleoperators\b/,
    'a leader calibrates as a teleoperator - the wrong type writes into the wrong directory tree')
  assert.match(p.command, /--device_model=so101_leader\b/)
  assert.equal(p.deviceType, 'teleoperators')
}

// --- refuse to guess: an unmeasured bus gets NO command, and a next step ---
{
  const p = calibratePlan({ device: '/dev/cu.usbmodem1', serial_number: 'X1' }, 'so101')
  assert.equal(p.command, null, 'no role measured = no command; a wrong role writes the wrong file')
  assert.equal(p.needsMeasurement, true)
  assert.match(p.reason, /measure the role first/)
}

// an arm whose supply is off cannot be calibrated (calibration must MOVE it)
{
  const p = calibratePlan({ device: '/dev/cu.usbmodem1', role: 'unpowered', role_volts: 5.5 }, 'so101')
  assert.equal(p.command, null)
  assert.equal(p.needsMeasurement, true)
  assert.match(p.reason, /USB logic rail/, 'must explain that 5.5V is the logic rail, not a dead arm')
}

// a faulty bus is a fault, never a role
{
  const p = calibratePlan({ device: '/dev/cu.usbmodem1', role: 'mixed' }, 'so101')
  assert.equal(p.command, null)
  assert.match(p.reason, /fault rather than a role/)
  assert.notEqual(p.needsMeasurement, true, 're-measuring will not fix wiring; do not offer it as the fix')
}

// unknown verdict is not a role either
assert.equal(calibratePlan({ device: '/dev/x', role: 'unknown' }, 'so101').command, null)

// --- the family is never assumed to be so101 ---
{
  const p = calibratePlan({ ...FOLLOWER, role: 'follower' }, 'so100')
  assert.match(p.command, /--device_model=so100_follower\b/, "an so100 owner must not be handed an so101 command")
}
{
  const p = calibratePlan(FOLLOWER, null)
  assert.equal(p.command, null, 'no family = no command, because the model name is half unknown')
  assert.notEqual(p.needsMeasurement, true, 'the bus measurement is not what is missing here')
}

// a registry entry that already carries the role is not double-suffixed
assert.equal(deviceModel('so101_follower', 'follower'), 'so101_follower')
assert.equal(deviceModel('SO101', 'leader'), 'so101_leader', 'family is normalised, not trusted verbatim')

// --- device_id is serial-scoped: two identical arms must not share one file ---
{
  const a = deviceId({ device: '/dev/a', serial_number: 'AAA' }, 'follower')
  const b = deviceId({ device: '/dev/b', serial_number: 'BBB' }, 'follower')
  assert.notEqual(a, b, 'two so101 followers would otherwise overwrite each other\'s calibration')
  assert.equal(deviceId({ device: '/dev/a' }, 'leader'), 'leader', 'no serial degrades to the bare role')
}

// --- a path with a space is quoted, an ordinary one is left readable ---
{
  const p = calibratePlan({ ...FOLLOWER, device: '/dev/odd name' }, 'so101')
  assert.match(p.command, /--port='\/dev\/odd name'/)
  assert.ok(!calibratePlan(FOLLOWER, 'so101').command.includes("'"), 'a normal port must not be needlessly quoted')
}

// --- a refusal ALWAYS explains itself: no silent empty state anywhere ---
for (const facts of [
  { device: '/dev/x' },
  { device: '/dev/x', role: 'unpowered', role_volts: 5.5 },
  { device: '/dev/x', role: 'mixed' },
  { device: '/dev/x', role: 'unknown' },
  { ...FOLLOWER },
]) {
  const p = calibratePlan(facts, facts.role === 'follower' ? null : 'so101')
  assert.ok(p.reason && p.reason.length > 20, 'every plan carries a human reason, command or not')
}

// --- the dashboard never commands motion: no verb that moves an arm ---
{
  const p = calibratePlan(FOLLOWER, 'so101')
  for (const forbidden of ['torque', 'move', 'write', 'goto', 'home']) {
    assert.ok(!p.command.includes(forbidden), `the command must not contain "${forbidden}"`)
  }
  assert.match(p.reason, /Run this in a terminal/, 'the human is the executor, and the copy says so')
}

console.log('calibrateCommand: all assertions passed')
