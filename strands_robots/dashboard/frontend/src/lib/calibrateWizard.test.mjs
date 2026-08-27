// View rules for the calibration wizard (lib/calibrateWizard.ts).
// Run: npx esbuild src/lib/calibrateWizard.ts --bundle --format=esm --outfile=/tmp/calibrateWizard.mjs && node src/lib/calibrateWizard.test.mjs
import assert from 'node:assert/strict'

const { wizardView, confirmSheet } = await import('/tmp/calibrateWizard.mjs')

const base = { id: 'x', alive: true }

// --- the confirm sheet names the physical consequence before anything runs ---
{
  const c = confirmSheet({ port: '/dev/cu.usbmodem1', deviceId: 'leader_arm', model: 'so101_follower' })
  assert.match(c.body, /torque switches OFF/i)
  assert.match(c.body, /LIMP/i, 'the operator must hear the arm will go limp BEFORE starting')
  assert.match(c.body, /\/dev\/cu\.usbmodem1/)
  assert.match(c.body, /"leader_arm"/, 'the id is the file name the calibration lands under')
  assert.match(c.body, /your hand does all the moving/i, 'nothing here commands motion — say so')
}

// --- every step has a way out while the run is live ---
for (const step of ['starting', 'reuse', 'middle', 'recording']) {
  const v = wizardView({ ...base, step, motors: [] })
  assert.ok(v.buttons.some(b => b.key === 'cancel'), `${step} must offer cancel`)
  assert.equal(v.finished, false)
}

// --- reuse: recalibrating is the primary action (the wizard exists because the old file is doubted) ---
{
  const v = wizardView({ ...base, step: 'reuse' })
  const primary = v.buttons.find(b => b.primary)
  assert.equal(primary.key, 'c')
  assert.ok(v.buttons.some(b => b.key === 'enter'), 'keeping the existing file stays one click away')
}

// --- middle: the limp-arm fact is repeated where the hands are ---
{
  const v = wizardView({ ...base, step: 'middle' })
  assert.match(v.body, /limp/i)
  assert.match(v.body, /MIDDLE/, 'the instruction is the middle of travel')
}

// --- recording: a joint that never moved is named BEFORE lerobot refuses it ---
{
  const motors = [
    { name: 'shoulder_pan', min: 1500, pos: 2100, max: 2600 },
    { name: 'elbow_flex', min: 2000, pos: 2000, max: 2000 }, // untouched
    { name: 'wrist_roll', min: 0, pos: 0, max: 0 },          // full-turn: exempt
  ]
  const v = wizardView({ ...base, step: 'recording', motors })
  assert.deepEqual(v.unmoved, ['elbow_flex'], 'min===max means unmoved; wrist_roll is exempt (full turn)')
  assert.equal(v.motors.length, 3, 'the live table renders every row')
  assert.match(v.body, /refuses to save/, 'the one-point-range refusal is pre-empted in prose')
}

// --- saved: a receipt with the path, and polling stops ---
{
  const v = wizardView({ ...base, step: 'saved', alive: false, path: '/x/robots/so101_follower/leader_arm.json' })
  assert.equal(v.tone, 'ok')
  assert.equal(v.finished, true)
  assert.match(v.body, /leader_arm\.json/)
  assert.ok(!v.buttons.some(b => b.key === 'cancel'), 'nothing to cancel after the save')
}

// --- failed: the named reason is the body, raw output only behind details ---
{
  const v = wizardView({
    ...base, step: 'failed', alive: false, returncode: 1,
    reason: "ConnectionError: Could not connect on port '/dev/cu.usbmodem1'",
    tail: ['Traceback...', "ConnectionError: Could not connect on port '/dev/cu.usbmodem1'"],
  })
  assert.equal(v.tone, 'bad')
  assert.equal(v.finished, true)
  assert.match(v.body, /Could not connect/)
  assert.ok(v.detail, 'the tail is offered as detail')
}

// failed with nothing to say still says something
{
  const v = wizardView({ ...base, step: 'failed', alive: false, returncode: 127 })
  assert.match(v.body, /127/)
}

console.log('calibrateWizard: all assertions passed')
