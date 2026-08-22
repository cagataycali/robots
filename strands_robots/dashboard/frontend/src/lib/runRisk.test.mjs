// Assertions for lib/runRisk.ts — is pressing ▶ about to move metal?
import assert from 'node:assert/strict'

const { runRisk } = await import('/tmp/runRisk.mjs')

// A real device names itself, and the reason must QUOTE it — that sentence is the operator's evidence.
{
  const r = runRisk({ hw: 'so_follower', robot_type: 'so101' })
  assert.equal(r.physical, true)
  assert.equal(r.device, 'so_follower')
  assert.match(r.reason, /so_follower/)
}

// A declared sim must NOT raise the physical-motion sheet: crying wolf here costs the sheet its
// authority everywhere else.
{
  const r = runRisk({ robot_type: 'sim', hw: 'mujoco_so101' })
  assert.equal(r.physical, false)
  assert.match(r.reason, /simulat/i)
}
{
  const r = runRisk({ hw: 'mujoco_so101' })
  assert.equal(r.physical, false, 'a mujoco backend is pixels')
}

// Silence is treated as danger — the deliberate asymmetry.
for (const p of [undefined, null, {}]) {
  const r = runRisk(p)
  assert.equal(r.physical, true, `unknown presence must err toward physical: ${JSON.stringify(p)}`)
}
// …and it must SAY it is guessing, so nobody reads it as evidence of hardware.
assert.match(runRisk({}).reason, /did not say/)

// Hardware present but disconnected: still physical. It can reconnect between the judgment and the
// click, and the cost of being wrong is a collision.
{
  const r = runRisk({ connected: false })
  assert.equal(r.physical, true)
  assert.match(r.reason, /not connected/)
}

console.log('runRisk: all assertions passed')
