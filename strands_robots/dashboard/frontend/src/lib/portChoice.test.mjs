import assert from 'node:assert/strict'
const { portChoice, blocksSpawn } = await import('/tmp/portChoice.mjs')

const A = '/dev/tty.usbmodem5AB0158428'
const B = '/dev/tty.usbmodem5AB0181806'

// The ordinary case stays quiet.
{
  const c = portChoice({ chosen: A, known: [A, B], claimed: [] })
  assert.deepEqual(c, { kind: 'ok', port: A })
  assert.equal(blocksSpawn(c), false)
}

// Nothing picked yet is not an error - the button already requires a port.
assert.deepEqual(portChoice({ chosen: '', known: [A], claimed: [] }), { kind: 'empty' })
assert.equal(blocksSpawn({ kind: 'empty' }), false)

// VANISHED: replug renamed the tty. The <select> renders blank while state still holds the old path,
// so the form looks unfilled and the spawn would open a device file that does not exist.
{
  const c = portChoice({ chosen: A, known: [B], claimed: [] })
  assert.equal(c.kind, 'vanished')
  assert.equal(blocksSpawn(c), true)
  assert.match(c.detail, /different \/dev path/)
  assert.match(c.detail, /picker shows blank/)
  assert.match(c.remedy, /rescan/)
}

// CLAIMED: something took the bus after it was picked. Two owners on one Feetech bus is the
// "Port is in use!" collision, which presents as an arm that starts and dies.
{
  const c = portChoice({ chosen: A, known: [A, B], claimed: [A] })
  assert.equal(c.kind, 'claimed')
  assert.equal(blocksSpawn(c), true)
  assert.match(c.detail, /Port is in use/)
  assert.match(c.remedy, /despawn/)
}

// Claimed OUTRANKS vanished: a bus held by a running robot is a fact about the fleet, while a path
// missing from a scan can be a scan that raced the device.
{
  const c = portChoice({ chosen: A, known: [B], claimed: [A] })
  assert.equal(c.kind, 'claimed')
}

// No scan has landed yet: absence is not evidence. This must NOT block - a false "vanished" on every
// panel open would train the operator to ignore the warning that matters.
for (const input of [
  { chosen: A, known: [], claimed: [] },
  { chosen: A, known: [], claimed: [], scanned: false },
  { chosen: A, known: [B], claimed: [], scanned: false },
]) {
  const c = portChoice(input)
  assert.equal(c.kind, 'unknown', JSON.stringify(input))
  assert.equal(blocksSpawn(c), false)
}

// A claimed port is still reported before any scan: that knowledge comes from the live children, not
// from the scan.
assert.equal(portChoice({ chosen: A, known: [], claimed: [A], scanned: false }).kind, 'claimed')

// Whitespace around a path is not a different device.
assert.equal(portChoice({ chosen: `  ${A}  `, known: [A], claimed: [] }).kind, 'ok')
// Paths are compared exactly - a prefix is a different device (usbmodem1 vs usbmodem11).
assert.equal(portChoice({ chosen: '/dev/tty.usbmodem1', known: ['/dev/tty.usbmodem11'], claimed: [] }).kind, 'vanished')

// Missing arrays must not throw: this runs on a doc that may still be null.
assert.equal(portChoice({ chosen: A }).kind, 'unknown')

console.log('portChoice: all assertions passed')
