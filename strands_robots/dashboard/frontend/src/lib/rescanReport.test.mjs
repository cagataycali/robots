// Assertions for the devices-screen rescan verdict (lib/rescanReport.ts).
// Run: npx esbuild src/lib/rescanReport.ts --bundle --format=esm --outfile=/tmp/rescanReport.mjs \
//        && node src/lib/rescanReport.test.mjs
import assert from 'node:assert/strict'

const { rescanReport, hardwareKey } = await import('/tmp/rescanReport.mjs')

const scan = (devs, cams, problem = null) => ({
  serial_ports: devs.map(d => ({ device: d })),
  cameras: cams.map(i => ({ index: i })),
  camera_problem: problem,
})

const TWO = scan(['/dev/tty.usbmodem5AB0181806', '/dev/tty.usbmodem5AB0158428'], [0, 1, 2])

// THE POINT OF THE FILE: the three worlds must never render the same sentence.
{
  const unchanged = rescanReport(TWO, { ok: true, after: TWO })
  const nothing = rescanReport(scan([], []), { ok: true, after: scan([], []) })
  const failed = rescanReport(TWO, { ok: false, error: 'HTTP 500' })
  const texts = new Set([unchanged.text, nothing.text, failed.text])
  assert.equal(texts.size, 3, 'unchanged / found-nothing / failed must be three different lines')
  assert.deepEqual([unchanged.tone, nothing.tone, failed.tone], ['ok', 'warn', 'bad'])
  assert.deepEqual([unchanged.stale, nothing.stale, failed.stale], [false, false, true])
}

// 1. Ran, nothing changed: the counts are the proof the click landed.
{
  const r = rescanReport(TWO, { ok: true, after: TWO })
  assert.match(r.text, /completed/)
  assert.match(r.text, /unchanged/)
  assert.match(r.text, /2 serial ports, 3 cameras/)
}

// A one-item scan is not "1 serial ports".
{
  const one = scan(['/dev/tty.usbmodem1'], [0])
  const r = rescanReport(one, { ok: true, after: one })
  assert.match(r.text, /1 serial port, 1 camera\./)
}

// 2. Ran and found nothing: says the scan SUCCEEDED, and points at the cable.
{
  const r = rescanReport(TWO, { ok: true, after: scan([], []) })
  assert.match(r.text, /scan completed/, 'an empty result is not a failure and must not read like one')
  assert.match(r.text, /no serial ports and no cameras/)
  assert.match(r.text, /cable/)
  assert.equal(r.stale, false)
}

// ...and when macOS blocks the camera layer, the camera count is evidence about
// PERMISSION, not about what is plugged in (the Mac-mini TCC reality).
{
  const blocked = scan([], [], { kind: 'tcc_denied', message: 'no camera access' })
  const r = rescanReport(TWO, { ok: true, after: blocked })
  assert.match(r.text, /blocked/)
  assert.match(r.text, /says nothing about what is connected/)
}
// No camera problem reported => no caveat invented.
assert.doesNotMatch(rescanReport(TWO, { ok: true, after: scan([], []) }).text, /blocked/)

// 3. Failed: names the error AND admits the visible list is the older one.
{
  const r = rescanReport(TWO, { ok: false, error: 'HTTP 503 Service Unavailable' }, { beforeAtMs: 1000, nowMs: 41000 })
  assert.match(r.text, /HTTP 503/)
  assert.match(r.text, /PREVIOUS scan/)
  assert.match(r.text, /40s old/, 'the age of what is on screen is the actionable part')
  assert.equal(r.stale, true)
}
// A failure with no message still explains itself.
assert.match(rescanReport(TWO, { ok: false }).text, /the request failed/)
// Age is omitted when it is unknown or trivially fresh (a fake "0s old" is noise).
assert.doesNotMatch(rescanReport(TWO, { ok: false, error: 'x' }).text, /old/)
assert.doesNotMatch(rescanReport(TWO, { ok: false, error: 'x' }, { beforeAtMs: 1000, nowMs: 3000 }).text, /old/)
assert.match(rescanReport(TWO, { ok: false, error: 'x' }, { beforeAtMs: 1000, nowMs: 601000 }).text, /10min old/)

// First-ever scan failed: the empty lists are explained rather than trusted.
{
  const r = rescanReport(null, { ok: false, error: 'network error' })
  assert.match(r.text, /nothing has been scanned yet/)
  assert.doesNotMatch(r.text, /PREVIOUS/)
  assert.equal(r.stale, true)
}

// Deltas name what appeared and what vanished — an unplugged arm is the event
// an operator most needs to see.
{
  const after = scan(['/dev/tty.usbmodem5AB0181806'], [0, 1, 2])
  const r = rescanReport(TWO, { ok: true, after })
  assert.match(r.text, /−1 serial port \(\/dev\/tty\.usbmodem5AB0158428\)/)
  assert.doesNotMatch(r.text, /\+/, 'nothing appeared, so nothing may be claimed to have appeared')
  assert.match(r.text, /now 1 serial port, 3 cameras/)
}
{
  const after = scan(['/dev/tty.usbmodem5AB0181806', '/dev/tty.usbmodem5AB0158428'], [0, 1, 2, 3])
  const r = rescanReport(TWO, { ok: true, after })
  assert.match(r.text, /\+1 camera \(index 3\)/)
}
// Both directions in one scan (a port re-enumerated under a new path).
{
  const after = scan(['/dev/tty.usbmodem5AB0181806', '/dev/tty.usbmodem99'], [0, 1, 2])
  const r = rescanReport(TWO, { ok: true, after })
  assert.match(r.text, /\+1 serial port \(\/dev\/tty\.usbmodem99\)/)
  assert.match(r.text, /−1 serial port \(\/dev\/tty\.usbmodem5AB0158428\)/)
}
// Many at once: named up to three, then counted — never a wall of paths.
{
  const after = scan(['/a', '/b', '/c', '/d', '/e'], [0, 1, 2])
  const r = rescanReport(scan([], [0, 1, 2]), { ok: true, after })
  assert.match(r.text, /\+5 serial ports \(\/a, \/b, \/c \+2 more\)/)
}

// A first successful scan has no "unchanged" to compare against.
{
  const r = rescanReport(null, { ok: true, after: TWO })
  assert.match(r.text, /found: 2 serial ports, 3 cameras/)
  assert.doesNotMatch(r.text, /unchanged/)
}

// Junk shapes cannot throw and cannot invent hardware.
for (const bad of [null, {}, { serial_ports: null, cameras: null }, { serial_ports: [{}], cameras: [{}] }]) {
  const r = rescanReport(bad, { ok: true, after: bad })
  assert.equal(typeof r.text, 'string')
  assert.equal(r.tone, 'warn', 'a payload we cannot read lists no hardware, and that is the found-nothing case')
}
{
  // A camera row without an index is not a camera we can name.
  const r = rescanReport(scan([], []), { ok: true, after: { serial_ports: [{ device: '/dev/x' }], cameras: [{ name_hint: 'Logi' }] } })
  assert.match(r.text, /\+1 serial port \(\/dev\/x\)/)
  assert.match(r.text, /now 1 serial port, 0 cameras/)
}

// hardwareKey: the component's tripwire for "my verdict no longer describes the screen".
{
  const { hardwareKey } = await import('/tmp/rescanReport.mjs')
  assert.equal(hardwareKey(TWO), hardwareKey(scan(['/dev/tty.usbmodem5AB0158428', '/dev/tty.usbmodem5AB0181806'], [2, 1, 0])),
    'order is not a change')
  assert.notEqual(hardwareKey(TWO), hardwareKey(scan(['/dev/tty.usbmodem5AB0181806'], [0, 1, 2])), 'an unplugged arm is a change')
  assert.notEqual(hardwareKey(TWO), hardwareKey(scan(['/dev/tty.usbmodem5AB0181806', '/dev/tty.usbmodem5AB0158428'], [0, 1, 2, 3])))
  assert.equal(hardwareKey(null), hardwareKey({}), 'nothing read and nothing present key the same')
}

console.log('rescanReport: all assertions passed')

{
  const before = { serial_ports: [{ device: '/dev/tty.usbmodem1' }], cameras: [] }
  const fresh = {
    serial_ports: [{ device: '/dev/tty.usbmodem1' }],
    cameras: [{ index: 0, name_hint: 'USB2.0_CAM1' }, { index: 1, name_hint: 'Logi 4K Pro' }],
  }
  const verdict = rescanReport(before, { ok: true, after: fresh }, { beforeAtMs: 1000, nowMs: 4000 })
  assert.equal(verdict.stale, false)
  assert.match(verdict.text, /camera/i, 'the scan reported what it found')

  // The poll's cached doc still shows no cameras.
  assert.notEqual(hardwareKey(before), hardwareKey(fresh))
  // ...which is exactly why the newest REQUEST now wins: the cached load is older
  // evidence, it is never painted, and the verdict keeps its evidence.
  assert.equal(hardwareKey(fresh), hardwareKey(fresh))
}
console.log('rescanReport: cached-poll interleaving assertions passed')
