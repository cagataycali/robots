import assert from 'node:assert/strict'
import { camerasField } from '/tmp/cameraRows.mjs'

// ── nothing filled in = spawn without cameras, never an error ──
assert.deepEqual(camerasField([]), { value: null, problem: null, note: null })
assert.deepEqual(camerasField([{ name: 'main', index: '' }]), { value: null, problem: null, note: null },
  'an unfilled row is not an error — it is the form at rest')

// ── one camera: the mapping shape the server refused a bare int for ──
{
  const f = camerasField([{ name: 'main', index: '3' }])
  assert.equal(f.problem, null)
  assert.deepEqual(f.value, { main: { index_or_path: 3 } },
    'a MAPPING per entry — "got int: 3" was a live ValueError')
}

// ── the whole point: main + wrist in one spawn, shared fps/size applied to each ──
{
  const f = camerasField(
    [{ name: 'main', index: '0' }, { name: 'wrist', index: '1' }],
    { fps: 30, width: 640, height: 480 },
  )
  assert.equal(f.problem, null)
  assert.deepEqual(f.value, {
    main: { index_or_path: 0, fps: 30, width: 640, height: 480 },
    wrist: { index_or_path: 1, fps: 30, width: 640, height: 480 },
  })
}

// ── an unfilled row BETWEEN filled ones is skipped, not judged ──
{
  const f = camerasField([
    { name: 'main', index: '0' },
    { name: '', index: '' }, // the "add camera" row nobody used
    { name: 'top', index: '2' },
  ])
  assert.equal(f.problem, null)
  assert.deepEqual(Object.keys(f.value), ['main', 'top'])
}

// ── refusals, each in words the operator can act on ──
assert.match(camerasField([{ name: '', index: '1' }]).problem, /needs a name/,
  'an index without a name cannot become a config key')
assert.match(camerasField([{ name: 'wrist cam', index: '1' }]).problem, /letters, digits and _/,
  'names become lerobot config keys — identifier-shaped only')
assert.match(camerasField([{ name: '2nd', index: '1' }]).problem, /starting with a letter/)
assert.match(
  camerasField([{ name: 'main', index: '0' }, { name: 'MAIN', index: '1' }]).problem,
  /two cameras named/,
  'case-insensitive: "MAIN" and "main" would collide in any sane config reader')
{
  // a shared index WARNS but does not block — the operator may know something we don't
  const f = camerasField([{ name: 'main', index: '1' }, { name: 'wrist', index: '1' }])
  assert.equal(f.problem, null, 'sendable: a warning is not a wall')
  assert.match(f.note, /index 1 is claimed by both/, 'but it is said out loud')
  assert.deepEqual(Object.keys(f.value), ['main', 'wrist'], 'both entries survive')
}
assert.match(camerasField([{ name: 'device_name', index: '1' }]).problem, /bookkeeping key/,
  "the dashboard's annotation key must never become a camera name — validate_cameras won't catch it")
assert.match(camerasField([{ name: 'main', index: 'usb0' }]).problem, /not a camera index/)
assert.match(camerasField([{ name: 'main', index: '-1' }]).problem, /not a camera index/)
assert.match(camerasField([{ name: 'main', index: '1.5' }]).problem, /not a camera index/,
  'indices are positions in a list — 1.5 is a typo, not a camera')

// ── zero is a real index (the first camera), not falsy-rejected ──
assert.deepEqual(camerasField([{ name: 'main', index: '0' }]).value, { main: { index_or_path: 0 } })

console.log('cameraRows: all assertions passed')
