import assert from 'node:assert/strict'

const { rowsFromConfig, configFromRows, parseIndexOrPath, applySummary } = await import('/tmp/cameraConfig.mjs')

// round trip: config -> rows -> config
{
  const cfg = { top: { index_or_path: 0, fps: 30, width: 1280, height: 720 }, wrist: { index_or_path: '/dev/video1' } }
  const back = configFromRows(rowsFromConfig(cfg))
  assert.deepEqual(back.cameras, cfg, 'a config must survive the editor untouched')
  assert.equal(back.error, undefined)
}

// index vs path
assert.equal(parseIndexOrPath('0'), 0)
assert.equal(parseIndexOrPath(' 2 '), 2)
assert.equal(parseIndexOrPath('/dev/video1'), '/dev/video1')
assert.equal(parseIndexOrPath(''), null)

// empty editor = detach all, a legal deliberate config
{
  const r = configFromRows([])
  assert.equal(r.cameras, null)
  assert.equal(r.error, undefined)
}

// refusals name the camera and the field
const row = (over) => ({ name: 'top', indexOrPath: '0', fps: '', width: '', height: '', ...over })
assert.match(configFromRows([row({ name: '' })]).error, /name/, 'nameless row refused')
assert.match(configFromRows([row({ indexOrPath: '' })]).error, /top.*index/, 'missing index refused')
assert.match(configFromRows([row({ fps: '0' })]).error, /fps=0.*1\.\.240/, 'fps bound mirrors the server')
assert.match(configFromRows([row({ fps: 'thirty' })]).error, /whole number/)
assert.match(configFromRows([row(), row()]).error, /both named/, 'duplicate names refused')

// half a rectangle is a typo
assert.match(configFromRows([row({ width: '1280' })]).error, /both width and height/)
assert.equal(configFromRows([row({ width: '1280', height: '720' })]).error, undefined)

// blank optional fields mean driver defaults — they must NOT appear as keys
{
  const r = configFromRows([row()])
  assert.deepEqual(r.cameras, { top: { index_or_path: 0 } })
}

// the confirm sentence names the respawn cost, and detach-all says so
assert.match(applySummary([row()], 'so101-arm-1'), /restarting it/, 'the cost is named')
assert.match(applySummary([], 'so101-arm-1'), /detach every camera/, 'detach-all is explicit')

console.log('cameraConfig: all assertions passed')

// --- previewRateNote: the fps field promises a capture rate, not a preview rate
{
  const { previewRateNote, DEFAULT_MESH_CAMERA_HZ } = await import('/tmp/cameraConfig.mjs')
  // The case that produces a false bug report: 30 fps chosen, 5/s delivered.
  const note = previewRateNote(30, 5)
  assert.ok(note && note.includes('30 fps') && note.includes('5/s'), note)
  assert.ok(note.includes('camera_hz'), 'must say where to change it')
  // Silent when the two rates cannot disagree - a caveat that is not true here
  // would train the operator to ignore the line that matters.
  assert.equal(previewRateNote(5, 5), null)
  assert.equal(previewRateNote(5, 30), null)
  assert.equal(previewRateNote(null, 5), null, 'blank fps = driver default, nothing claimed yet')
  // No config answer yet: assume the rate every spawned child inherits rather
  // than staying silent, because silence is what the bug report grew from.
  assert.ok(previewRateNote(30, null)?.includes(`${DEFAULT_MESH_CAMERA_HZ}/s`))
  assert.ok(previewRateNote(30, 0)?.includes(`${DEFAULT_MESH_CAMERA_HZ}/s`), 'a nonsense hz is not a rate')
}

// --- focusTarget: opening the sheet from a camera tile lands on THAT camera
{
  const { focusTarget } = await import('/tmp/cameraConfig.mjs')
  const rows = [
    { name: 'top', indexOrPath: '0', fps: '', width: '', height: '' },
    { name: 'wrist', indexOrPath: '1', fps: '30', width: '', height: '' },
  ]
  // a named camera focuses its own row's fps — the setting the click was about
  assert.deepEqual(focusTarget(rows, 'wrist', false), { index: 1, field: 'fps' })
  assert.deepEqual(focusTarget(rows, 'top', false), { index: 0, field: 'fps' })
  // adding focuses the NEW (last) row's name
  assert.deepEqual(focusTarget([...rows, { name: '', indexOrPath: '', fps: '', width: '', height: '' }], null, true),
    { index: 2, field: 'name' })
  // adding with no rows at all claims nothing (nothing to focus)
  assert.deepEqual(focusTarget([], null, true), null)
  // a camera that no longer exists in the config focuses nothing rather than the wrong row
  assert.equal(focusTarget(rows, 'gone', false), null)
  // generic open (header button): no claim
  assert.equal(focusTarget(rows, null, false), null)
  console.log('focusTarget: all assertions passed')
}
