// U19: assertions for the camera editor's pure half (lib/cameraConfig.ts).
// Run: npx esbuild src/lib/cameraConfig.ts --bundle --format=esm --outfile=/tmp/cameraConfig.mjs && node src/lib/cameraConfig.test.mjs
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
