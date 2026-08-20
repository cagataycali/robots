import assert from 'node:assert/strict'
import { pacedFps, pacingFromNotice, nextRequestedFps } from './cameraPacing.ts'

// --- the discriminator is the flag, not the wording -------------------------------
assert.equal(pacingFromNotice({ type: 'camera_error', error: 'OpenCVCamera(1) read failed' }), null,
  'a real camera error must stay an error')
assert.equal(pacingFromNotice({ type: 'camera_error', throttled: false, error: 'x' }), null)
assert.equal(pacingFromNotice(null), null)
assert.equal(pacingFromNotice('camera_error'), null)
assert.equal(pacingFromNotice({ type: 'frame', throttled: true }), null,
  'only camera_error carries this')

// --- a throttle notice becomes a calm note + a rate -------------------------------
const p = pacingFromNotice({
  type: 'camera_error', throttled: true,
  error: 'this viewer opened this camera 92 times in the last minute, so the server is pacing it at 2 fps until that settles',
})
assert.ok(p, 'a throttle notice must be recognised')
assert.equal(p.fps, 2)
assert.ok(/paced by the server at 2 fps/.test(p.note), p.note)
assert.ok(!/error/i.test(p.note), 'a paced tile is not an error: ' + p.note)

// A server that stops naming a number still gets an honest note, never a crash.
const vague = pacingFromNotice({ type: 'camera_error', throttled: true, error: 'slow down' })
assert.equal(vague.fps, null)
assert.ok(/pacing/.test(vague.note))

// --- parsing the rate out of a sentence -------------------------------------------
assert.equal(pacedFps('pacing it at 0.5 fps until'), 0.5)
assert.equal(pacedFps('at 2fps'), 2)
assert.equal(pacedFps('no rate here'), null)
assert.equal(pacedFps('at 0 fps'), null, 'zero would freeze the tile forever')
assert.equal(pacedFps('at -3 fps'), null)

// --- the lower rate wins ----------------------------------------------------------
assert.equal(nextRequestedFps(1, 2), 1, "our own degraded rate is not talked back up")
assert.equal(nextRequestedFps(null, 2), 2)
assert.equal(nextRequestedFps(1, null), 1)
assert.equal(nextRequestedFps(null, null), null)

console.log('cameraPacing: all assertions passed')
