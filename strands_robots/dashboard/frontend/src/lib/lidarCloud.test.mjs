/**
 * The point-cloud rules a 3D tile must not re-answer for itself.
 *
 * Build + run:  node scripts/run-lib-tests.mjs lidarCloud
 */
import assert from 'node:assert/strict'
import {
  BYTES_PER_POINT, MAX_POINTS, CLOUD_HZ, MAX_BYTES_PER_SECOND,
  decodeCloud, cloudBudget, intensityColor, cloudMetaFromEvent, coverageNote,
  lidarThrottleNotice, hostIsLittleEndian,
} from '/tmp/lidarCloud.mjs'

/** Build a wire frame the way the publisher does: little-endian xyzi float32. */
const frame = (rows) => {
  const buf = new ArrayBuffer(rows.length * BYTES_PER_POINT)
  const dv = new DataView(buf)
  rows.forEach((r, i) => r.forEach((v, j) => dv.setFloat32(i * BYTES_PER_POINT + j * 4, v, true)))
  return buf
}

// ---- decode -------------------------------------------------------------------------------------

{
  const rows = [[1, 2, 3, 0.25], [-4, 5, -6, 0.75]]
  const c = decodeCloud(frame(rows))
  assert.equal(c.n, 2, 'two points in, two points out')
  assert.deepEqual([...c.xyz], [1, 2, 3, -4, 5, -6], 'xyz is interleaved in wire order')
  assert.deepEqual([...c.intensity], [0.25, 0.75], 'the fourth component is intensity')
}

{
  // A partial point is refused, not rounded down: rounding builds a point out of two rows.
  const buf = new ArrayBuffer(BYTES_PER_POINT + 4)
  assert.equal(decodeCloud(buf), null, 'a length that is not a whole number of points is refused')
  assert.equal(decodeCloud(new ArrayBuffer(0)), null, 'an empty frame is not a cloud')
}

{
  // A caller may hand over a SLICE of a bigger read buffer. `new Float32Array(buf, 1)` throws
  // "start offset should be a multiple of 4" (measured in node), so the decoder copies instead.
  const rows = [[7, 8, 9, 1]]
  const src = frame(rows)
  const padded = new Uint8Array(src.byteLength + 1)
  padded.set(new Uint8Array(src), 1)
  const view = new Uint8Array(padded.buffer, 1, src.byteLength)
  assert.equal(view.byteOffset % 4, 1, 'the fixture really is misaligned')
  const c = decodeCloud(view)
  assert.deepEqual([...c.xyz], [7, 8, 9], 'a misaligned view decodes rather than throwing')
}

{
  // The wire format is NAMED little-endian, so the decoder checks rather than assuming.
  assert.equal(hostIsLittleEndian(), true, 'this host is little-endian (every browser target is)')
}

{
  // Why the PUBLISHER drops non-finite rows: the idiomatic spread reduction a tile uses to frame
  // the camera returns NaN from a single one, so the whole cloud loses its bounding box.
  const withNaN = new Float32Array([1, 2, NaN, 4])
  assert.ok(Number.isNaN(Math.min(...withNaN)), 'one NaN poisons Math.min over the buffer')
  const clean = decodeCloud(frame([[1, 1, 1, 0], [2, 2, 2, 1]]))
  assert.ok(Number.isFinite(Math.min(...clean.xyz)), 'a published cloud reduces to a finite bound')
}

// ---- budget -------------------------------------------------------------------------------------

{
  assert.equal(MAX_BYTES_PER_SECOND, MAX_POINTS * BYTES_PER_POINT * CLOUD_HZ,
               'the cap is what the publisher limits multiply out to, not a separate number')
  const ok = cloudBudget(MAX_POINTS, CLOUD_HZ)
  assert.equal(ok.bytesPerSecond, 320000, '4000 points at 5 Hz is 320 kB/s')
  assert.equal(ok.withinCap, true, 'the publisher default sits exactly ON the cap')
  assert.equal(ok.reason, '', 'nothing to report when it fits')
}

{
  const over = cloudBudget(MAX_POINTS * 2, CLOUD_HZ)
  assert.equal(over.withinCap, false, 'twice the points is over the cap')
  assert.match(over.reason, /kB\/s exceeds/, 'and it says so in bytes, not in a boolean')
}

{
  // A tile is free to ask for LESS. Halving the rate halves the cost, exactly.
  assert.equal(cloudBudget(MAX_POINTS, CLOUD_HZ / 2).bytesPerSecond, MAX_BYTES_PER_SECOND / 2)
  for (const bad of [0, -1, NaN, Infinity]) {
    assert.equal(cloudBudget(bad, CLOUD_HZ).bytesPerSecond, 0, `${bad} points costs nothing`)
    assert.equal(cloudBudget(MAX_POINTS, bad).bytesPerSecond, 0, `${bad} Hz costs nothing`)
  }
}

// ---- intensity ramp ---------------------------------------------------------------------------

{
  const lo = intensityColor(0)
  const hi = intensityColor(1)
  assert.notDeepEqual(lo, hi, 'the ramp actually ramps')
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    const c = intensityColor(t)
    assert.equal(c.length, 3, 'an rgb triple')
    c.forEach(v => assert.ok(v >= 0 && v <= 1, `${v} is a 0..1 channel at t=${t}`))
  }
  // Monotone in luminance, so brighter return reads as brighter point.
  const lum = t => { const [r, g, b] = intensityColor(t); return 0.2126 * r + 0.7152 * g + 0.0722 * b }
  for (let t = 0; t < 1; t += 0.1) assert.ok(lum(t) < lum(t + 0.1) + 1e-9, `luminance rises across ${t}`)
}

{
  // Out of range clamps rather than wrapping to the far end of the ramp.
  assert.deepEqual(intensityColor(-5), intensityColor(0), 'below the floor clamps to the floor')
  assert.deepEqual(intensityColor(99), intensityColor(1), 'above the ceiling clamps to the ceiling')
  // A sensor's own range: 0..255 reflectivity must span the ramp, not sit pinned at the top.
  assert.deepEqual(intensityColor(128, 0, 255), intensityColor(128 / 255), 'lo/hi stretches the ramp')
  assert.deepEqual(intensityColor(NaN), intensityColor(0), 'an unmeasured return colours as the floor')
  assert.deepEqual(intensityColor(5, 3, 3), intensityColor(0), 'a zero-width range is not a divide')
}

// ---- the /ws/mesh notification -----------------------------------------------------------------

{
  const meta = cloudMetaFromEvent({
    type: 'lidar_cloud', peer_id: 'g1',
    data: { t: 12.5, n: 4000, raw_count: 24000, stride: 6, bytes: 64000, encoding: 'xyzi_f32le' },
  })
  assert.equal(meta.peerId, 'g1')
  assert.equal(meta.n, 4000)
  assert.equal(meta.rawCount, 24000)
  assert.equal(meta.stride, 6)
  assert.equal(meta.bytes, 64000)
}

{
  assert.equal(cloudMetaFromEvent({ type: 'lidar', kind: 'summary', peer_id: 'g1', data: {} }), null,
               'the scalar summary frame is not a cloud notification')
  assert.equal(cloudMetaFromEvent({ type: 'lidar_cloud' }), null, 'a frame naming no peer is unusable')
  assert.equal(cloudMetaFromEvent(null), null)
  const bare = cloudMetaFromEvent({ type: 'lidar_cloud', peer_id: 'g1', data: { n: 10 } })
  assert.equal(bare.rawCount, null, 'a publisher that did not say stays null, not zero')
}

{
  // The point of carrying raw_count: an operator must not read a sixth of the returns as all of them.
  const m = n => cloudMetaFromEvent({ type: 'lidar_cloud', peer_id: 'g1', data: n })
  assert.equal(coverageNote(m({ n: 4000, raw_count: 24000, stride: 6 })), '4000 of 24000 points (every 6th)')
  assert.equal(coverageNote(m({ n: 900, raw_count: 900 })), '900 points', 'a full sweep says so plainly')
  assert.equal(coverageNote(m({ n: 900 })), '900 points', 'no raw_count claims no downsample')
  assert.equal(coverageNote(m({ n: 0 })), 'no points')
  assert.equal(coverageNote(m({ n: 4000, raw_count: 24000 })), '4000 of 24000 points (every 6th)',
               'the stride is derived when the publisher did not send it')
}

// ---- throttle notice ---------------------------------------------------------------------------

{
  assert.match(lidarThrottleNotice({ type: 'lidar_error', throttled: true, error: 'pacing it at 1 fps' }),
               /1 fps/, 'the server sentence reaches the tile')
  assert.equal(lidarThrottleNotice({ type: 'lidar_error', error: 'sensor offline' }), null,
               'a REAL error is not a throttle notice')
  assert.equal(lidarThrottleNotice({ type: 'camera_error', throttled: true, error: 'x' }), null,
               'the camera tile has its own reader')
  assert.ok(lidarThrottleNotice({ type: 'lidar_error', throttled: true }),
            'a throttle with no sentence still says something')
}

console.log('lidarCloud: all rules hold')
