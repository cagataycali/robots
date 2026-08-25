/**
 * A LiDAR point cloud, from the bytes on /ws/lidar to something a tile can draw.
 *
 * The pure half of the 3D tile: decoding, budgeting and colouring, with no renderer and no React,
 * so every rule here is testable in node. The tile that mounts a GPU renderer over this is a
 * separate concern; what it must not do is invent its own answer to any of these.
 */

/** Bytes per point on the wire: x, y, z, intensity, each a float32. */
export const BYTES_PER_POINT = 16

/** Points in one published cloud, at most -- mirrors mesh.session.LIDAR_CLOUD_MAX_POINTS. */
export const MAX_POINTS = 4000

/** Publish rate of the cloud topic (Hz) -- mirrors mesh.session.LIDAR_CLOUD_HZ. */
export const CLOUD_HZ = 5

/** The wire budget those three imply: the most this stream may ever cost, in bytes/second. */
export const MAX_BYTES_PER_SECOND = MAX_POINTS * BYTES_PER_POINT * CLOUD_HZ

export interface LidarCloud {
  /** Points decoded. */
  n: number
  /** 3n interleaved xyz, ready for a BufferAttribute. */
  xyz: Float32Array
  /** n intensities, in whatever range the sensor reports. */
  intensity: Float32Array
}

/**
 * Is this host little-endian? The wire format is named `xyzi_f32le`, and a typed array reads with
 * HOST byte order -- so on a big-endian host the same buffer decodes to different numbers. Measured
 * rather than assumed, because the failure is silent: every coordinate is finite and wrong.
 */
export function hostIsLittleEndian(): boolean {
  const probe = new ArrayBuffer(4)
  new DataView(probe).setFloat32(0, 1.5, true)
  return new Float32Array(probe)[0] === 1.5
}

/**
 * Decode a /ws/lidar frame into xyz + intensity, or null when it is not a cloud.
 *
 * Refused rather than salvaged:
 *  - a length that is not a whole number of points. Rounding down would build points whose
 *    coordinates come from two different rows -- a plausible-looking shape that was never measured.
 *  - a big-endian host, for the reason in `hostIsLittleEndian`.
 *
 * A misaligned view is COPIED, not refused: `new Float32Array(buf, byteOffset)` throws
 * "start offset of Float32Array should be a multiple of 4" (measured), and a caller handing over a
 * slice of a larger read buffer has done nothing wrong.
 */
export function decodeCloud(frame: ArrayBuffer | ArrayBufferView): LidarCloud | null {
  if (!hostIsLittleEndian()) return null

  let buffer: ArrayBuffer
  let byteOffset: number
  let byteLength: number
  if (frame instanceof ArrayBuffer) {
    buffer = frame; byteOffset = 0; byteLength = frame.byteLength
  } else {
    buffer = frame.buffer as ArrayBuffer; byteOffset = frame.byteOffset; byteLength = frame.byteLength
  }
  if (byteLength === 0 || byteLength % BYTES_PER_POINT !== 0) return null

  if (byteOffset % 4 !== 0) {
    const copy = new Uint8Array(byteLength)
    copy.set(new Uint8Array(buffer, byteOffset, byteLength))
    buffer = copy.buffer; byteOffset = 0
  }

  const flat = new Float32Array(buffer, byteOffset, byteLength / 4)
  const n = byteLength / BYTES_PER_POINT
  const xyz = new Float32Array(n * 3)
  const intensity = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    xyz[i * 3] = flat[i * 4]
    xyz[i * 3 + 1] = flat[i * 4 + 1]
    xyz[i * 3 + 2] = flat[i * 4 + 2]
    intensity[i] = flat[i * 4 + 3]
  }
  return { n, xyz, intensity }
}

export interface CloudBudget {
  points: number
  hz: number
  bytesPerSecond: number
  /** Within the ceiling MAX_POINTS/CLOUD_HZ imply. */
  withinCap: boolean
  /** Why it is over, in the operator's words -- empty when it is not. */
  reason: string
}

/**
 * What a stream of `points` at `hz` costs on the wire, and whether that is inside the cap.
 *
 * The cap is not a taste: it is what the publisher's own two limits multiply out to. A tile that
 * asks for more than this is asking for traffic the sensor cannot produce, so the honest answer is
 * to say so rather than to let the socket find out.
 */
export function cloudBudget(points: number, hz: number, cap: number = MAX_BYTES_PER_SECOND): CloudBudget {
  const p = Number.isFinite(points) && points > 0 ? Math.floor(points) : 0
  const rate = Number.isFinite(hz) && hz > 0 ? hz : 0
  const bytesPerSecond = p * BYTES_PER_POINT * rate
  const withinCap = bytesPerSecond <= cap
  return {
    points: p,
    hz: rate,
    bytesPerSecond,
    withinCap,
    reason: withinCap
      ? ''
      : `${Math.round(bytesPerSecond / 1024)} kB/s exceeds the ${Math.round(cap / 1024)} kB/s this stream is budgeted for`,
  }
}

/**
 * Intensity -> an rgb triple in 0..1, dark blue through cyan and yellow to white.
 *
 * `lo`/`hi` are the range to stretch across, because a sensor's intensity range is its own: a
 * MID-360 reports 0..255 reflectivity where a simulated cloud may report 0..1, and a ramp hard-coded
 * to either renders the other as one flat colour.
 *
 * A non-finite intensity colours as the floor rather than propagating: the point's POSITION is a
 * real measurement even when its return strength is not, so dropping it would hide geometry.
 */
export function intensityColor(value: number, lo = 0, hi = 1): [number, number, number] {
  const span = hi - lo
  const raw = !Number.isFinite(value) || span <= 0 ? 0 : (value - lo) / span
  const t = Math.min(1, Math.max(0, raw))
  // Four stops, linearly blended: 0 deep blue, 1/3 cyan, 2/3 yellow, 1 white.
  const stops: [number, number, number][] = [[0.05, 0.1, 0.5], [0, 0.8, 0.9], [1, 0.9, 0.2], [1, 1, 1]]
  const scaled = t * (stops.length - 1)
  const i = Math.min(stops.length - 2, Math.floor(scaled))
  const f = scaled - i
  const a = stops[i]
  const b = stops[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

export interface CloudMeta {
  peerId: string
  /** Points in the cloud that just arrived. */
  n: number
  /** Points the sensor produced before the publisher's budget downsampled it, when it said. */
  rawCount: number | null
  /** Downsample stride the publisher used, when it said. */
  stride: number | null
  bytes: number
  t: number | null
}

/**
 * Read a `lidar_cloud` frame off /ws/mesh, or null when the event is something else.
 *
 * This frame carries NO points -- it is the notification that a new cloud is available, so a tile
 * knows to expect bytes on /ws/lidar and a fleet page can say "publishing" without opening one.
 */
export function cloudMetaFromEvent(ev: unknown): CloudMeta | null {
  if (!ev || typeof ev !== 'object') return null
  const e = ev as { type?: unknown; peer_id?: unknown; data?: unknown }
  if (e.type !== 'lidar_cloud' || typeof e.peer_id !== 'string') return null
  const d = (e.data && typeof e.data === 'object' ? e.data : {}) as Record<string, unknown>
  const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
  return {
    peerId: e.peer_id,
    n: num(d.n) ?? 0,
    rawCount: num(d.raw_count),
    stride: num(d.stride),
    bytes: num(d.bytes) ?? 0,
    t: num(d.t),
  }
}

/**
 * How many of the sensor's own points this cloud represents, in the operator's words.
 *
 * A downsample presented as the whole sweep is the thing this sentence exists to prevent: 4000
 * points drawn from 24000 is a sixth of the returns, and an operator reading density off the tile
 * should be told that rather than left to assume.
 */
export function coverageNote(meta: CloudMeta): string {
  if (meta.n <= 0) return 'no points'
  if (meta.rawCount === null || meta.rawCount <= meta.n) return `${meta.n} points`
  return `${meta.n} of ${meta.rawCount} points (every ${meta.stride ?? Math.round(meta.rawCount / meta.n)}th)`
}

/** A `lidar_error` frame that means the SERVER is throttling this tile, not that the sensor died. */
export function lidarThrottleNotice(ev: unknown): string | null {
  if (!ev || typeof ev !== 'object') return null
  const e = ev as { type?: unknown; throttled?: unknown; error?: unknown }
  if (e.type !== 'lidar_error' || e.throttled !== true) return null
  return typeof e.error === 'string' && e.error
    ? e.error
    : 'the server is pacing this point cloud while it settles'
}
