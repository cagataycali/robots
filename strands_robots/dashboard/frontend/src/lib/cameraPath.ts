/** How a camera frame reached this dashboard, for the tile caption.
 *
 *  The bridge tags every frame it files with `via` (`inline`: the JPEG rode the mesh topic;
 *  `s3`: the robot published an S3 reference over AWS IoT Core and the bridge fetched the
 *  frame server side) and `latency_ms` (publisher clock to bridge receive). Over the LAN that
 *  is tens of milliseconds and not worth a word; over IoT it is the number a fleet owner wants
 *  to see next to the picture, so the caption shows it once it passes `LATENCY_SHOWN_MS`. */

export interface CameraPathMeta { via?: string; latency_ms?: number | null }

/** Below this the latency is the LAN's and the caption stays quiet. */
export const LATENCY_SHOWN_MS = 100

/** `'S3'` for a fetched reference, `''` for an inline frame or an older server. */
export function cameraPathLabel(meta: CameraPathMeta | undefined): string {
  return meta?.via === 's3' ? 'S3' : ''
}

/** `'420 ms'` when the frame took a WAN-sized time to arrive, `''` otherwise. */
export function cameraLatencyLabel(meta: CameraPathMeta | undefined): string {
  const ms = meta?.latency_ms
  if (typeof ms !== 'number' || !isFinite(ms) || ms < LATENCY_SHOWN_MS) return ''
  return ms >= 10_000 ? `${(ms / 1000).toFixed(0)} s` : `${Math.round(ms)} ms`
}
