import { useEffect, useRef, useState } from 'react'
import { wsUrl } from '../lib/endpoints'
import { classifyCamera, type CamStatus } from '../lib/cameraState'
import { planRetry } from '../lib/cameraRetry'

interface Meta { t?: number; shape?: number[]; encoding?: string; displayable?: boolean; error?: string }

/**
 * Binary JPEG stream over /ws/camera/{peer}/{cam} → <img>.
 *
 * A camera tile that is simply black is the least debuggable thing on the
 * dashboard, and one showing a frozen last-good frame at full brightness is
 * actively misleading. Every state is named by `classifyCamera` and, when the
 * pixels are stale, the image itself is dimmed so it cannot pass for live.
 *
 * A closed socket also retries with backoff instead of sitting dead until
 * someone reloads the page - and it stops retrying when the close was a refusal
 * (1008), because hammering a door that said no is not resilience.
 *
 * The backoff counter is only cleared by EVIDENCE (a frame, or a socket that stayed
 * open): this server accepts and authenticates the handshake before discovering that
 * nothing is publishing, so `onopen` firing means nothing. Resetting there is what
 * turned ten hours of a missing arm into 63,906 sockets from a phone on cellular
 * data (BUGS.md Q40) - see lib/cameraRetry.
 */
export default function CameraTile({ peerId, cam, big = false, meta }: {
  peerId: string; cam: string; big?: boolean; meta?: Meta
}) {
  const imgRef = useRef<HTMLImageElement>(null)
  const [status, setStatus] = useState<CamStatus>(() =>
    classifyCamera({ now: Date.now(), conn: 'connecting', frames: 0 }))
  const [fps, setFps] = useState(0)

  // Per-frame bookkeeping lives in refs: 30 setState calls a second per tile is
  // how six camera tiles turn a fluid dashboard into a slideshow.
  const frames = useRef(0)
  const lastFrameAt = useRef<number | undefined>(undefined)
  const times = useRef<number[]>([])
  const conn = useRef<'connecting' | 'open' | 'closed'>('connecting')
  const error = useRef<string | null>(null)
  const retryAt = useRef<number | undefined>(undefined)
  // Attempts survive a re-run of the effect: peerId/cam churn must not hand a dead
  // endpoint a fresh 1s retry budget.
  const tries = useRef(0)
  // Read inside the ticker, so a metadata update (meta.t changes with every published
  // frame) no longer tears the socket down and rebuilds it.
  const metaRef = useRef(meta)
  metaRef.current = meta

  useEffect(() => {
    let ws: WebSocket | null = null
    let url: string | null = null
    let stopped = false
    let retryTimer: number | undefined
    let openedAt: number | undefined
    let framesThisSocket = 0

    // One ticker owns every visible change, so a stall becomes visible on its
    // own: when a stream dies, nothing arrives to trigger a render.
    const tick = window.setInterval(() => {
      const now = Date.now()
      setStatus(classifyCamera({
        now, conn: conn.current, frames: frames.current,
        lastFrameAt: lastFrameAt.current, error: error.current,
        publishedAt: metaRef.current?.t,
        retryInMs: retryAt.current !== undefined ? Math.max(0, retryAt.current - now) : undefined,
        attempt: retryAt.current !== undefined ? tries.current : undefined,
      }))
      const t = performance.now()
      times.current = times.current.filter(x => t - x < 2000)
      setFps(times.current.length > 1 ? (times.current.length - 1) / ((t - times.current[0]) / 1000) : 0)
    }, 400)

    const open = () => {
      if (stopped) return
      conn.current = 'connecting'
      ws = new WebSocket(wsUrl(`/ws/camera/${encodeURIComponent(peerId)}/${encodeURIComponent(cam)}`))
      ws.binaryType = 'blob'
      // NOT a reset: this handshake succeeding says nothing about whether frames exist.
      ws.onopen = () => { conn.current = 'open'; openedAt = Date.now(); retryAt.current = undefined }
      ws.onmessage = (msg) => {
        if (typeof msg.data === 'string') {
          // The server tells us *why* there are no pixels (e.g. the peer is
          // publishing raw frames it could not transcode).
          try {
            const ev = JSON.parse(msg.data)
            if (ev.type === 'camera_error') error.current = ev.error
          } catch { /* ignore */ }
          return
        }
        if (!(msg.data instanceof Blob)) return
        const next = URL.createObjectURL(msg.data)
        if (imgRef.current) imgRef.current.src = next
        if (url) URL.revokeObjectURL(url)
        url = next
        error.current = null
        frames.current += 1
        framesThisSocket += 1
        lastFrameAt.current = Date.now()
        times.current.push(performance.now())
      }
      ws.onclose = (ev) => {
        conn.current = 'closed'
        const plan = planRetry({
          attempt: tries.current,
          frames: framesThisSocket,
          openMs: openedAt !== undefined ? Date.now() - openedAt : undefined,
          code: ev.code,
        })
        tries.current = plan.attempt
        if (plan.delayMs === null) {
          error.current = 'unauthorized'
          retryAt.current = undefined
          return
        }
        if (stopped) return
        retryAt.current = Date.now() + plan.delayMs
        retryTimer = window.setTimeout(open, plan.delayMs)
      }
    }
    open()

    return () => {
      stopped = true
      window.clearInterval(tick)
      if (retryTimer) window.clearTimeout(retryTimer)
      ws?.close()
      if (url) URL.revokeObjectURL(url)
    }
  }, [peerId, cam])

  const shape = meta?.shape?.length ? `${meta.shape[1]}×${meta.shape[0]}` : null
  const cls = ['camtile', big ? 'big' : '', `cam-${status.kind}`, status.frozen ? 'frozen' : '']
    .filter(Boolean).join(' ')

  return (
    <div className={cls}>
      <img ref={imgRef} alt={`${peerId} ${cam} camera`} />
      {!status.live && (
        // aria-live: a stream dying is news. A sighted user gets it from the
        // overlay appearing; a screen reader needs it announced.
        <div className="camstate" role="status" aria-live="polite">
          <b>{status.title}</b>
          {status.detail && <span>{status.detail}</span>}
          {status.frozen && <span className="camstale">showing the last frame received</span>}
        </div>
      )}
      <span className={status.live ? 'camlabel live' : 'camlabel'}>
        {cam}
        {status.live && fps > 0 && <em> {fps.toFixed(0)}fps</em>}
        {shape && <em> {shape}</em>}
      </span>
    </div>
  )
}
