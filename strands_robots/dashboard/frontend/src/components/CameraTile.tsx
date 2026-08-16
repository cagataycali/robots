import { useEffect, useRef, useState } from 'react'
import { wsUrl } from '../lib/endpoints'

interface Meta { t?: number; shape?: number[]; encoding?: string; displayable?: boolean; error?: string }

/**
 * Binary JPEG stream over /ws/camera/{peer}/{cam} → <img>.
 *
 * A camera tile that is simply black is the least debuggable thing on the
 * dashboard: no frames, undecodable frames, a closed socket and a stopped robot
 * all look identical. Each of those states says what it is here.
 */
export default function CameraTile({ peerId, cam, big = false, meta }: {
  peerId: string; cam: string; big?: boolean; meta?: Meta
}) {
  const imgRef = useRef<HTMLImageElement>(null)
  const [frames, setFrames] = useState(0)
  const [conn, setConn] = useState<'connecting' | 'open' | 'closed'>('connecting')
  const [error, setError] = useState<string | null>(null)
  const [fps, setFps] = useState(0)
  const times = useRef<number[]>([])

  useEffect(() => {
    const ws = new WebSocket(wsUrl(`/ws/camera/${encodeURIComponent(peerId)}/${encodeURIComponent(cam)}`))
    ws.binaryType = 'blob'
    let url: string | null = null
    ws.onopen = () => setConn('open')
    ws.onmessage = (msg) => {
      if (typeof msg.data === 'string') {
        // The server tells us *why* there are no pixels (e.g. the peer is
        // publishing raw frames it could not transcode).
        try {
          const ev = JSON.parse(msg.data)
          if (ev.type === 'camera_error') setError(ev.error)
        } catch { /* ignore */ }
        return
      }
      if (!(msg.data instanceof Blob)) return
      const next = URL.createObjectURL(msg.data)
      if (imgRef.current) imgRef.current.src = next
      if (url) URL.revokeObjectURL(url)
      url = next
      setError(null)
      setFrames(n => n + 1)
      const now = performance.now()
      times.current = [...times.current, now].filter(t => now - t < 2000)
      if (times.current.length > 1) {
        setFps((times.current.length - 1) / ((now - times.current[0]) / 1000))
      }
    }
    ws.onclose = (ev) => { setConn('closed'); if (ev.code === 1008) setError('unauthorized') }
    return () => { ws.close(); if (url) URL.revokeObjectURL(url) }
  }, [peerId, cam])

  const live = frames > 0 && !error
  const shape = meta?.shape?.length ? `${meta.shape[1]}×${meta.shape[0]}` : null

  return (
    <div className={big ? 'camtile big' : 'camtile'}>
      <img ref={imgRef} alt={`${peerId}/${cam}`} />
      {!live && (
        <div className="camstate">
          {error ? <><b>no image</b><span>{error}</span></>
            : conn === 'closed' ? <><b>disconnected</b><span>stream closed</span></>
            : meta?.t ? <><b>waiting</b><span>frames published, none arrived yet</span></>
            : <><b>no frames</b><span>{conn === 'open' ? 'camera silent' : 'connecting…'}</span></>}
        </div>
      )}
      <span className={live ? 'camlabel live' : 'camlabel'}>
        {cam}{live && fps > 0 && <em> {fps.toFixed(0)}fps</em>}{shape && <em> {shape}</em>}
      </span>
    </div>
  )
}
