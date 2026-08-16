import { useEffect, useRef, useState } from 'react'

/** Binary JPEG stream over /ws/camera/{peer}/{cam} → <img>. */
export default function CameraTile({ peerId, cam, big = false }: { peerId: string; cam: string; big?: boolean }) {
  const imgRef = useRef<HTMLImageElement>(null)
  const [live, setLive] = useState(false)

  useEffect(() => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/camera/${peerId}/${cam}`)
    ws.binaryType = 'blob'
    let url: string | null = null
    ws.onmessage = (msg) => {
      if (!(msg.data instanceof Blob)) return
      const next = URL.createObjectURL(msg.data)
      if (imgRef.current) imgRef.current.src = next
      if (url) URL.revokeObjectURL(url)
      url = next
      setLive(true)
    }
    ws.onclose = () => setLive(false)
    return () => { ws.close(); if (url) URL.revokeObjectURL(url) }
  }, [peerId, cam])

  return (
    <div className={big ? 'camtile big' : 'camtile'}>
      <img ref={imgRef} alt={`${peerId}/${cam}`} />
      <span className={live ? 'camlabel live' : 'camlabel'}>{cam}</span>
    </div>
  )
}
