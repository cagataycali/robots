import { useEffect, useRef, useState } from 'react'
import type { MeshEvent, Peer } from '../types'

export type ConnState = 'connecting' | 'open' | 'closed'

export interface MeshStore {
  conn: ConnState
  dashboardId: string
  peers: Record<string, Peer>
  safetyFlash: string | null
}

/** One WebSocket to /ws/mesh → normalized reactive fleet store. */
export function useMesh(): MeshStore {
  const [conn, setConn] = useState<ConnState>('connecting')
  const [dashboardId, setDashboardId] = useState('')
  const [peers, setPeers] = useState<Record<string, Peer>>({})
  const [safetyFlash, setSafetyFlash] = useState<string | null>(null)
  const retryRef = useRef(0)

  useEffect(() => {
    let ws: WebSocket | null = null
    let closed = false
    let flashTimer: ReturnType<typeof setTimeout>

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/ws/mesh`)
      setConn('connecting')

      ws.onopen = () => { setConn('open'); retryRef.current = 0 }
      ws.onclose = () => {
        setConn('closed')
        if (!closed) {
          const delay = Math.min(1000 * 2 ** retryRef.current++, 15000)
          setTimeout(connect, delay)
        }
      }
      ws.onmessage = (msg) => {
        let ev: MeshEvent
        try { ev = JSON.parse(msg.data) } catch { return }
        switch (ev.type) {
          case 'snapshot':
            setDashboardId(ev.dashboard_peer_id)
            setPeers(ev.peers)
            break
          case 'presence':
            setPeers(p => ({ ...p, [ev.peer_id]: { ...p[ev.peer_id], peer_id: ev.peer_id, presence: ev.data, last_seen: Date.now() / 1000, stale: false } }))
            break
          case 'state':
            setPeers(p => ({ ...p, [ev.peer_id]: { ...p[ev.peer_id], peer_id: ev.peer_id, state: ev.data, last_seen: Date.now() / 1000, stale: false } }))
            break
          case 'stream':
            setPeers(p => ({ ...p, [ev.peer_id]: { ...p[ev.peer_id], peer_id: ev.peer_id, stream: ev.data, last_seen: Date.now() / 1000, stale: false } }))
            break
          case 'camera_meta':
            setPeers(p => {
              const peer = p[ev.peer_id] ?? { peer_id: ev.peer_id }
              return { ...p, [ev.peer_id]: { ...peer, cameras: { ...peer.cameras, [ev.cam]: ev.data }, last_seen: Date.now() / 1000, stale: false } }
            })
            break
          case 'safety':
            setSafetyFlash(ev.kind)
            clearTimeout(flashTimer)
            flashTimer = setTimeout(() => setSafetyFlash(null), 5000)
            break
        }
      }
    }
    connect()

    // stale sweep every 5s
    const sweep = setInterval(() => {
      const now = Date.now() / 1000
      setPeers(p => {
        let changed = false
        const next = { ...p }
        for (const [id, peer] of Object.entries(next)) {
          const stale = now - (peer.last_seen ?? 0) > 15
          if (stale !== peer.stale) { next[id] = { ...peer, stale }; changed = true }
        }
        return changed ? next : p
      })
    }, 5000)

    return () => { closed = true; clearInterval(sweep); ws?.close() }
  }, [])

  return { conn, dashboardId, peers, safetyFlash }
}
