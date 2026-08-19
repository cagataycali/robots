import { useEffect, useRef, useState } from 'react'
import type { ActivityEntry, MeshEvent, MeshInfo, Peer } from '../types'
import { wsUrl } from './endpoints'

export type ConnState = 'connecting' | 'open' | 'closed' | 'unauthorized'

export interface MeshStore {
  conn: ConnState
  dashboardId: string
  peers: Record<string, Peer>
  safetyFlash: string | null
  mesh: MeshInfo
  activity: ActivityEntry[]
  /** true once a snapshot has arrived - "no robots" only means something then. */
  loaded: boolean
  /** epoch ms of the last frame on the socket (undefined = none yet) */
  lastEventAt?: number
  /** true once the socket has opened at least once this session */
  everOpen: boolean
}

const ACTIVITY_CAP = 200

/** One WebSocket to /ws/mesh → normalized reactive fleet store. */
export function useMesh(): MeshStore {
  const [conn, setConn] = useState<ConnState>('connecting')
  const [dashboardId, setDashboardId] = useState('')
  const [peers, setPeers] = useState<Record<string, Peer>>({})
  const [safetyFlash, setSafetyFlash] = useState<string | null>(null)
  const [mesh, setMesh] = useState<MeshInfo>({})
  const [activity, setActivity] = useState<ActivityEntry[]>([])
  const [loaded, setLoaded] = useState(false)
  // When the last frame arrived, and whether the socket ever opened this
  // session: lib/linkHealth needs both to tell ordinary startup from a
  // reconnect, and to say how old a frozen fleet view is IN SECONDS.
  const [lastEventAt, setLastEventAt] = useState<number | undefined>(undefined)
  const [everOpen, setEverOpen] = useState(false)
  const retryRef = useRef(0)

  useEffect(() => {
    let ws: WebSocket | null = null
    let closed = false
    let flashTimer: ReturnType<typeof setTimeout>

    const connect = () => {
      ws = new WebSocket(wsUrl('/ws/mesh'))
      setConn('connecting')

      ws.onopen = () => { setConn('open'); setEverOpen(true); retryRef.current = 0 }
      ws.onclose = (ev) => {
        // 1008 is the server refusing our token. Retrying forever just hides a
        // fixable problem behind a spinner.
        if (ev.code === 1008) { setConn('unauthorized'); return }
        setConn('closed')
        if (!closed) {
          const delay = Math.min(1000 * 2 ** retryRef.current++, 15000)
          setTimeout(connect, delay)
        }
      }
      ws.onmessage = (msg) => {
        let ev: MeshEvent
        try { ev = JSON.parse(msg.data) } catch { return }
        // Batched with the state update below in the same handler, so this is
        // one render, not two.
        setLastEventAt(Date.now())
        switch (ev.type) {
          case 'snapshot':
            setDashboardId(ev.dashboard_peer_id)
            setPeers(ev.peers)
            if (ev.mesh) setMesh(ev.mesh)
            setLoaded(true)
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
          case 'activity':
            setActivity(a => [ev.data, ...a].slice(0, ACTIVITY_CAP))
            break
          case 'mesh_reconfigured':
            // The session was re-pointed under us: the old peer list belongs to
            // the old mesh, so drop it rather than show ghosts.
            setMesh(ev.mesh)
            setPeers({})
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

  return { conn, dashboardId, peers, safetyFlash, mesh, activity, loaded, lastEventAt, everOpen }
}
