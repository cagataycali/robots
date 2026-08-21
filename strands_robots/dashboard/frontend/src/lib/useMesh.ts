import { useEffect, useRef, useState } from 'react'
import { planRetry } from './cameraRetry'
import { frameProvesLiveness } from './liveness'
import type { ActivityEntry, MeshEvent, MeshInfo, Peer } from '../types'
import { authToken, wsUrl } from './endpoints'
import { sessionVerdict } from './sessionExpiry'

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
    let retryTimer: ReturnType<typeof setTimeout> | undefined

    // Per-socket evidence, reset by connect() itself: how many frames THIS socket
    // delivered and when it opened. `retryRef` deliberately outlives them.
    let framesThisSocket = 0
    let openedAt: number | undefined

    const connect = () => {
      // `closed` was checked when the retry was SCHEDULED, not when it fired, so
      // a backoff already in flight at teardown opened a socket afterwards: a
      // zombie /ws/mesh nobody closes, still pushing peers into an unmounted
      // tree. Both halves matter - clear the pending timer AND refuse here.
      if (closed) return
      framesThisSocket = 0
      openedAt = undefined
      ws = new WebSocket(wsUrl('/ws/mesh'))
      setConn('connecting')

      // Opening is NOT success: the server accepts and authenticates the socket before
      // it knows whether the mesh bridge can produce a snapshot, so a reset here is the
      // Q40 mistake — the delay would stay at 1s forever against a broken bridge. The
      // first frame is the evidence, and on a healthy socket it lands in milliseconds.
      // openedAt is stamped INSIDE onopen, not here: a handshake that hangs for 30
      // seconds and never opens would otherwise satisfy the "stayed open long enough"
      // evidence and clear the very history it is proving.
      ws.onopen = () => { openedAt = Date.now(); setConn('open'); setEverOpen(true) }
      ws.onclose = (ev) => {
        // 1008 is the server refusing our token. Retrying forever just hides a
        // fixable problem behind a spinner. planRetry owns that rule too.
        const plan = planRetry({
          attempt: retryRef.current,
          frames: framesThisSocket,
          openMs: openedAt !== undefined ? Date.now() - openedAt : undefined,
          code: ev.code,
          // Q88: a lapsed sign-in means the HANDSHAKE is refused (403) and no close code
          // ever arrives, so `code === 1008` cannot see it. Without this the fleet view
          // just freezes and reconnects forever — the exact shape that ran for 19.3 hours
          // on the camera sockets. AuthGate re-gates within 30s; this stops the traffic now.
          sessionExpired: sessionVerdict(authToken(), Date.now() / 1000).refusesUntilSignIn,
        })
        retryRef.current = plan.attempt
        if (plan.delayMs === null) { setConn('unauthorized'); return }
        setConn('closed')
        if (!closed) retryTimer = setTimeout(connect, plan.delayMs)
      }
      ws.onmessage = (msg) => {
        let ev: MeshEvent
        try { ev = JSON.parse(msg.data) } catch { return }
        // Batched with the state update below in the same handler, so this is
        // one render, not two.
        framesThisSocket += 1
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
              // A camera frame only vouches for the peer if the PEER captured it
              // recently. The server replays a camera's last cached frame to
              // every new subscriber, so mounting a tile used to resurrect a peer
              // that died hours ago: last_seen refreshed, stale cleared, card
              // green. The frame's own capture time settles it.
              const fresh = frameProvesLiveness({ frameT: (ev.data as any)?.t, nowS: Date.now() / 1000 })
              const cameras = { ...peer.cameras, [ev.cam]: ev.data }
              return {
                ...p,
                [ev.peer_id]: fresh
                  ? { ...peer, cameras, last_seen: Date.now() / 1000, stale: false }
                  : { ...peer, cameras },
              }
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

    return () => {
      closed = true
      clearInterval(sweep)
      if (retryTimer) clearTimeout(retryTimer)
      if (flashTimer) clearTimeout(flashTimer)
      ws?.close()
    }
  }, [])

  return { conn, dashboardId, peers, safetyFlash, mesh, activity, loaded, lastEventAt, everOpen }
}
