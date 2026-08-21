import { useEffect, useRef, useState } from 'react'
import { planRetry } from './cameraRetry'
import { mergeMeshEvent, sweepStale } from './meshPeers'
import type { ActivityEntry, MeshEvent, MeshInfo, Peer } from '../types'
import { authRefusedRecently, authToken, wsUrl } from './endpoints'
import { sessionVerdict } from './sessionExpiry'

import type { AbsentChild } from './absentChildren'

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
  /** dead managed children, already pruned from `peers` (U22). [] = none, and an
   *  older server that never sends the field also yields [] - absentNotice treats
   *  both as "nothing to say", never as "all present". */
  absentChildren: AbsentChild[]
  /** Q155b: ids we hold a live child process for that the fleet has never heard of. */
  quietChildren: string[]
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
  const [absentChildren, setAbsentChildren] = useState<AbsentChild[]>([])
  const [quietChildren, setQuietChildren] = useState<string[]>([])
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
          // Q102: expiry is only ONE way to be refused. A token rotated by a dashboard restart, a
          // stale ?token= link or a revoked session is invalid without being expired, and the
          // handshake refusal reaches the browser as 1006 — so without this the fleet view
          // reconnects against a door that already said no, exactly like the camera tiles did.
          pageRefused: authRefusedRecently(),
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
            // Peer merging lives in ./meshPeers as pure functions (tested there; inside this
            // handler it needed a live websocket). The snapshot's last_seen values are the
            // SERVER's clock, so mergeMeshEvent rebases them by AGE into this browser's.
            setPeers(p => mergeMeshEvent(p, ev, Date.now() / 1000))
            if (ev.mesh) setMesh(ev.mesh)
            // Replaced wholesale, never merged: the list IS the current answer, so a child
            // that came back must disappear from it on the very next snapshot.
            setAbsentChildren(Array.isArray(ev.absent_children) ? ev.absent_children : [])
            // An older server sends no field; [] means "nothing to say", which quietNotice
            // renders as nothing rather than as "all children reported in".
            setQuietChildren(Array.isArray(ev.managed_no_presence) ? ev.managed_no_presence : [])
            setLoaded(true)
            break
          case 'presence':
          case 'state':
          case 'stream':
          case 'camera_meta':
            setPeers(p => mergeMeshEvent(p, ev, Date.now() / 1000))
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
            setPeers(p => mergeMeshEvent(p, ev, Date.now() / 1000))
            break
        }
      }
    }
    connect()

    // stale sweep every 5s
    const sweep = setInterval(() => {
      const now = Date.now() / 1000
      setPeers(p => sweepStale(p, now))
    }, 5000)

    return () => {
      closed = true
      clearInterval(sweep)
      if (retryTimer) clearTimeout(retryTimer)
      if (flashTimer) clearTimeout(flashTimer)
      ws?.close()
    }
  }, [])

  return { conn, dashboardId, peers, safetyFlash, mesh, activity, absentChildren, quietChildren, loaded, lastEventAt, everOpen }
}
