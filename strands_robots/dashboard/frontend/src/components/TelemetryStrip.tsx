import { useEffect, useRef, useState } from 'react'
import type { Peer } from '../types'

const CAP = 120   // ~12 s at the 10 Hz state topic

interface Sample { t: number; motion: number }

function jointValues(peer: Peer): number[] {
  const joints = peer.state?.joints
  if (!joints) return []
  return Object.values(joints).map(v => {
    if (typeof v === 'number') return v
    if (Array.isArray(v)) return v[0] ?? 0
    return (v as { position?: number }).position ?? 0
  })
}

/**
 * Rate + motion telemetry from the state topic.
 *
 * "Is this robot actually alive?" is not answerable from a presence dot: a peer
 * can heartbeat happily while its state topic has been frozen for a minute.
 * Measured Hz and a motion trace answer it directly.
 */
export default function TelemetryStrip({ peer }: { peer: Peer }) {
  const ring = useRef<Sample[]>([])
  const prev = useRef<number[]>([])
  const lastT = useRef<number | undefined>(undefined)
  const [, tick] = useState(0)

  const stateT = peer.state?.t

  useEffect(() => {
    if (stateT === undefined || stateT === lastT.current) return
    lastT.current = stateT
    const values = jointValues(peer)
    let motion = 0
    if (prev.current.length === values.length && values.length) {
      for (let i = 0; i < values.length; i++) motion += Math.abs(values[i] - prev.current[i])
      motion /= values.length
    }
    prev.current = values
    ring.current = [...ring.current, { t: Date.now() / 1000, motion }].slice(-CAP)
    tick(n => n + 1)
  }, [stateT])  // eslint-disable-line react-hooks/exhaustive-deps

  const samples = ring.current
  if (samples.length < 2) return null

  const span = samples[samples.length - 1].t - samples[0].t
  const hz = span > 0 ? (samples.length - 1) / span : 0
  const peak = Math.max(...samples.map(s => s.motion), 1e-6)
  const moving = samples.slice(-10).some(s => s.motion > peak * 0.05)
  const age = Date.now() / 1000 - samples[samples.length - 1].t

  const points = samples
    .map((s, i) => `${(i / (CAP - 1)) * 100},${20 - (s.motion / peak) * 18}`)
    .join(' ')

  return (
    <div className="telemetry">
      <svg className="spark" viewBox="0 0 100 20" preserveAspectRatio="none" aria-hidden>
        <polyline points={points} />
      </svg>
      <span className="metric" title="measured state-topic rate (nominal 10 Hz)">{hz.toFixed(1)} Hz</span>
      <span className={moving ? 'metric moving' : 'metric'} title="mean absolute joint delta">
        {moving ? 'moving' : 'still'}
      </span>
      {age > 3 && <span className="metric warn" title="no state message recently">stale {age.toFixed(0)}s</span>}
      {peer.state?.sim_time !== undefined && (
        <span className="metric" title="simulation clock">t={peer.state.sim_time.toFixed(1)}</span>
      )}
    </div>
  )
}
