import { useRef } from 'react'
import type { Presence, PeerState } from '../types'

/**
 * Per-joint observed range, learned from the stream.
 *
 * A single global span (`|pos| > 4 ? 100 : π`) puts a servo reading 0..100 and a
 * radian joint on the same axis, so a gripper at 45 % and an elbow at 0.7 rad
 * render identically and neither bar means anything. Tracking min/max per joint
 * makes each bar a real fraction of that joint's travel - and it costs one
 * comparison per sample.
 */
interface Range { lo: number; hi: number }

function readValue(v: unknown): number {
  if (typeof v === 'number') return v
  if (Array.isArray(v)) return (v[0] as number) ?? 0
  if (v && typeof v === 'object') return (v as { position?: number }).position ?? 0
  return 0
}

export default function JointStrip({ state, presence }: { state?: PeerState; presence?: Presence }) {
  const ranges = useRef<Record<string, Range>>({})

  const joints = state?.joints
  if (!joints || Object.keys(joints).length === 0) {
    // "no joint data" reads like a broken robot; say which case it is.
    const expected = presence?.action_keys?.length
    return (
      <div className="joints empty">
        {expected ? `waiting for state (${expected} joints expected)` : 'no joint data on this peer'}
      </div>
    )
  }

  const entries = Object.entries(joints).slice(0, 12)
  return (
    <div className="joints">
      {entries.map(([name, v]) => {
        const pos = readValue(v)
        const seen = ranges.current[name]
        // Seed from the unit the value looks like, then widen with observation:
        // a joint that has only ever been at 0 still needs a sane axis.
        const seed: Range = Math.abs(pos) > 4 ? { lo: -100, hi: 100 } : { lo: -Math.PI, hi: Math.PI }
        const range: Range = seen
          ? { lo: Math.min(seen.lo, pos), hi: Math.max(seen.hi, pos) }
          : { lo: Math.min(seed.lo, pos), hi: Math.max(seed.hi, pos) }
        ranges.current[name] = range

        const width = range.hi - range.lo || 1
        const pct = Math.max(0, Math.min(100, ((pos - range.lo) / width) * 100))
        const vel = (v && typeof v === 'object' && !Array.isArray(v))
          ? (v as { velocity?: number }).velocity
          : undefined
        return (
          <div
            className="joint"
            key={name}
            title={`${name}: ${pos.toFixed(3)}${vel !== undefined ? ` (v=${vel.toFixed(2)})` : ''} · observed ${range.lo.toFixed(2)}…${range.hi.toFixed(2)}`}
          >
            <div className="jname">{name.replace(/(_pos|\.pos)$/, '')}</div>
            <div className="jbar">
              <div className="jfill" style={{ width: `${pct}%` }} />
              {vel !== undefined && Math.abs(vel) > 1e-3 && <span className="jvel" />}
            </div>
          </div>
        )
      })}
    </div>
  )
}
