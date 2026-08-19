import { useRef } from 'react'
import type { Presence, PeerState } from '../types'
import { decideStripScale, fillPercent, type ScaleMemo } from '../lib/jointScale'

/**
 * One scale per strip, learned from the stream.
 *
 * The scale decision lives in `../lib/jointScale` as pure functions: a robot
 * reports every joint in one unit, so the unit belongs to the strip, and a
 * change of unit needs several consecutive frames of agreement before the
 * axis moves. See that module for why the old per-joint
 * `|pos| > 4 ? 100 : PI` rule made bars incomparable and jumpy.
 */

function readValue(v: unknown): number {
  if (typeof v === 'number') return v
  if (Array.isArray(v)) return (v[0] as number) ?? 0
  if (v && typeof v === 'object') return (v as { position?: number }).position ?? 0
  return 0
}

export default function JointStrip({ state, presence }: { state?: PeerState; presence?: Presence }) {
  const memo = useRef<ScaleMemo | undefined>(undefined)

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
  const samples: Array<[string, number]> = entries.map(([name, v]) => [name, readValue(v)])
  memo.current = decideStripScale(samples, memo.current)
  const { unit, ranges } = memo.current

  return (
    <div className="joints" data-unit={unit}>
      {entries.map(([name, v], i) => {
        const pos = samples[i][1]
        const range = ranges[name]
        const pct = fillPercent(pos, range)
        const vel = (v && typeof v === 'object' && !Array.isArray(v))
          ? (v as { velocity?: number }).velocity
          : undefined
        return (
          <div
            className="joint"
            key={name}
            title={`${name}: ${pos.toFixed(3)}${vel !== undefined ? ` (v=${vel.toFixed(2)})` : ''} · ${unit} scale ${range.lo.toFixed(2)}…${range.hi.toFixed(2)}`}
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
