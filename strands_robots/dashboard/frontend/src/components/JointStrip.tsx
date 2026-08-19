import { useEffect, useRef, useState } from 'react'
import type { Presence, PeerState } from '../types'
import { decideStripScale, fillPercent, type ScaleMemo } from '../lib/jointScale'
import { createHistory, pushFrame, HISTORY_WINDOW_MS } from '../lib/jointHistory'
import JointSpark from './JointSpark'
import { humanJointNames, stripLegend } from '../lib/jointLabels'

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

export default function JointStrip({
  state, presence, history: showHistory = true,
}: { state?: PeerState; presence?: Presence; history?: boolean }) {
  const memo = useRef<ScaleMemo | undefined>(undefined)
  const hist = useRef(createHistory())
  const [frame, setFrame] = useState(0)
  const pending = useRef<Array<[string, number]> | null>(null)

  // History is appended in an effect, never during render: React may render the
  // same state twice and a double-appended frame would be a fabricated sample.
  useEffect(() => {
    if (!showHistory || !pending.current) return
    pushFrame(hist.current, pending.current, Date.now())
    setFrame((f) => f + 1)
  }, [state, showHistory])

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
  pending.current = samples

  // The number, not just the bar: bars without values are decoration on a
  // machine that can hit things (UX review #4). Precision adapts to the
  // stream's magnitude; the unit is NOT invented — 'servo' streams mix
  // degrees and 0..100 grippers, so a fabricated '°' would lie for some rows.
  const fmt = (n: number) =>
    Math.abs(n) >= 100 ? n.toFixed(0) : Math.abs(n) >= 10 ? n.toFixed(1) : n.toFixed(2)

  // UX_REVIEW #4's remaining half: the numbers were there, but nothing on
  // screen said what unit they are in or what the bar and the line mean — that
  // lived only in `title=`, which does not exist on a touch screen. Labels fall
  // back to raw keys if humanising two rows would read the same.
  const labels = humanJointNames(entries.map(([name]) => name))

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
            <div className="jname" title={name}>{labels[i]}</div>
            <div className="jval">{fmt(pos)}</div>
            <div className="jbar">
              <div className="jfill" style={{ width: `${pct}%` }} />
              {vel !== undefined && Math.abs(vel) > 1e-3 && <span className="jvel" />}
            </div>
            {showHistory && (
              <div className="jhist" title={`last ${Math.round(HISTORY_WINDOW_MS / 1000)}s of ${name}`}>
                <JointSpark track={hist.current.get(name)} range={range} frame={frame} />
              </div>
            )}
          </div>
        )
      })}
      <div className="jlegend">{stripLegend(unit, HISTORY_WINDOW_MS, entries.map(([n]) => n))}</div>
    </div>
  )
}
