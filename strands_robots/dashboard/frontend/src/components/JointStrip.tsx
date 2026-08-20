import { useEffect, useRef, useState } from 'react'
import type { Presence, PeerState } from '../types'
import { decideStripScale, fillPercent, type ScaleMemo } from '../lib/jointScale'
import { createHistory, pushFrame, HISTORY_WINDOW_MS } from '../lib/jointHistory'
import JointSpark from './JointSpark'
import { humanJointNames, stripLegend } from '../lib/jointLabels'
import { jointAgeNote } from '../lib/jointFreshness'
import { jointAbsence } from '../lib/jointAbsence'

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
  state, presence, problem, history: showHistory = true,
}: {
  state?: PeerState; presence?: Presence; history?: boolean
  /** the backend's `joint_problem` verdict for this peer (Q80) — absent means nothing is known */
  problem?: { kind?: string | null; headline?: string | null; remedy?: string | null; detail?: string | null } | null
}) {
  const memo = useRef<ScaleMemo | undefined>(undefined)
  const hist = useRef(createHistory())
  const [frame, setFrame] = useState(0)
  const pending = useRef<Array<[string, number]> | null>(null)
  // When the newest frame arrived. The strip's values are only "the arm's
  // position" for as long as this is recent - see lib/jointFreshness.ts.
  const lastAt = useRef<number | null>(null)
  const [nowMs, setNowMs] = useState(() => Date.now())

  // History is appended in an effect, never during render: React may render the
  // same state twice and a double-appended frame would be a fabricated sample.
  useEffect(() => {
    if (!pending.current) return
    lastAt.current = Date.now()
    setNowMs(lastAt.current)
    if (!showHistory) return
    pushFrame(hist.current, pending.current, lastAt.current)
    setFrame((f) => f + 1)
  }, [state, showHistory])

  // An age that only updates when new data arrives can never say "no new data".
  // 1Hz is enough to cross both thresholds visibly and costs nothing.
  useEffect(() => {
    const t = setInterval(() => setNowMs(Date.now()), 1000)
    return () => clearInterval(t)
  }, [])

  const joints = state?.joints
  if (!joints || Object.keys(joints).length === 0) {
    // "no joint data on this peer" was the sentence a LOCKED-OUT arm showed for ten
    // hours while it published state 0.3s old: alive, talking, and only the joints
    // missing. lib/jointAbsence separates the three situations the operator has to
    // act on differently, and points at the log rather than guessing the cause.
    // Since Q80 the backend may KNOW the reason (it reads the child's log): a held serial port and
    // an uncalibrated board are opposite remedies that both used to render as the same shrug.
    const note = jointAbsence({ state, presence, problem, nowS: nowMs / 1000 })
    return (
      <div className="joints empty" data-tone={note.tone} title={note.detail ?? undefined}>
        {note.tone === 'attention' ? '⚠ ' : ''}{note.text}
        {note.hint && <span className="hint">{note.hint}</span>}
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

  // The strip carries its own freshness because it is rendered ALONE in the
  // collect panel, where the operator is hand-guiding the leader and reading
  // these very numbers to see the follower track. A neighbour's "stale" chip
  // (TelemetryStrip, on the cards) cannot cover that screen.
  const fresh = jointAgeNote(lastAt.current === null ? null : nowMs - lastAt.current)

  return (
    <div className="joints" data-unit={unit} data-fresh={fresh.level}>
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
      {/* role=status, not alert: it is a degradation to notice, not a modal
          interruption - and it must be announced when it appears. */}
      {fresh.text && (
        <div className={`jstale${fresh.dim ? ' frozen' : ''}`} role="status" aria-live="polite">
          {fresh.text}
        </div>
      )}
      <div className="jlegend">{stripLegend(unit, HISTORY_WINDOW_MS, entries.map(([n]) => n))}</div>
    </div>
  )
}
