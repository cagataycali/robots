import type { PeerState } from '../types'

/** Live joint position bars from the state topic. */
export default function JointStrip({ state }: { state?: PeerState }) {
  const joints = state?.joints
  if (!joints) return <div className="joints empty">no joint data</div>
  const entries = Object.entries(joints).slice(0, 12)
  return (
    <div className="joints">
      {entries.map(([name, v]) => {
        let pos = 0
        if (typeof v === 'number') pos = v
        else if (Array.isArray(v)) pos = v[0] ?? 0
        else if (v && typeof v === 'object') pos = (v as { position?: number }).position ?? 0
        // map roughly [-3.14, 3.14] (or [-100,100] servo) into 0..100%
        const span = Math.abs(pos) > 4 ? 100 : Math.PI
        const pct = Math.max(0, Math.min(100, ((pos + span) / (2 * span)) * 100))
        return (
          <div className="joint" key={name} title={`${name}: ${pos.toFixed(3)}`}>
            <div className="jname">{name.replace(/(_pos|\.pos)$/, '')}</div>
            <div className="jbar"><div className="jfill" style={{ width: `${pct}%` }} /></div>
          </div>
        )
      })}
    </div>
  )
}
