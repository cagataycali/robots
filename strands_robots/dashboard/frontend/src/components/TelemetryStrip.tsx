import type { Peer } from '../types'
import { useTelemetry, TELEMETRY_CAP } from '../lib/useTelemetry'

/**
 * Rate + motion telemetry from the state topic.
 *
 * "Is this robot actually alive?" is not answerable from a presence dot: a peer
 * can heartbeat happily while its state topic has been frozen for a minute.
 * Measured Hz and a motion trace answer it directly.
 *
 * The ring itself lives in useTelemetry, shared with the card's status
 * sentence so both read the SAME motion judgment.
 */
export default function TelemetryStrip({ peer }: { peer: Peer }) {
  const { samples, hz, moving, stateAgeS } = useTelemetry(peer)
  if (samples.length < 2) return null

  const peak = Math.max(...samples.map(s => s.motion), 1e-6)
  const age = stateAgeS ?? 0

  const points = samples
    .map((s, i) => `${(i / (TELEMETRY_CAP - 1)) * 100},${20 - (s.motion / peak) * 18}`)
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
