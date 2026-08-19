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
      {/* UX_REVIEW #6: a bare "9.7 Hz" next to a camera tile is ambiguous — say
          WHICH rate it is on screen, not only in a tooltip. */}
      <span className="metric" title="measured rate of this robot's state topic (nominal 10 Hz)">
        state {hz.toFixed(1)} Hz
      </span>
      {/* ...and "is it moving RIGHT NOW" is the question an operator asks before
          reaching over the desk, so it gets a chip with a dot instead of one more
          grey word in a row of grey words. */}
      <span
        className={`motionchip ${moving ? 'moving' : 'still'}`}
        title={moving
          ? 'joints are changing (mean absolute delta over the last samples) — keep hands clear'
          : 'joints are not changing (mean absolute delta over the last samples)'}
      >
        <span className="motiondot" aria-hidden />{moving ? 'moving' : 'still'}
      </span>
      {age > 3 && <span className="metric warn" title="no state message recently">stale {age.toFixed(0)}s</span>}
      {peer.state?.sim_time !== undefined && (
        <span className="metric" title="simulation clock">t={peer.state.sim_time.toFixed(1)}</span>
      )}
    </div>
  )
}
