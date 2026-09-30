import type { PingResult, RegistryThing } from '../types'
import { pingLabel } from '../lib/pingVerdict'

/** How long ago `t` (epoch seconds) was, or "never heard" when nothing is known. */
export function lastSeenLabel(t: number | null | undefined, now = Date.now() / 1000): string {
  if (typeof t !== 'number' || !isFinite(t) || t <= 0) return 'never heard'
  const s = Math.max(0, Math.round(now - t))
  if (s < 90) return `${s}s ago`
  if (s < 5400) return `${Math.round(s / 60)} min ago`
  if (s < 172800) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} d ago`
}

/** A provisioned AWS IoT Thing that is not speaking on the mesh: a grey card so a fleet owner
 *  sees what exists next to what is live. Read only but for ping: one `status` read sent point to
 *  point over AWS IoT Core, answered by the broker (404 = offline) or by the Thing. */
export default function RegistryCard({ thing, onPing, ping }: {
  thing: RegistryThing; onPing?: (name: string) => void; ping?: PingResult
}) {
  const attrs = Object.entries(thing.attributes ?? {})
  const verdict = thing.connectivity === 'connected' ? 'broker says connected'
    : thing.connectivity === 'disconnected' ? 'broker says disconnected'
    : 'no connectivity index'
  return (
    <div className="card registry stale-known" role="group" aria-label={`provisioned thing ${thing.thing_name}`}>
      <div className="card-head">
        <span className="typebadge thing" title="an AWS IoT Thing in the registry; nothing heard from it on the mesh">thing</span>
        <span className="peername" title={thing.thing_name}>{thing.thing_name}</span>
        <span className="reachchip registry" title="known from the AWS IoT registry only">registry</span>
        {thing.thing_type && <span className="host">{thing.thing_type}</span>}
        <span className="dot off" role="img" aria-label="not heard on the mesh" title="not heard on the mesh" />
      </div>
      <div className="regnote">last seen {lastSeenLabel(thing.last_seen)} · {verdict}</div>
      {attrs.length > 0 && (
        <div className="regattrs">{attrs.map(([k, v]) => `${k}=${v}`).join(' ')}</div>
      )}
      {onPing && (
        <div className="controls">
          <button className="btn ghost small" onClick={() => onPing(thing.thing_name)} disabled={!!ping?.pending}
                  title="one direct message round trip over AWS IoT Core; an offline Thing answers 404 in under a second">
            ping
          </button>
          {ping && <span className={`pingnote ${ping.verdict}`} role="status" title={ping.reason || undefined}>{pingLabel(ping)}</span>}
        </div>
      )}
    </div>
  )
}
