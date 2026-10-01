/** The ping button's answer, as the registry card reads it.
 *
 *  `POST /api/robots/{thing}/ping` sends one `status` read point to point over AWS IoT Core and
 *  maps what came back to a closed set of words (routes_mesh.PING_VERDICTS). The card shows the
 *  word and, where the broker or the Thing answered, how long the round trip took. */

import type { PingResult } from '../types'

export function pingLabel(ping: PingResult | undefined): string {
  if (!ping) return ''
  if (ping.pending) return 'pinging…'
  const ms = typeof ping.latency_ms === 'number' && isFinite(ping.latency_ms) ? ` in ${Math.round(ping.latency_ms)} ms` : ''
  switch (ping.verdict) {
    case 'answered': return `answered${ms}`
    case 'offline': return `offline (broker 404${ms})`
    case 'forbidden': return 'forbidden for this operator'
    case 'silent': return 'delivered, no answer'
    case 'unavailable': return 'no direct send on this backend'
    case 'refused': return 'refused'
    default: return ping.reason ? `error: ${ping.reason}` : 'error'
  }
}
