/**
 * What an e-stop may claim when it targeted NOTHING. The sheet said `No live peers were on the
 * mesh.` — the most dangerous sentence in this dashboard, because it is read one second after
 * someone hit the stop button while watching an arm move.
 */

export type EstopReach = { headline: string; detail: string; cutPower: boolean }

export function estopNothingTargeted(opts: { staleSkipped?: string[] }): EstopReach {
  const stale = (opts.staleSkipped ?? []).filter(Boolean)
  if (stale.length > 0) {
    return {
      headline: 'nothing was stopped — every peer was unreachable',
      detail:
        `no stop was delivered: ${stale.join(', ')} had no heartbeat, so ${stale.length === 1 ? 'it was' : 'they were'} ` +
        `skipped rather than stopped. A peer with no heartbeat can still be moving — losing telemetry is not ` +
        `stopping. If anything is moving, cut power at the supply.`,
      cutPower: true,
    }
  }
  return {
    headline: 'nothing was stopped — this view had no live peer to target',
    detail:
      'no stop was delivered, because this dashboard sees no live peer. That is a statement about this ' +
      'fleet view, not about the room: a robot started outside this dashboard, one whose presence has not ' +
      'arrived yet, or one already dropped as stale is invisible here and can still be moving. If anything ' +
      'is moving, cut power at the supply.',
    cutPower: true,
  }
}
