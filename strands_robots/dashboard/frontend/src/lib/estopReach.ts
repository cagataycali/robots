/**
 * What an e-stop may claim when it targeted NOTHING.
 *
 * The sheet said `No live peers were on the mesh.` — the most dangerous sentence in this
 * dashboard, because it is read one second after someone hit the stop button while
 * watching an arm move. `targeted` is not the fleet: it is the peers THIS dashboard's
 * snapshot could see and considered live. A child whose presence has not arrived yet, a
 * peer pruned as stale, a process joined to the mesh that this view never learned about
 * (BUGS.md Q32: three ghost pytest processes were live on the fleet for days, and Q28:
 * delivery to a non-hub peer is not a property of the fleet) are all invisible here — and
 * every one of them can be holding a torqued servo.
 *
 * So the empty case must say what happened (nothing was sent), why it might be wrong, and
 * the one action that works regardless of the mesh: cut power. `stale_skipped` peers are
 * NOT reassurance either — they were skipped precisely because they could not be reached,
 * so they are named as unstopped rather than counted as handled.
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
