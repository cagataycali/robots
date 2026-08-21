/**
 * How an arm reports that its serial bus had to be un-stranded (BUGS.md Q81).
 *
 * The backend cure is invisible on purpose: an exchange that dies mid-conversation leaves the motor
 * bus marked in-use, every later read fails before the port is even touched, and until this was fixed
 * an arm simply went mute for hours with healthy-looking presence. Now any read clears that flag and
 * the arm heals inside one telemetry cycle - which is right, and which is exactly why the operator
 * must still be told it happened. A flag strands for PHYSICAL reasons: a marginal USB cable, a hub
 * browning out under load, a connector working loose as the arm moves. Silence would hide a
 * degrading rig behind perfect joint traces until the recovery itself failed.
 *
 * So: nothing at all while the count is zero (a healthy arm earns no ornament), a quiet neutral mark
 * for the occasional hiccup, and a warn tone once the number stops looking like bad luck. The
 * threshold is a judgement, not a measurement, and it is stated here rather than buried in JSX so it
 * can be argued with.
 */

/** Above this many strandings in one session, the cable is the suspect - not chance. */
export const BUS_RECOVERY_WARN_AT = 5

export type BusRecoveryBadge = { label: string; tone: '' | 'warn'; title: string }

export function busRecoveryBadge(count: unknown): BusRecoveryBadge | null {
  const n = typeof count === 'number' && Number.isFinite(count) ? Math.floor(count) : 0
  if (n <= 0) return null
  const tone = n >= BUS_RECOVERY_WARN_AT ? 'warn' : ''
  const plural = n === 1 ? 'once' : `${n} times`
  const why =
    'This arm\'s serial bus was left marked in-use by an exchange that never finished, and the '
    + `dashboard cleared it and read again - ${plural} since this robot started.\n\n`
    + 'Nothing is wrong with the reading you are looking at: the joints below are real. But a bus '
    + 'strands for physical reasons - a marginal USB cable, a hub browning out under load, a '
    + 'connector working loose as the arm moves.\n\n'
  const verdict = n >= BUS_RECOVERY_WARN_AT
    ? 'This has now happened often enough to be a pattern rather than bad luck: swap the cable, '
      + 'try a powered hub or a different port, and prefer a direct connection over a chain of hubs. '
      + 'Recording a dataset through a bus this flaky risks episodes with gaps in them.'
    : 'Once or twice is a hiccup and needs nothing from you. Worth remembering if it keeps climbing.'
  return { label: n === 1 ? 'bus healed once' : `bus healed ×${n}`, tone, title: why + verdict }
}
