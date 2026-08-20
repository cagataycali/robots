/**
 * What the devices drawer may claim when a spawn/despawn fails.
 *
 * `act()` answered every thrown request with `⚠ <message>`, which reads as "the
 * button did nothing". Same trap as lib/taskOutcome.ts and lib/submitOutcome.ts:
 * api() throws HttpError(0) for a request that never left the machine AND for one
 * that reached the server, ran, and lost the answer; a 5xx means the handler
 * already executed. Here the two worlds are:
 *
 *   spawn   -> a child process may ALREADY be holding the USB port and reading
 *              the servo bus. A second spawn on the same port is the "[TxRxResult]
 *              Port is in use!" collision class (see strands_robots/bus_access.py).
 *   despawn -> the robot may ALREADY have been killed. If it was mid-episode,
 *              that take is gone; believing it still runs is worse than knowing.
 *
 * And there is an observer to hand off to: /api/devices lists what is actually
 * managed and alive, so the honest move is to refresh it and say "if it appears
 * (or disappears) there, that is your answer" instead of guessing here. `act()`
 * only reloaded on the SUCCESS path, which is exactly backwards - the ambiguous
 * case is the one that needs the list.
 */
import { refusedBeforeActing } from './estopOutcome'

export type DeviceAction = 'spawn' | 'despawn'

export interface DeviceFailureVerdict {
  text: string
  /** true = the action may have taken effect; refresh the list before retrying. */
  ambiguous: boolean
}

export function deviceActionFailure(input: {
  kind: DeviceAction
  status?: number | null
  message?: string | null
}): DeviceFailureVerdict {
  const why = String(input.message ?? '').trim() || 'no detail'
  const spawn = input.kind === 'spawn'

  if (refusedBeforeActing(input.status)) {
    return {
      text: spawn
        ? `✗ refused (${input.status}): ${why} — no process was started, nothing new is holding the serial port.`
        : `✗ refused (${input.status}): ${why} — the robot was NOT stopped; it is still running (and still recording, if it was).`,
      ambiguous: false,
    }
  }
  const head = Number(input.status ?? 0)
    ? `⚠ unknown — the server failed mid-request (${input.status}: ${why})`
    : `⚠ unknown — no answer came back (${why})`
  return {
    text: spawn
      ? `${head}: the robot MAY have started and a process MAY already hold that port. `
        + 'Refreshing the device list — if it appears there, it started. '
        + 'Spawning again could put a second process on the same bus, which is the "Port is in use!" collision.'
      : `${head}: the robot MAY already have been killed — if it was mid-episode, that take is gone. `
        + 'Refreshing the device list — if it disappears from it, the despawn landed.',
    ambiguous: true,
  }
}
