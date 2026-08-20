/**
 * What the "Servo boards" list may say when it has nothing to show.
 *
 * It said `no servo board detected` whenever the array was empty — and the array is
 * `doc?.serial_ports ?? []`, so it was empty in three completely different situations:
 * the scan has not answered yet, the scan FAILED (a 401 through the tunnel, a dead
 * dashboard, an exception in enumeration), or the scan genuinely found nothing. Only
 * the third is a statement about hardware, and it is the one an operator acts on: with
 * two arms plugged in and a failing request, the screen told them their boards were
 * gone. This is the same rule the camera copy just learned (lib/cameraEvidence) and the
 * same idiom `portChoice({ scanned })` already uses in this file: absence of an answer
 * is not an answer.
 *
 * `detected` is the only verdict allowed to name hardware. The others name the SCAN,
 * because that is what is actually known.
 */

export type BoardListEmpty =
  | { kind: 'scanning'; message: string }
  | { kind: 'unscanned'; message: string }
  | { kind: 'detected'; message: string }

export function boardListEmptyLine(opts: { scanned: boolean; error?: string | null }): BoardListEmpty {
  const err = (opts.error ?? '').trim()
  if (!opts.scanned) {
    if (err) {
      return {
        kind: 'unscanned',
        // Never "no boards": nothing was asked successfully, so nothing is known.
        message: `the device scan failed (${err}) — this list is empty because nothing answered, not because nothing is plugged in`,
      }
    }
    return { kind: 'scanning', message: 'scanning USB for servo boards…' }
  }
  // The scan answered and listed none: now the claim is about the hardware, and it says
  // what was actually looked for so an operator can tell it apart from a wiring fault.
  const suffix = err ? ` (the last refresh also reported: ${err})` : ''
  return {
    kind: 'detected',
    message: `no servo board detected — nothing on USB enumerated as a serial bus${suffix}`,
  }
}


/** The shared half of the rule: what to say when the scan itself has not spoken.
 *  One wording for every list on the devices screen, because "empty because nothing
 *  answered" is the SAME fact regardless of which array is empty, and three phrasings of
 *  it would read as three different problems. */
function unanswered(what: string, error?: string | null): BoardListEmpty {
  const err = (error ?? '').trim()
  if (err) {
    return {
      kind: 'unscanned',
      message: `the device scan failed (${err}) — this list is empty because nothing answered, not because there ${what}`,
    }
  }
  return { kind: 'scanning', message: 'scanning this machine…' }
}

/**
 * `Managed robots (0) — None.` came from `doc?.managed ?? {}`, so a failed or pending
 * /api/devices reported zero children — while children were running, publishing to the mesh
 * and holding serial ports. The count in the heading lies with it. Only an answered scan may
 * say "none".
 */
export function managedListEmptyLine(opts: { scanned: boolean; error?: string | null }): BoardListEmpty {
  if (!opts.scanned) return unanswered('are none', opts.error)
  return {
    kind: 'detected',
    message:
      'None. Spawning one starts a child process that joins the mesh as its own peer — ' +
      'use it for a MuJoCo sim, or to drive a real arm from this machine.',
  }
}

/**
 * `No cameras probed — plug one in and rescan.` came from `doc?.cameras ?? []`: an
 * instruction (go plug in hardware) derived from a request that may never have answered. And
 * on this rig the cameras ARE plugged in while macOS refuses them (Q25), so "plug one in" was
 * the wrong next move even when the scan did answer — the answered branch now points at the
 * probe result rather than at the cable.
 */
export function cameraGridEmptyLine(opts: { scanned: boolean; error?: string | null }): BoardListEmpty {
  if (!opts.scanned) return unanswered('are no cameras', opts.error)
  return {
    kind: 'detected',
    message: 'No camera index answered a probe — plug one in, or rescan if you just did.',
  }
}
