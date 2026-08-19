/**
 * R1 / UX_REVIEW:107 — the copy for the scariest button in the collect flow.
 *
 * It used to say `open session`, which describes a data structure. What the
 * click actually does to the room: both arm peers are DESPAWNED (their cards
 * and telemetry leave the fleet), their USB ports are handed to the recorder,
 * and the follower's motors are energised to hold position. Nothing is
 * recorded yet — that waits for the first episode — but two real arms have
 * changed state, one of them now stiff.
 *
 * So the button names the action and the panel states the consequence BEFORE
 * the click, and in rehearsal mode (recordApi fell back to the in-browser mock
 * on a 404) the button says so itself — a user must never fill this form and
 * find out at the end that nothing was written.
 */

export interface OpenAction {
  label: string
  /** the consequence, shown next to the button — never empty */
  hint: string
  /** extra class for the button, '' when this is the real thing */
  cls: string
  aria: string
}

const REAL_HINT =
  'Both arms leave the fleet while recording: their peers are despawned and the ' +
  'ports handed to the recorder, and the follower is energised to hold position. ' +
  'Nothing is written until you start an episode.'

const MOCK_HINT =
  'Rehearsal: this backend has no /api/record, so no arm is touched, no port is ' +
  'taken and no dataset is written. The buttons work so you can learn the flow.'

/**
 * @param mock true = in-browser rehearsal, false = real recorder,
 *             null/undefined = not known yet (claim neither).
 */
export function openActionCopy(mock: boolean | null | undefined): OpenAction {
  if (mock === true) {
    return {
      label: 'open a rehearsal session',
      hint: MOCK_HINT,
      cls: 'rehearsal',
      aria: 'open a rehearsal session — no arm is touched and nothing is written',
    }
  }
  if (mock === false) {
    return {
      label: 'open the arms for recording',
      hint: REAL_HINT,
      cls: '',
      aria: 'open the arms for recording — despawns both peers and energises the follower',
    }
  }
  // Not probed yet: name the action, promise nothing about which recorder runs.
  return {
    label: 'open the arms for recording',
    hint: REAL_HINT,
    cls: '',
    aria: 'open the arms for recording — despawns both peers and energises the follower',
  }
}
