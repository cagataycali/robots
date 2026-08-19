/**
 * UX_REVIEW #10: "flag rehearsal features in the nav, not after the click".
 *
 * `⏺ record` looks exactly as real as every other button until the sheet opens
 * and admits that this backend has no `/api/record` and nothing will be written
 * to disk. An operator who is about to spend twenty minutes teleoperating an arm
 * deserves to know that BEFORE the first episode, not after.
 *
 * The rule this module encodes is the same one the fleet's origin badge follows:
 * SILENCE IS NOT EVIDENCE. Until the probe has actually answered, the nav says
 * nothing — a chip that guessed "rehearsal" from a slow probe would train
 * operators to ignore the word on the day it is true, and a chip that guessed
 * "real" is worse.
 */

export interface NavFlag {
  /** true only when the backend is KNOWN to be a rehearsal */
  flagged: boolean
  /** appended to the chip's visible label, '' when not flagged */
  suffix: string
  /** extra class for the chip, '' when not flagged */
  cls: string
  /** hover text — always says what the button does; adds the warning when flagged */
  title: string
  /** accessible name, so the warning is not colour- or glyph-only */
  aria: string
}

/**
 * @param mock  true = probe selected the in-browser rehearsal,
 *              false = the real backend answered,
 *              null/undefined = not probed yet (say nothing).
 */
export function recordNavFlag(mock: boolean | null | undefined, base = 'Record teleop episodes into a dataset'): NavFlag {
  if (mock !== true) {
    return { flagged: false, suffix: '', cls: '', title: base, aria: 'record' }
  }
  return {
    flagged: true,
    suffix: ' · rehearsal',
    cls: 'rehearsal',
    title: `${base}. REHEARSAL: this backend has no /api/record, so the buttons work but nothing is written to disk and no dataset is produced.`,
    aria: 'record — rehearsal only, nothing is written to disk',
  }
}
