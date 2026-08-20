/**
 * R2 / UX_REVIEW:61 — the button that SPAWNS A PROCESS was labelled `⿻`.
 *
 * A CJK ideograph plus a `title=` tooltip is invisible to a sighted user on
 * touch (no hover exists) and unreadable to everyone who does not already know
 * the glyph, yet the action behind it starts or kills a MuJoCo sim peer of a
 * real arm (`POST /api/robots/<id>/twin`, server.py:1426). So it gets words.
 *
 * The wording also has to be honest about the one thing an operator standing
 * next to two powered arms will worry about: a twin is a SEPARATE simulated
 * peer, and neither spawning nor stopping it touches the metal.
 */

export interface TwinButtonCopy {
  label: string
  title: string
  aria: string
  /** extra class: 'on' while a twin is live, '' otherwise */
  cls: string
  /**
   * aria-pressed for the toggle. The `on` CLASS made the live state visible to a sighted user only —
   * a screen reader announced the same "button" whether a MuJoCo sim peer was running or not, and
   * the label alone is ambiguous ("twin on" reads as an instruction as easily as a state). Derived
   * from twinLive here rather than from `cls` in the markup, so the pressed state and the styling
   * cannot drift apart, and it stays TRUE while busy: the twin is live, we are merely waiting.
   */
  pressed: boolean
}

export function twinButtonCopy(o: { peerId: string; twinLive: boolean; busy?: boolean }): TwinButtonCopy {
  const twinId = `${o.peerId}-twin`
  if (o.busy) {
    return {
      label: '…',
      cls: o.twinLive ? 'on' : '',
      pressed: !!o.twinLive,
      title: `waiting for ${twinId} — a sim peer takes a moment to start or stop`,
      aria: `sim twin of ${o.peerId}: working`,
    }
  }
  if (o.twinLive) {
    return {
      label: 'twin on',
      cls: 'on',
      pressed: true,
      title: `${twinId} is running: tasks sent to this robot are mirrored to it. `
        + `Click to stop the twin — the real arm is not affected either way.`,
      aria: `stop the sim twin of ${o.peerId}`,
    }
  }
  return {
    label: '+ twin',
    cls: '',
    pressed: false,
    title: `Start ${twinId}, a simulated copy of this arm as its own peer. `
      + `Tasks you send this robot are mirrored to it, so you can watch a policy `
      + `in sim before trusting it on metal. The real arm is not touched.`,
    aria: `start a sim twin of ${o.peerId}`,
  }
}
