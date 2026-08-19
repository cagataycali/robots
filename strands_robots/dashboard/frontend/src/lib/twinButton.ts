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
}

export function twinButtonCopy(o: { peerId: string; twinLive: boolean; busy?: boolean }): TwinButtonCopy {
  const twinId = `${o.peerId}-twin`
  if (o.busy) {
    return {
      label: '…',
      cls: o.twinLive ? 'on' : '',
      title: `waiting for ${twinId} — a sim peer takes a moment to start or stop`,
      aria: `sim twin of ${o.peerId}: working`,
    }
  }
  if (o.twinLive) {
    return {
      label: 'twin on',
      cls: 'on',
      title: `${twinId} is running: tasks sent to this robot are mirrored to it. `
        + `Click to stop the twin — the real arm is not affected either way.`,
      aria: `stop the sim twin of ${o.peerId}`,
    }
  }
  return {
    label: '+ twin',
    cls: '',
    title: `Start ${twinId}, a simulated copy of this arm as its own peer. `
      + `Tasks you send this robot are mirrored to it, so you can watch a policy `
      + `in sim before trusting it on metal. The real arm is not touched.`,
    aria: `start a sim twin of ${o.peerId}`,
  }
}
