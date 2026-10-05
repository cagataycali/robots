/** R2 / UX_REVIEW:61 — the button that SPAWNS A PROCESS was labelled `⿻`. */
import { missingExtra, type MissingExtra } from './missingExtra'

export interface TwinButtonCopy {
  label: string
  title: string
  aria: string
  /** extra class: 'on' while a twin is live, '' otherwise */
  cls: string
  /** aria-pressed for the toggle. */
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

/** Why a twin did not come up, read off the twin route's answer, or `null` when it did.
 *
 * The route answers like a Devices spawn: a 412 whose `error` dict names the extra this
 * environment lacks (the preflight), or a 200 whose `error` says why the child died inside
 * the settle window. A pid alone is not a twin, so either shape becomes a sentence on the
 * card and, when an extra is named, the install button.
 */
export function twinFailure(peerId: string, body: unknown): { text: string; gap: MissingExtra | null } | null {
  const top = body && typeof body === 'object' ? (body as Record<string, any>) : null
  if (!top) return null
  const said = typeof top.error === 'string' ? top.error
    : typeof top.error?.error === 'string' ? top.error.error
    : typeof top.detail?.error === 'string' ? top.detail.error
    : null
  if (!said) return null
  return { text: `${peerId}-twin did not start: ${said}`, gap: missingExtra(top) }
}
