/**
 * Should the dashboard tell the operator that the SERVER is refusing somebody? (Q88)
 *
 * The server now counts refused handshakes and says so in /api/health (refusals.py). That was
 * written for the incident where a phone's camera tiles were refused for 19.3 hours: correct
 * refusals, a cheerful `status: ok`, and the only record in a 34 MB log. But nothing in this
 * frontend reads /api/health at all — so the news existed only for whoever thought to curl it,
 * which is not a surface, it is a hiding place.
 *
 * This file is the judgement, kept out of the component so it can be argued with:
 *
 *   - A LOOP is worth interrupting for. Ten refusals from one client in five minutes is a
 *     machine, and it will not fix itself: some tab or script is holding a credential this
 *     server will not accept, and it hammers until a human reloads it.
 *   - A HANDFUL is not. Someone signing in, a script being fixed, a page reloading — showing a
 *     banner for that teaches the operator to ignore banners, and this dashboard's banners
 *     guard an e-stop.
 *   - "It stopped" is not news either. The server reports it (so "did my fix work?" has an
 *     answer in the payload) but the screen stays quiet: a resolved problem announcing itself
 *     is the same nag with better manners.
 *   - It is never about THIS page. The reader is signed in — their own session works, by
 *     construction — so the sentence must not read as "you are locked out". Q88's own banner
 *     (linkHealth) covers that case, and the two must not be confused.
 */

export interface RefusedHandshakes {
  total?: number
  recent?: number
  window_s?: number
  clients?: number
  storm?: boolean
  untracked?: number
  worst?: { client?: string; path?: string; kind?: string; count?: number }
  text?: string
}

export interface ServerNotice {
  /** what to show — '' means say nothing at all */
  text: string
  /** true when the identities were withheld (an unauthenticated /api/health read) */
  vague: boolean
}

const NOTHING: ServerNotice = { text: '', vague: false }

/**
 * @param block  the `refused_handshakes` object from /api/health, if present
 */
export function serverNotice(block: RefusedHandshakes | null | undefined): ServerNotice {
  if (!block || block.storm !== true) return NOTHING
  const recent = block.recent ?? 0
  if (recent <= 0) return NOTHING // a storm flag with nothing recent is stale; trust the count
  const worst = block.worst
  if (!worst?.client) {
    // The server withheld identities (that read was not authenticated — it happens when the
    // token is in a query string the fetch did not carry). Say the true, smaller thing rather
    // than nothing: an operator who knows a client is looping can find it.
    return {
      text:
        `Something is being refused by this server — ${recent} handshake(s) in the last ` +
        `${Math.round((block.window_s ?? 300) / 60)} minutes. Your own session is fine.`,
      vague: true,
    }
  }
  // The server's own sentence already names client, path, rate and the one fix, and it is
  // written to be read by a human (refusals.py). Repeating that judgement here would be two
  // sources of truth for one number; what this layer adds is the part the server cannot know —
  // that the person reading it is signed in and NOT the one being refused.
  const suffix = 'Your own session is fine — this is another client.'
  return { text: `${block.text ?? `${worst.client} is being refused repeatedly.`} ${suffix}`, vague: false }
}
