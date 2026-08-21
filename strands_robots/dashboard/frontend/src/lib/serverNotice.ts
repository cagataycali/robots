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

/* ---------------------------------------------------------------------------------------------
 * WHICH BUILD IS ANSWERING (ec5aabb4) — and the discipline that keeps it from becoming a nag.
 *
 * /api/health now always carries a `build` stamp, so a response WITHOUT one is, by construction,
 * a server older than that commit. On this fleet that is not hypothetical: the live server has
 * been up since Aug 19 and returns no `build` at all, while the bundle it serves has since grown
 * fields it renders nothing for (peer `origin`, remembered device profiles, …).
 *
 * The temptation is to announce staleness on sight. This file's own doctrine forbids it: a server
 * being older than HEAD is the NORMAL state of any long-running process, and a banner every
 * operator sees every day is how banners stop being read — and here they guard an e-stop.
 *
 * So the rule is EVIDENCE ABOVE STRUCTURE, the same shape as U15's origin badge:
 *   - staleness ALONE is not news;
 *   - a field the page wanted and did not get is not news either (it could be a robot that never
 *     reported it);
 *   - the two TOGETHER are, because then the operator is being shown less than this bundle can
 *     show and there is exactly one remedy.
 *
 * And the inverse matters just as much: when the server DOES stamp itself, a missing field is not
 * staleness, so this stays silent rather than sending someone to restart a current server — the
 * wrong-remedy failure that `read_commit` returning None was written to avoid on the other side.
 */

export interface ServerBuild {
  commit?: string | null
  version?: string | null
  started?: number | null
}

/** Does this /api/health response come from a server older than the build stamp?
 *
 * Only a real health payload can answer. `undefined` (not fetched yet), a non-object, or an empty
 * object are NO EVIDENCE — treating "I have not asked yet" as "your server is old" would put the
 * notice on screen during every page load, which is worse than never showing it.
 */
export function serverPredatesBuildStamp(health: unknown): boolean {
  if (!health || typeof health !== 'object' || Array.isArray(health)) return false
  const h = health as Record<string, unknown>
  if (typeof h.status !== 'string' && typeof h.t !== 'number') return false // not a health payload
  return !('build' in h)
}

/** Fields this bundle renders that the whole fleet is silent about, in the operator's words. */
const FIELD_GAPS: { field: string; says: string }[] = [
  // U15: `origin` tells managed peers from external ones. Absent on EVERY peer means the server
  // predates it, not that the fleet is unusual — a peer either gets annotated or none do, because
  // MeshBridge.snapshot() stamps them all from one live-managed set.
  { field: 'origin', says: 'which robots this dashboard started itself' },
  // NOTHING ELSE BELONGS HERE YET, and the two rejected candidates are the rule:
  //   * `remembered` (Q41) is a field of a DEVICE ROW, not of a peer — peers have never carried it
  //     in any version, so peer silence about it is not evidence of anything and the sentence
  //     would promise a restart something a restart cannot change. Caught by measuring the live
  //     payload rather than by reading my own table.
  //   * a NEWS-ONLY field (joint_problem, refused_handshakes: present only when something is
  //     wrong) can never qualify either — its absence is the healthy case, so "absent on every
  //     peer" would accuse a current server of being old.
  // The test for a new entry: would EVERY peer on a current server carry it? If not, it is not
  // evidence.
]

export function fleetFieldGaps(peers: unknown): string[] {
  const list = Array.isArray(peers)
    ? peers
    : peers && typeof peers === 'object'
      ? Object.values(peers as Record<string, unknown>)
      : []
  const rows = list.filter((p): p is Record<string, unknown> => !!p && typeof p === 'object')
  if (rows.length === 0) return [] // an empty fleet cannot be evidence of anything
  return FIELD_GAPS.filter(g => rows.every(r => !(g.field in r))).map(g => g.says)
}

/**
 * @param health  the whole /api/health payload (not a sub-block — absence is the signal)
 * @param gaps    what this page wanted to render and the entire fleet was silent about
 */
export function staleServerNotice(health: unknown, gaps: string[]): ServerNotice {
  if (!serverPredatesBuildStamp(health)) return NOTHING
  if (gaps.length === 0) return NOTHING
  const list = gaps.length === 1 ? gaps[0] : `${gaps.slice(0, -1).join(', ')} and ${gaps[gaps.length - 1]}`
  return {
    text:
      `This page can show ${list}, but the server answering it is older than the code you are ` +
      `looking at, so it never sends them. Restart the dashboard from a terminal to pick them up ` +
      `— a restart is also the only way macOS will grant it camera access.`,
    vague: false,
  }
}
