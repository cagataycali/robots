/**
 * What the record screen says when there is nothing to record WITH (Q44).
 *
 * `pairArms([])` answers "no arms on the mesh" — true, and a dead end. The commonest way to arrive
 * here is a dashboard restart: the arms are unplugged from nothing, they are simply not running, and
 * the devices screen already knows both of them by USB serial with a one-click respawn (Q41). A
 * screen that states the absence without naming that route asks the operator to go hunting for a
 * feature we shipped.
 *
 * It stays a SENTENCE plus an optional route, never a redirect: an operator who opened the record
 * screen deliberately should not be thrown onto another one.
 */

export interface RememberedBoard {
  /** the peer the board would come back as */
  peer_id: string
  /** true when something already runs on that bus — then it is not a route out of "no arms" */
  claimed?: boolean
}

export interface NoArmsVerdict {
  /** the whole message, ready to render */
  text: string
  /**
   * Just the actionable half — no "no arms are on the mesh" prefix. The home screen's heading
   * ALREADY says the fleet is empty, and a screen that says it twice reads like a stutter and gets
   * skimmed. Same words as `text` otherwise: two screens must not develop two vocabularies for one
   * fact (Q45).
   */
  route: string
  /** offer the devices screen? Only when it can actually help. */
  offerDevices: boolean
}

const ABSENCE = 'no arms are on the mesh'

function verdict(joiner: string, route: string): NoArmsVerdict {
  return { text: `${ABSENCE}${joiner}${route}`, route, offerDevices: true }
}

export function noArmsVerdict(
  peerCount: number,
  remembered: RememberedBoard[] | null | undefined,
): NoArmsVerdict | null {
  // Arms present: this screen has nothing to explain, and a banner would be noise.
  if (peerCount > 0) return null

  // null = the devices request failed. Saying "no boards are remembered" then would be a claim we
  // did not make; absence of evidence is not evidence of absence.
  if (remembered == null) {
    return verdict(', and ',
      'the devices screen could not be reached to say whether any are configured — open it to check')
  }

  const bringable = remembered.filter(b => b.peer_id && !b.claimed)
  if (bringable.length) {
    const names = bringable.map(b => b.peer_id).join(' and ')
    // Named, because "you have 2 configured devices" makes the operator go and look at what they
    // already know: these are their arms, by the names they gave them.
    return verdict(', but ', `the devices screen remembers ${names} — one click there brings a board ` +
      'back up with the config it last ran with')
  }

  if (remembered.length) {
    // Every remembered board has something running on its bus, yet no arm reached the mesh: a real
    // state (a child that came up and never announced itself), and NOT one a respawn fixes.
    return verdict(', though ', 'a process already holds every configured board — check those ' +
      "children's logs on the devices screen rather than spawning again")
  }

  // Nothing configured at all: the first-run state. The route is still the devices screen, but the
  // work there is different, so the sentence must be too.
  return verdict(' and ', 'no board is configured yet — plug an arm in and spawn it once from the ' +
    'devices screen; it is remembered by its USB serial afterwards')
}
