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
  /** offer the devices screen? Only when it can actually help. */
  offerDevices: boolean
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
    return {
      text: 'no arms are on the mesh, and the devices screen could not be reached to say whether any ' +
        'are configured — open it to check',
      offerDevices: true,
    }
  }

  const bringable = remembered.filter(b => b.peer_id && !b.claimed)
  if (bringable.length) {
    const names = bringable.map(b => b.peer_id).join(' and ')
    return {
      // Named, because "you have 2 configured devices" makes the operator go and look at what they
      // already know: these are their arms, by the names they gave them.
      text: `no arms are on the mesh, but the devices screen remembers ${names} — one click there ` +
        'brings a board back up with the config it last ran with',
      offerDevices: true,
    }
  }

  if (remembered.length) {
    // Every remembered board has something running on its bus, yet no arm reached the mesh: a real
    // state (a child that came up and never announced itself), and NOT one a respawn fixes.
    return {
      text: 'no arms are on the mesh, though a process already holds every configured board — check ' +
        'those children\'s logs on the devices screen rather than spawning again',
      offerDevices: true,
    }
  }

  return {
    // Nothing configured at all: the first-run state. The route is still the devices screen, but the
    // work there is different, so the sentence must be too.
    text: 'no arms are on the mesh and no board is configured yet — plug an arm in and spawn it once ' +
      'from the devices screen; it is remembered by its USB serial afterwards',
    offerDevices: true,
  }
}
