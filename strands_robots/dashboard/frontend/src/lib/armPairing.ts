/**
 * Which arm is the leader and which is the follower — decided from what was
 * MEASURED off each servo bus, not from what the peer is called.
 *
 * Why this file exists: RecordPanel used to default the pair with
 * `/leader|arm-2/` for the leader and `/leader|arm-1/` for the follower. On this
 * machine arm-2 measures 12.6V — it IS the follower — so the record screen
 * confidently pre-filled the pair BACKWARDS, and a backwards pair means the
 * operator hand-moves the torqued 12V arm while the 7.4V one is told to mirror
 * it. The names never carried this information; the bus does (12V = follower,
 * 7.4V = leader), and /api/devices now reports the measurement per managed peer.
 *
 * Design rule: a confident wrong default is worse than an empty slot. Anything
 * this module is not sure about it leaves blank and SAYS SO, so the human fills
 * it in knowing the dashboard has no opinion, rather than trusting one that is
 * inverted.
 */

export interface RoleCandidate {
  peer_id: string
  /** 'leader' | 'follower' when measured; absent = nobody has measured it. */
  role?: string | null
  role_volts?: number | null
}

export interface ArmPair {
  leader: string
  follower: string
  /** How each slot was decided — shown to the operator, never inferred by them. */
  basis: 'measured' | 'named' | 'none'
  /** One sentence for the UI when the pair is not fully measured. */
  note?: string
}

const NAMED_LEADER = /(^|[^a-z])leader([^a-z]|$)/i
const NAMED_FOLLOWER = /(^|[^a-z])follower([^a-z]|$)/i

/** Peers whose bus measured as `role`. */
export function measured(candidates: RoleCandidate[], role: string): RoleCandidate[] {
  return candidates.filter(c => c.role === role)
}

/**
 * Pick the pair. Measurement wins; an explicit name is a weak second; an index
 * in a name ("arm-2") is NOT evidence and is never used.
 */
export function pairArms(candidates: RoleCandidate[]): ArmPair {
  const leaders = measured(candidates, 'leader')
  const followers = measured(candidates, 'follower')

  // The happy path: the hardware answered for both slots.
  if (leaders.length === 1 && followers.length === 1) {
    return { leader: leaders[0].peer_id, follower: followers[0].peer_id, basis: 'measured' }
  }

  // Exactly one side is known. Fill THAT slot only: guessing the other from a
  // name is how the pair got inverted in the first place.
  if (leaders.length === 1 && followers.length === 0) {
    return {
      leader: leaders[0].peer_id, follower: '', basis: 'measured',
      note: `${leaders[0].peer_id} measured as the leader; no arm has measured as a 12V follower yet — ` +
            `pick the follower yourself, or measure it on the devices screen`,
    }
  }
  if (followers.length === 1 && leaders.length === 0) {
    return {
      leader: '', follower: followers[0].peer_id, basis: 'measured',
      note: `${followers[0].peer_id} measured as the follower; no arm has measured as a 7.4V leader yet ` +
            `(an unpowered arm reads 5.5V on the USB rail) — pick the leader yourself, or measure it ` +
            `on the devices screen`,
    }
  }

  // Two arms measured the SAME role. That is a real situation (two followers on
  // the bench) and it is not the dashboard's job to break the tie silently.
  if (leaders.length > 1 || followers.length > 1) {
    const which = leaders.length > 1 ? 'leader' : 'follower'
    const n = leaders.length > 1 ? leaders.length : followers.length
    return {
      leader: '', follower: '', basis: 'none',
      note: `${n} arms measured as the ${which}, so which one drives is your call — ` +
            `check the volts next to each name`,
    }
  }

  // Nothing measured. A name that literally says "leader"/"follower" is a stated
  // intent and worth honouring; "arm-1"/"arm-2" is just an index and is ignored.
  const named = {
    leader: candidates.find(c => NAMED_LEADER.test(c.peer_id))?.peer_id ?? '',
    follower: candidates.find(c => NAMED_FOLLOWER.test(c.peer_id))?.peer_id ?? '',
  }
  if (named.leader && named.follower && named.leader !== named.follower) {
    return {
      ...named, basis: 'named',
      note: 'paired from the peer names — nobody has measured these buses, so the names are ' +
            'being taken at their word',
    }
  }

  return {
    leader: '', follower: '', basis: 'none',
    note: candidates.length
      ? 'no arm has been measured, and the names do not say which is which — the leader is the ' +
        'lighter 7.4V arm, or measure both on the devices screen'
      : 'no arms on the mesh',
  }
}

/** The label an option carries, so the volts travel with the name. */
export function roleLabel(c: RoleCandidate): string {
  if (!c.role) return `${c.peer_id} — role not measured`
  const v = typeof c.role_volts === 'number' ? ` · ${c.role_volts}V` : ''
  return `${c.peer_id} — ${c.role}${v}`
}

/**
 * The contradiction check: the operator has chosen a slot that the hardware
 * says is the other thing. Returns null when there is nothing to warn about —
 * an unmeasured arm is never a warning, because we do not know.
 */
export function contradiction(
  candidates: RoleCandidate[], slot: 'leader' | 'follower', chosen: string,
): string | null {
  if (!chosen) return null
  const c = candidates.find(x => x.peer_id === chosen)
  if (!c?.role) return null
  if (c.role === slot) return null
  const v = typeof c.role_volts === 'number' ? `${c.role_volts}V` : 'its bus'
  if (c.role === 'unpowered') {
    return `${chosen} read ${v} — that is the USB logic rail, so its power supply is off. ` +
           `It can report positions but it cannot hold or mirror anything.`
  }
  if (c.role === 'mixed') {
    return `${chosen} reported inconsistent voltages across its servos — that is a wiring fault, ` +
           `not a role. Fix it before recording an episode with it.`
  }
  return `${chosen} measured ${v} — it is the ${c.role}, not the ${slot}. ` +
         (slot === 'leader'
           ? 'You would be hand-moving a torqued 12V arm while the 7.4V one tries to mirror it.'
           : 'The arm being recorded should be the 12V one that mirrors your hand.')
}
