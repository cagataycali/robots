/**
 * U22 slice 3a: WHO could lead this arm, and what must be true before frames flow.
 *
 * Starting teleop moves a real arm, so this rule's whole job is to say what is known and refuse what is
 * not — before anything is sent. It exists in the shape it does because of two measured events:
 *
 *  1. A real SO-101 leader published 176 frames and the follower applied NONE of them: the mesh's
 *     per-frame envelope is 4*pi, a RADIAN assumption, and an SO-101 reports DEGREES (wrist_roll sits at
 *     170). Every frame from a real arm is out of range. That is not a fault to discover in a child log
 *     afterwards — it is a CONSENT to collect beforehand (teleop_degree_units), which is why `consents`
 *     is part of the plan rather than an error path.
 *  2. "A process is not an arm" (armHosts): peers are named parent / parent__child, and on this fleet the
 *     simulator PROCESS reports 0 joints while the robot under it reports 6. The friendlier name is the
 *     wrong one to command.
 *
 * Nothing here starts, stops or widens anything. It produces sentences and a list of preconditions.
 */
import { armHosts, isChildOf, type HostInput } from './armHosts'

export interface PairInput extends HostInput {
  role?: string | null
  role_volts?: number | null
  role_source?: string | null
}

export interface LeaderOption {
  peer_id: string
  /** may this be offered as the leader? */
  ok: boolean
  /** why not — or, when ok, what the operator should know about it */
  why?: string
}

/** Consent kinds the dashboard already knows how to ask for and revoke. */
export type TeleopConsent = 'agent_physical_motion' | 'teleop_degree_units'

export interface PairPlan {
  leader: string
  follower: string
  /** hard preconditions: while any of these stand, no frames may be sent */
  blockers: string[]
  /** grants to collect BEFORE starting, each with the reason the operator needs to judge it */
  consents: TeleopConsent[]
  /** true but not disqualifying — evidence the operator should read */
  notes: string[]
}

const jointCount = (p: PairInput | undefined): number =>
  typeof p?.joints === 'number' && p.joints > 0 ? p.joints : 0

/** Which peers may be offered as the leader for `followerId`, each with its sentence. */
export function leaderOptions(followerId: string, peers: PairInput[] | null | undefined): LeaderOption[] {
  const list = (peers ?? []).filter(p => p && p.peer_id)
  const hosts = armHosts(list)
  const out: LeaderOption[] = []
  for (const p of list) {
    if (p.peer_id === followerId) continue                    // an arm cannot follow itself; not an option at all
    const host = hosts[p.peer_id]
    if (host) { out.push({ peer_id: p.peer_id, ok: false, why: host.why }); continue }
    if (!jointCount(p)) {
      // The current state of BOTH real arms on this fleet: connected, camera flowing, no joints. A leader
      // that cannot READ its own position cannot publish one, and the remedy is in its own log.
      out.push({ peer_id: p.peer_id, ok: false,
        why: 'reports no joints — it cannot publish a position it cannot read; its log (devices › logs) says why' })
      continue
    }
    if (isChildOf(p.peer_id, followerId) || isChildOf(followerId, p.peer_id)) {
      out.push({ peer_id: p.peer_id, ok: false, why: 'the same robot as the follower, under its process name' })
      continue
    }
    out.push({ peer_id: p.peer_id, ok: true,
      why: p.role === 'leader' && p.role_source === 'measured'
        ? `measured as a leader (${p.role_volts ?? '?'}V)`
        : p.role === 'follower' && p.role_source === 'measured'
          ? `measured as a FOLLOWER (${p.role_volts ?? '?'}V) — check this is the arm you intend to hand-guide`
          : 'role not measured — the dashboard cannot tell which arm this is wired as' })
  }
  return out
}

/** Everything that must be true, granted or understood before this pair may stream. */
export function pairPlan(followerId: string, leaderId: string, peers: PairInput[] | null | undefined): PairPlan | null {
  if (!followerId || !leaderId || followerId === leaderId) return null
  const list = (peers ?? []).filter(p => p && p.peer_id)
  const follower = list.find(p => p.peer_id === followerId)
  const leader = list.find(p => p.peer_id === leaderId)
  const blockers: string[] = []
  const notes: string[] = []

  if (!follower) blockers.push(`${followerId} is not on the mesh — the dashboard cannot command an arm it cannot see`)
  if (!leader) blockers.push(`${leaderId} is not on the mesh`)
  if (follower && !jointCount(follower))
    blockers.push(`${followerId} reports no joints, so nothing could be applied to it (its log says why)`)
  if (leader && !jointCount(leader))
    blockers.push(`${leaderId} reports no joints, so it has no position to publish (its log says why)`)
  const hosts = armHosts(list)
  for (const id of [followerId, leaderId]) if (hosts[id]) blockers.push(`${id} ${hosts[id].why}`)

  const fj = jointCount(follower), lj = jointCount(leader)
  if (fj && lj && fj !== lj)
    // Not a refusal: the mesh maps by joint NAME. But a shape mismatch is the operator's business.
    notes.push(`${leaderId} reports ${lj} joints and ${followerId} reports ${fj} — only the names they share can be followed`)
  if (follower?.role === 'leader' && follower.role_source === 'measured')
    notes.push(`${followerId} measures as a LEADER (${follower.role_volts ?? '?'}V) — it is about to be DRIVEN, so make sure that is the arm you want moving`)

  return {
    leader: leaderId, follower: followerId, blockers, notes,
    // Both grants are asked for BEFORE anything is sent. agent_physical_motion because frames move a real
    // arm; teleop_degree_units because an SO-101 publishes degrees into a radian envelope and every frame
    // would otherwise be refused — the failure that took a log dive to find the first time.
    consents: ['agent_physical_motion', 'teleop_degree_units'],
  }
}

/** One line for a screen: is this pair startable, and if not, what stands in the way. */
export function pairSentence(plan: PairPlan | null): string | null {
  if (!plan) return null
  if (plan.blockers.length) return `cannot start: ${plan.blockers.join('; ')}`
  const tail = plan.notes.length ? ` · ${plan.notes.join(' · ')}` : ''
  return `${plan.follower} could follow ${plan.leader}${tail}`
}
