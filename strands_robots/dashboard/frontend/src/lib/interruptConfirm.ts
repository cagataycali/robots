/**
 * The agent's motion confirm, rendered from the interrupt's own words.
 *
 * The server pauses a turn ({type:'interrupt'} on /ws/chat) instead of
 * refusing it; these rules turn that event into a question a human can
 * answer, and the answer into the frame the server expects back.
 */

export interface MotionConfirm {
  id: string
  target: string
  instruction: string
  duration: number | null
  whyPhysical: string
}

/** One {type:'interrupt'} event -> a renderable confirm, or null if malformed. */
export function parseInterruptEvent(ev: any): MotionConfirm | null {
  if (!ev || ev.type !== 'interrupt' || typeof ev.id !== 'string' || !ev.id.trim()) return null
  const r = ev.reason && typeof ev.reason === 'object' ? ev.reason : {}
  const dur = typeof r.duration === 'number' && isFinite(r.duration) && r.duration > 0 ? r.duration : null
  return {
    id: ev.id.trim(),
    target: typeof r.target === 'string' && r.target.trim() ? r.target.trim() : 'a robot',
    instruction: typeof r.instruction === 'string' ? r.instruction.trim() : '',
    duration: dur,
    whyPhysical: typeof r.why_physical === 'string' ? r.why_physical.trim() : '',
  }
}

/** agent_status.interrupt (a reload found a confirm parked) -> the same shape. */
export function parseStatusInterrupt(interrupt: any): MotionConfirm | null {
  if (!interrupt || typeof interrupt !== 'object') return null
  return parseInterruptEvent({ type: 'interrupt', id: interrupt.id, reason: interrupt.reason })
}

function fmtDuration(s: number): string {
  if (s >= 60) {
    const m = Math.floor(s / 60)
    const rest = Math.round(s % 60)
    return rest ? `${m}m ${rest}s` : `${m}m`
  }
  return `${Math.round(s * 10) / 10}s`
}

/** The question, in the reason's own words: which arm, what instruction, how long. */
export function confirmQuestion(c: MotionConfirm): string {
  const doing = c.instruction ? `run "${c.instruction}"` : 'start a task'
  const span = c.duration != null ? ` for ${fmtDuration(c.duration)}` : ''
  return `The agent wants ${c.target} to ${doing}${span} — real motion.`
}

/** Why this counts as real hardware — shown small, never invented. */
export function confirmDetail(c: MotionConfirm): string {
  return c.whyPhysical ? `${c.target}: ${c.whyPhysical}.` : ''
}

/** The exact answer frame /ws/chat expects. Anything but an explicit yes is a no. */
export function interruptResponseBody(id: string, approve: boolean): {
  type: 'interrupt_response'; id: string; response: { approve: boolean }
} {
  return { type: 'interrupt_response', id, response: { approve } }
}

/** What the transcript records for an answer, so the decision is auditable in place. */
export function answerNotice(c: MotionConfirm, approve: boolean): string {
  return approve
    ? `✓ approved — ${c.target} may ${c.instruction ? `run "${c.instruction}"` : 'start the task'} this once`
    : `✗ declined — nothing was sent to ${c.target}`
}
