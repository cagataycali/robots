/**
 * The agent's motion confirm, rendered from the interrupt's own words.
 *
 * The server pauses a turn ({type:'interrupt'} on /ws/chat) instead of
 * refusing it; these rules turn that event into a question a human can
 * answer, and the answer into the frame the server expects back.
 *
 * Two gates raise these interrupts and their reason dicts differ:
 *  - the dashboard's own MotionInterruptHook (fleet task):
 *      {tool, action, target, instruction, duration, why_physical}
 *  - the SDK's robot_mesh tool (tell/send/stop/broadcast/emergency_stop/rpc):
 *      {action, target, function, command, instruction, warning}
 * Both are rendered; the ANSWER is a plain "y"/"n" string because that is
 * the one shape BOTH approval checks accept (robot_mesh refuses non-strings).
 */

export interface MotionConfirm {
  id: string
  target: string
  /** true when the reason names the whole fleet (*ALL_PEERS*). */
  fleetWide: boolean
  action: string
  instruction: string
  /** robot_mesh rpc: the device-native function being invoked. */
  func: string
  /** the validated command body, already stringified for display. */
  command: string
  duration: number | null
  whyPhysical: string
  /** the gate's own scope warning, verbatim. */
  warning: string
}

const ALL_PEERS = '*ALL_PEERS*'

function str(v: any): string {
  return typeof v === 'string' ? v.trim() : ''
}

/** One {type:'interrupt'} event -> a renderable confirm, or null if malformed. */
export function parseInterruptEvent(ev: any): MotionConfirm | null {
  if (!ev || ev.type !== 'interrupt' || typeof ev.id !== 'string' || !ev.id.trim()) return null
  const r = ev.reason && typeof ev.reason === 'object' ? ev.reason : {}
  const dur = typeof r.duration === 'number' && isFinite(r.duration) && r.duration > 0 ? r.duration : null
  const rawTarget = str(r.target)
  let command = ''
  if (r.command != null && r.command !== '') {
    try { command = typeof r.command === 'string' ? r.command : JSON.stringify(r.command) } catch { command = '' }
  }
  return {
    id: ev.id.trim(),
    target: rawTarget === ALL_PEERS ? 'every robot on the mesh' : (rawTarget || 'a robot'),
    fleetWide: rawTarget === ALL_PEERS,
    action: str(r.action),
    instruction: str(r.instruction),
    func: str(r.function),
    command,
    duration: dur,
    whyPhysical: str(r.why_physical),
    warning: str(r.warning),
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

/** What the agent is asking for, most specific evidence first. */
function deed(c: MotionConfirm): string {
  if (c.instruction) return `run "${c.instruction}"`
  if (c.func) return `invoke ${c.func}()`
  if (c.command) return `receive ${c.command}`
  if (c.action === 'stop' || c.action === 'emergency_stop') return 'stop'
  if (c.action) return `do "${c.action}"`
  return 'start a task'
}

/** The question, in the reason's own words: which arm, what instruction, how long. */
export function confirmQuestion(c: MotionConfirm): string {
  const span = c.duration != null ? ` for ${fmtDuration(c.duration)}` : ''
  const stopping = c.action === 'stop' || c.action === 'emergency_stop'
  // A stop is still worth confirming when the gate says so, but it must not
  // be dressed up as motion - the honest phrase is the opposite.
  const tail = stopping ? '' : ' — real motion'
  return `The agent wants ${c.target} to ${deed(c)}${span}${tail}.`
}

/** Why this counts as real hardware / the gate's scope warning — never invented. */
export function confirmDetail(c: MotionConfirm): string {
  if (c.whyPhysical) return `${c.target}: ${c.whyPhysical}.`
  if (c.warning) return c.warning
  return ''
}

/**
 * The exact answer frame /ws/chat expects. The response is the literal
 * string "y" / "n": the dashboard hook accepts it, and robot_mesh's
 * _interrupt_approves accepts ONLY canonical affirmative strings.
 */
export function interruptResponseBody(id: string, approve: boolean): {
  type: 'interrupt_response'; id: string; response: string
} {
  return { type: 'interrupt_response', id, response: approve ? 'y' : 'n' }
}

/** What the transcript records for an answer, so the decision is auditable in place. */
export function answerNotice(c: MotionConfirm, approve: boolean): string {
  return approve
    ? `✓ approved — ${c.target} may ${deed(c)} this once`
    : `✗ declined — nothing was sent to ${c.target}`
}
