/**
 * What the agent dock is allowed to claim about a message and an answer.
 *
 * The dock had two silences. (1) `send()` cleared the input, appended the user's
 * bubble and *then* tried to open the socket — so a message that never left the
 * browser rendered exactly like a delivered one, and the typed text was gone.
 * (2) When the socket closed mid-turn the only reported case was 1008; any other
 * drop reset `busy` and said nothing, leaving a HALF-STREAMED answer on screen
 * looking like a finished one.
 *
 * (2) is the dangerous half here, because this agent starts and stops real
 * robots. A turn cut off after `⚙ fleet_stop` began is not "no answer": the
 * command may have run. So the verdicts below separate three things a chat UI
 * usually blurs:
 *
 *   - never delivered  → safe to retry, and the text must come back
 *   - delivered, outcome unknown → retrying may run it TWICE
 *   - delivered, answer truncated → the text on screen is not the whole reply
 */

export interface SendFailure {
  error?: string | null
}

export interface SendVerdict {
  /** The notice to show. */
  text: string
  /** True when the message provably never left the browser. */
  retrySafe: boolean
}

/**
 * The socket never opened, so nothing was sent: say so plainly, and promise the
 * text is not lost (the caller restores it).
 */
export function sendFailureVerdict(f: SendFailure): SendVerdict {
  const why = String(f.error ?? '').trim() || 'the agent socket could not be opened'
  return {
    text: `⚠ not sent: ${why}. Nothing reached the agent, so nothing ran — your message is back in the box, press ↑ to try again.`,
    retrySafe: true,
  }
}

export interface Interruption {
  /** WebSocket close code, when the close event provided one. */
  code?: number | null
  /** Was a turn in flight? An idle socket closing is not an event. */
  wasBusy: boolean
  /** How much answer text had already streamed in. */
  partialChars?: number
  /** Tools the agent had STARTED and never reported finishing. */
  runningTools?: string[]
}

export interface InterruptionNotice {
  text: string
  tone: 'bad' | 'warn'
  /** True when re-sending the same message could execute it a second time. */
  doubleRunRisk: boolean
}

const AUTH_TEXT = 'the server rejected this token — set it in Settings'

/**
 * The verdict for a socket that closed. Returns null when there is nothing
 * honest to report — a dock that cries wolf on every idle reconnect trains the
 * operator to ignore the line that matters.
 */
export function interruptionNotice(i: Interruption): InterruptionNotice | null {
  const auth = i.code === 1008
  if (!i.wasBusy) {
    // An idle drop costs nothing (the next send reopens) EXCEPT when it was a
    // refusal: that one will happen again and needs fixing, not retrying.
    return auth ? { text: `⚠ ${AUTH_TEXT}`, tone: 'bad', doubleRunRisk: false } : null
  }

  const tools = (i.runningTools ?? []).filter(t => String(t ?? '').trim())
  const partial = Math.max(0, Number(i.partialChars ?? 0) || 0)
  const head = auth
    ? `⚠ the turn was cut off: ${AUTH_TEXT}.`
    : '⚠ the connection dropped before the answer finished.'

  const parts: string[] = [head]
  if (tools.length) {
    const named = tools.slice(0, 3).join(', ')
    const more = tools.length > 3 ? ` +${tools.length - 3} more` : ''
    parts.push(
      `${tools.length === 1 ? 'The tool' : 'The tools'} ${named}${more} had already started, so the agent may have ACTED on the fleet — check Activity before assuming nothing happened.`,
    )
  }
  parts.push(
    partial > 0
      ? 'The text above stops mid-answer; it is not a complete reply.'
      : 'Nothing came back, but your message had already been sent — re-sending could run it a second time.',
  )
  if (partial > 0 || tools.length) {
    parts.push('Re-sending repeats the request.')
  }

  return {
    text: parts.join(' '),
    tone: 'bad',
    // Delivered-then-dropped is exactly the case where a blind retry is unsafe.
    doubleRunRisk: true,
  }
}

/** How a user bubble should read. Undelivered must never look delivered. */
export function bubbleLabel(delivered: boolean | undefined): string | null {
  return delivered === false ? 'not sent' : null
}
