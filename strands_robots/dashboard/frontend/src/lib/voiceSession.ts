export type VoiceState = 'idle' | 'connecting' | 'live' | 'error'

/**
 * Reading what the voice session says — as pure functions.
 *
 * These decisions lived inside `ws.onmessage`, wrapped in `catch { /* ignore *\/ }`. That combination is
 * why they deserve a test more than most: any mistake in there is not a stack trace, it is SILENCE in a
 * channel whose entire job is to talk back.
 */

/** A base64 PCM rate the audio API will actually accept. */
const DEFAULT_RATE = 24000

export interface VoiceEffect {
  /** new playback sample rate */
  rate?: number
  /** base64 PCM16 to play */
  play?: string
  /** a line for the transcript */
  transcript?: string
  /** a state change */
  state?: VoiceState
  /** a consent payload for the dock's ConsentSheet */
  need?: unknown
}

function words(value: unknown, fallback: string, cap = 200): string {
  if (value === undefined || value === null || value === '') return fallback
  const text = typeof value === 'string' ? value : (() => {
    try { return JSON.stringify(value) ?? String(value) } catch { return String(value) }
  })()
  return text.slice(0, cap)
}

/**
 * One frame from /ws/voice → what it should change. Unknown frames change nothing, deliberately: a
 * server that grows a new frame type must not break an older bundle.
 */
export function interpretVoiceEvent(ev: any): VoiceEffect {
  if (!ev || typeof ev !== 'object') return {}
  switch (ev.type) {
    case 'voice_meta': {
      // A zero, negative or non-numeric rate makes createBuffer throw NotSupportedError — and that
      // throw happens inside onmessage's `catch { ignore }`, so a single bad meta frame silently kills
      // ALL audio for the session with no error anywhere. The rate has to be usable or ignored.
      const rate = Number(ev.rate)
      return { rate: Number.isFinite(rate) && rate > 0 ? rate : DEFAULT_RATE }
    }
    case 'audio':
      return typeof ev.data === 'string' && ev.data ? { play: ev.data } : {}
    case 'transcript':
      if (!ev.text) return {}
      return { transcript: `${ev.role === 'user' ? '🎙' : '🤖'} ${words(ev.text, '', 4000)}` }
    case 'needs_consent': {
      const effect: VoiceEffect = { need: ev.need ?? null }
      // The sheet only opens for a real payload. A needs_consent frame carrying NOTHING renderable
      // used to vanish completely — setNeed(undefined) is falsy, so no sheet, and with no `spoken`
      // field no transcript either: a voice turn refused in total silence. Say something regardless;
      // the refusal itself is the news, and the grant must always be a tap, never a spoken yes.
      if (ev.spoken) effect.transcript = `⚠ ${words(ev.spoken, '')}`
      else if (!ev.need) effect.transcript = '⚠ that voice turn was refused — open the dock for details'
      return effect
    }
    case 'error':
      return {
        transcript: `⚠ ${words(ev.error, 'the voice session reported an error')}`,
        state: 'error',
      }
    default:
      return {}
  }
}

/**
 * What a closed voice socket means. 1008 is the policy close the server uses for an unauthorized
 * token — worth its own sentence, because "it just stopped listening" sends the operator hunting a
 * microphone problem that is really a settings problem.
 *
 * Any other close leaves an existing `error` standing: the close is the CONSEQUENCE of the error
 * frame that arrived a moment earlier, and overwriting it with a bland "idle" hides the reason.
 */
export function voiceCloseState(code: number | undefined, current: VoiceState): { state: VoiceState; transcript?: string } {
  if (code === 1008) {
    return { state: 'error', transcript: '⚠ unauthorized — set the dashboard token in Settings' }
  }
  return { state: current === 'error' ? current : 'idle' }
}
