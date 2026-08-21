// Run: node scripts/run-lib-tests.mjs voiceSession
//
// These decisions lived inside ws.onmessage, wrapped in `catch { ignore }`. Any mistake in there is not a
// stack trace — it is SILENCE, in the one channel whose entire job is to talk back.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const { interpretVoiceEvent, voiceCloseState } = await import('/tmp/voiceSession.mjs')

// ── 1. frames that change nothing ──
for (const junk of [null, undefined, 'hello', 42, {}, { type: 'something_new_in_v9' }]) {
  assert.deepEqual(interpretVoiceEvent(junk), {},
                   `${JSON.stringify(junk)} changes nothing — an older bundle must survive a new frame type`)
}
assert.deepEqual(interpretVoiceEvent({ type: 'transcript', text: '' }), {}, 'an empty transcript is not a line')
assert.deepEqual(interpretVoiceEvent({ type: 'audio' }), {}, 'an audio frame with no data plays nothing')
assert.deepEqual(interpretVoiceEvent({ type: 'audio', data: '' }), {}, 'nor an empty one')

// ── 2. a bad sample rate must not silently kill ALL audio ──
// createBuffer(1, n, 0) throws NotSupportedError, and that throw lands in onmessage's `catch { ignore }`:
// one malformed meta frame and the session never speaks again, with no error visible anywhere.
assert.deepEqual(interpretVoiceEvent({ type: 'voice_meta', rate: 16000 }), { rate: 16000 })
for (const bad of [0, -1, 'fast', null, undefined, NaN, Infinity]) {
  assert.equal(interpretVoiceEvent({ type: 'voice_meta', rate: bad }).rate, 24000,
               `rate ${String(bad)} is unusable, so the default stands instead of poisoning createBuffer`)
}
assert.equal(interpretVoiceEvent({ type: 'voice_meta', rate: '48000' }).rate, 48000, 'a numeric string is fine')

// ── 3. transcript and audio ──
assert.equal(interpretVoiceEvent({ type: 'transcript', role: 'user', text: 'move the arm' }).transcript,
             '🎙 move the arm', 'who said it is part of the record')
assert.equal(interpretVoiceEvent({ type: 'transcript', role: 'assistant', text: 'refusing' }).transcript,
             '🤖 refusing')
assert.equal(interpretVoiceEvent({ type: 'audio', data: 'AAAB' }).play, 'AAAB')

// ── 4. a refusal is never silent ──
const withNeed = interpretVoiceEvent({ type: 'needs_consent', need: { kind: 'env', keys: ['HF_TOKEN'] } })
assert.deepEqual(withNeed.need, { kind: 'env', keys: ['HF_TOKEN'] }, 'the payload the ConsentSheet needs')
assert.equal(withNeed.transcript, undefined, 'the sheet itself is the message when there is a payload')
assert.equal(interpretVoiceEvent({ type: 'needs_consent', need: { k: 1 }, spoken: 'I need the HF token' }).transcript,
             '⚠ I need the HF token',
             'a spoken refusal is written down too: the dock may be collapsed, and a half-heard sentence ' +
             'is not a record of what was refused')

// THE ONE THAT VANISHED: setNeed(undefined) is falsy, so no sheet opened — and with no `spoken` field
// there was no transcript either. A voice turn refused in total silence.
const empty = interpretVoiceEvent({ type: 'needs_consent' })
assert.match(empty.transcript, /refused/,
             'a needs_consent frame carrying nothing renderable STILL says a refusal happened')
assert.equal(empty.need, null, 'and clears the sheet payload explicitly rather than leaving undefined')
assert.ok('need' in interpretVoiceEvent({ type: 'needs_consent' }),
          'the key must be PRESENT so the hook knows to write it — `if (eff.need)` would drop the clear')

// ── 5. errors say something even when the server sent no words ──
const err = interpretVoiceEvent({ type: 'error', error: 'model unavailable' })
assert.deepEqual(err, { transcript: '⚠ model unavailable', state: 'error' })
assert.equal(interpretVoiceEvent({ type: 'error' }).transcript, '⚠ the voice session reported an error',
             'an error frame with no message is still an error, not a blank warning triangle')
assert.equal(interpretVoiceEvent({ type: 'error', error: { code: 7 } }).transcript, '⚠ {"code":7}',
             'a structured error is rendered, not printed as [object Object]')

// ── 6. what a closed socket means ──
assert.deepEqual(voiceCloseState(1008, 'live'),
                 { state: 'error', transcript: '⚠ unauthorized — set the dashboard token in Settings' },
                 'the policy close is a SETTINGS problem — saying "it stopped listening" sends the ' +
                 'operator hunting a microphone fault instead')
assert.deepEqual(voiceCloseState(1000, 'live'), { state: 'idle' }, 'a normal close is just idle')
assert.deepEqual(voiceCloseState(undefined, 'connecting'), { state: 'idle' }, 'never connected: idle')
assert.deepEqual(voiceCloseState(1006, 'error'), { state: 'error' },
                 'an existing error SURVIVES the close it caused — the close is the consequence, and ' +
                 'overwriting it with "idle" hides the only explanation the operator had')
assert.equal(voiceCloseState(1008, 'error').transcript !== undefined, true,
             'but an unauthorized close still replaces the reason, because that one is actionable')

// ── 7. Q90 — the wiring: the microphone must not outlive the socket ──
// getUserMedia resolves BEFORE the socket opens, and the stream used to be stored only inside ws.onopen.
// So a refused (1008) or unreachable /ws/voice left a HOT MIC — recording indicator on, tracks live — for
// the whole life of the tab, with no UI in any state able to release it. Pure tests cannot see this.
const src = readFileSync(new URL('./useVoice.ts', import.meta.url), 'utf8')
const gum = src.indexOf('getUserMedia')
const onopen = src.indexOf('ws.onopen')
const owned = src.indexOf('nodesRef.current = { stream }')
assert.ok(owned > gum && owned < onopen,
          'Q90: the stream must be owned between getUserMedia and ws.onopen, not inside it')
assert.match(src, /ws\.onerror = \(\) => releaseMic\(\)/, 'a socket error releases the mic')
const closeBody = src.slice(src.indexOf('ws.onclose'), src.indexOf('ws.onopen'))
assert.match(closeBody, /releaseMic\(\)/,
             'and so does a close — including the 1008 refusal that never reaches onopen')
assert.match(src, /const releaseMic = useCallback/, 'releasing is its own function, callable from any state')
assert.match(src, /getTracks\(\)\.forEach\(t => t\.stop\(\)\)/, 'it actually stops the tracks')
assert.match(src, /if \(wsRef\.current\?\.readyState === WebSocket\.OPEN\) wsRef\.current\.send/,
             'stop() must not send into a socket that never opened — that throw used to skip the teardown ' +
             'below it, which is the second way the mic stayed live')

console.log('voiceSession.test.mjs: all assertions passed')
