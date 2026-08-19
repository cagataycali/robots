import assert from 'node:assert/strict'
import { statusSentence, ribbonDetail } from '/tmp/statusSentence.mjs'

const base = { stale: false, lastSeenAgoS: 1, hwConnected: true, taskStatus: 'idle',
  instruction: null, taskDurationS: null, moving: false, stateAgeS: 0.2 }

// the ribbon shows the word as a chip; the sentence must not repeat it
const idle = statusSentence(base)
assert.equal(idle.word, 'idle')
assert.equal(idle.text, 'idle and still — safe to approach')
assert.equal(ribbonDetail(idle), 'and still — safe to approach')
// ...and the underlying sentence is untouched for every other consumer
assert.equal(idle.text, 'idle and still — safe to approach')

// a sentence that does NOT open with the word is passed through verbatim
const moving = statusSentence({ ...base, moving: true })
assert.equal(moving.word, 'moving')
assert.match(moving.text, /keep hands clear/)
assert.equal(ribbonDetail(moving), moving.text, 'must not rewrite a non-duplicate')

// stripping can never empty the detail — a chip alone says too little
assert.equal(ribbonDetail({ severity: 'ok', word: 'idle', text: 'idle' }), 'idle')
assert.equal(ribbonDetail({ severity: 'ok', word: 'idle', text: 'idle — ' }), 'idle — ')
// nor can a missing/odd word break it
assert.equal(ribbonDetail({ severity: 'ok', word: '', text: 'anything' }), 'anything')
assert.equal(ribbonDetail({ severity: 'ok', word: 'no hw', text: 'no hw connected — unplugged' }),
  'connected — unplugged')
// case-insensitive, and the leading punctuation goes with the word
// matching is case-insensitive: a chip reading WEDGED? still deduplicates
assert.equal(ribbonDetail({ severity: 'warn', word: 'Wedged?', text: 'wedged? maybe' }), 'maybe')
// ...but only a WHOLE leading word counts. 'idle' must not eat the front of
// 'idling along' and leave 'ing along' — the bug this assertion caught.
assert.equal(ribbonDetail({ severity: 'ok', word: 'idle', text: 'idling along nicely' }), 'idling along nicely')
assert.equal(ribbonDetail({ severity: 'ok', word: 'still', text: 'stillness is fine' }), 'stillness is fine')
assert.equal(ribbonDetail({ severity: 'warn', word: 'IDLE', text: 'idle: nothing to do' }), 'nothing to do')

// the safety-critical states keep their whole warning
for (const f of [{ ...base, stale: true }, { ...base, hwConnected: false },
                 { ...base, taskStatus: 'running', moving: false, taskDurationS: 9 }]) {
  const l = statusSentence(f)
  assert.ok(ribbonDetail(l).length > 12, `${l.word}: detail got thin: ${ribbonDetail(l)}`)
}
console.log('statusSentence/ribbonDetail: all assertions passed')
