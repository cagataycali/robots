// Assertions for the agent dock's delivery honesty (lib/chatDelivery.ts).
// Run: npx esbuild src/lib/chatDelivery.ts --bundle --format=esm --outfile=/tmp/chatDelivery.mjs \
//        && node src/lib/chatDelivery.test.mjs
import assert from 'node:assert/strict'

const { sendFailureVerdict, interruptionNotice, bubbleLabel } = await import('/tmp/chatDelivery.mjs')

// NEVER DELIVERED: says nothing ran, and promises the text back.
{
  const v = sendFailureVerdict({ error: 'could not reach the agent socket' })
  assert.match(v.text, /not sent/)
  assert.match(v.text, /could not reach the agent socket/)
  assert.match(v.text, /nothing ran/)
  assert.match(v.text, /back in the box/)
  assert.equal(v.retrySafe, true, 'a message that never left the browser is safe to retry')
}
// A failure with no message still explains itself.
assert.match(sendFailureVerdict({}).text, /socket could not be opened/)

// IDLE DROP: nothing to say. A dock that cries wolf on every reconnect teaches
// the operator to ignore the line that matters.
assert.equal(interruptionNotice({ wasBusy: false, code: 1001 }), null)
assert.equal(interruptionNotice({ wasBusy: false }), null)

// ...except a refusal, which will repeat and needs fixing rather than retrying.
{
  const n = interruptionNotice({ wasBusy: false, code: 1008 })
  assert.match(n.text, /rejected this token/)
  assert.equal(n.tone, 'bad')
  assert.equal(n.doubleRunRisk, false, 'nothing was in flight, so nothing can run twice')
}

// DROPPED MID-ANSWER: the truncation must be stated, or a half answer reads as
// the whole one.
{
  const n = interruptionNotice({ wasBusy: true, partialChars: 120 })
  assert.match(n.text, /connection dropped before the answer finished/)
  assert.match(n.text, /stops mid-answer/)
  assert.match(n.text, /not a complete reply/)
  assert.match(n.text, /Re-sending repeats the request/)
  assert.equal(n.doubleRunRisk, true)
}

// THE ONE THAT MATTERS: a turn cut off after a tool started may have MOVED A
// ROBOT. "no answer" would be a lie about the physical world.
{
  const n = interruptionNotice({ wasBusy: true, partialChars: 0, runningTools: ['fleet_stop'] })
  assert.match(n.text, /The tool fleet_stop had already started/)
  assert.match(n.text, /may have ACTED on the fleet/)
  assert.match(n.text, /check Activity/)
  assert.equal(n.doubleRunRisk, true)
}
{
  const n = interruptionNotice({ wasBusy: true, runningTools: ['a', 'b', 'c', 'd'] })
  assert.match(n.text, /The tools a, b, c \+1 more had already started/)
}
// Blank/whitespace tool names are not tools; with none left, no fleet claim.
{
  const n = interruptionNotice({ wasBusy: true, runningTools: ['', '  '] })
  assert.doesNotMatch(n.text, /had already started/)
  assert.doesNotMatch(n.text, /ACTED/)
}

// NOTHING CAME BACK: delivery already happened, so a blind retry is the risk to
// name — not "your message was lost".
{
  const n = interruptionNotice({ wasBusy: true, partialChars: 0 })
  assert.match(n.text, /Nothing came back/)
  assert.match(n.text, /already been sent/)
  assert.match(n.text, /run it a second time/)
  assert.doesNotMatch(n.text, /not sent/, 'it WAS sent — that is the whole point')
  assert.equal(n.doubleRunRisk, true)
}

// The three worlds are three different sentences.
{
  const texts = new Set([
    sendFailureVerdict({ error: 'x' }).text,
    interruptionNotice({ wasBusy: true, partialChars: 0 }).text,
    interruptionNotice({ wasBusy: true, partialChars: 50 }).text,
  ])
  assert.equal(texts.size, 3)
}
// ...and only the never-delivered one is safe to repeat automatically.
assert.equal(sendFailureVerdict({ error: 'x' }).retrySafe, true)
assert.equal(interruptionNotice({ wasBusy: true }).doubleRunRisk, true)

// An auth refusal mid-turn keeps the cause AND the truncation warning.
{
  const n = interruptionNotice({ wasBusy: true, code: 1008, partialChars: 10 })
  assert.match(n.text, /cut off/)
  assert.match(n.text, /rejected this token/)
  assert.match(n.text, /stops mid-answer/)
}

// Junk numbers cannot produce a nonsense verdict.
for (const bad of [NaN, -5, undefined, null, 'x']) {
  const n = interruptionNotice({ wasBusy: true, partialChars: bad })
  assert.match(n.text, /Nothing came back/, 'unreadable length is treated as no text, never as a partial answer')
}

// Bubble label: only a KNOWN failure marks a bubble; unknown stays quiet.
assert.equal(bubbleLabel(false), 'not sent')
assert.equal(bubbleLabel(true), null)
assert.equal(bubbleLabel(undefined), null)

console.log('chatDelivery: all assertions passed')
