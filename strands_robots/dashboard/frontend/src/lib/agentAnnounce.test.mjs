import assert from 'node:assert/strict'
const { turnAnnouncement, clip, SPOKEN_MAX } = await import('/tmp/agentAnnounce.mjs')

// --- THE DEFECT: a reply nobody was told about ---------------------------------------
{
  const said = turnAnnouncement({ busy: false, last: { role: 'agent', text: 'four robots are online' } })
  assert.equal(said, 'the agent replied: four robots are online')
}

// --- silence while streaming, because the transcript mutates once per token ----------
{
  assert.equal(turnAnnouncement({ busy: true, last: { role: 'agent', text: 'four rob' } }), '',
    'announcing mid-stream would stutter the answer word by word — worse than silence')
  assert.equal(turnAnnouncement({ busy: false }), '', 'nothing to say about an empty dock')
}

// --- an error outranks the transcript, and interrupts a stream -----------------------
{
  const e = turnAnnouncement({ busy: true, error: 'this page is not signed in any more' })
  assert.match(e, /^the fleet agent could not be reached: this page is not signed in/)
}

// --- tools are part of the outcome, and silence-with-tools is not an empty reply -----
{
  const one = turnAnnouncement({ busy: false, last: { role: 'agent', text: 'stopped it', tools: [{ name: 'stop', status: 'ok' }] } })
  assert.equal(one, 'the agent replied after 1 tool: stopped it')
  const two = turnAnnouncement({ busy: false, last: { role: 'agent', text: 'done', tools: [{ name: 'a', status: 'ok' }, { name: 'b', status: 'ok' }] } })
  assert.match(two, /after 2 tools: done$/)
  const mute = turnAnnouncement({ busy: false, last: { role: 'agent', text: '  ', tools: [{ name: 'stop', status: 'ok' }] } })
  assert.equal(mute, 'the agent ran 1 tool and said nothing', 'an action taken without words is still news')
  assert.equal(turnAnnouncement({ busy: false, last: { role: 'agent', text: '' } }), '',
    'an empty bubble with no tools is the placeholder, not an answer')
}

// --- a failed send: the last word is the user's own, which means it never left --------
{
  assert.match(turnAnnouncement({ busy: false, last: { role: 'user', text: 'stop', delivered: false } }),
    /^your message was not delivered/)
  assert.equal(turnAnnouncement({ busy: false, last: { role: 'user', text: 'stop' } }), '',
    'a delivered message is not announced — the reply will be')
  assert.match(turnAnnouncement({ busy: false, last: { role: 'notice', text: 'arm-1 stopped' } }),
    /^notice from the fleet: arm-1 stopped$/)
}

// --- clip: long speech ends at a word and points at the page -------------------------
{
  assert.equal(clip('short'), 'short')
  const long = clip('word '.repeat(400))
  assert.ok(long.length < SPOKEN_MAX + 60, 'bounded')
  assert.match(long, /the rest is in the conversation$/)
  assert.doesNotMatch(long, /wor… /, 'cut at a word boundary')
  const nospace = clip('x'.repeat(SPOKEN_MAX + 50))
  assert.match(nospace, /x… the rest is in the conversation$/, 'no space to cut at is still bounded')
}
console.log('agentAnnounce: ok')
