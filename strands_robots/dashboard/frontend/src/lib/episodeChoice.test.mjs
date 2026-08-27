import assert from 'node:assert/strict'
import { episodeChoice } from '/tmp/datasetSelection.mjs'

const row = (extra = {}) => ({ repo_id: 'cagatay/pick', root: '/data/pick', local: true, ...extra })

// --- the default is the OLD behaviour, so nobody's muscle memory breaks --------------------------
for (const blank of [undefined, null, '', '  ']) {
  const v = episodeChoice(row({ total_episodes: 40 }), blank)
  assert.ok(v.ok && v.episode === 0, `blank (${JSON.stringify(blank)}) must mean episode 0`)
}
assert.match(episodeChoice(row({ total_episodes: 40 })).reason, /of 40/, 'the count belongs in the label')

// --- a real index is honoured (this is the whole bug: 0 was hardcoded) --------------------------
assert.deepEqual(episodeChoice(row({ total_episodes: 40 }), 39).episode, 39)
assert.deepEqual(episodeChoice(row({ total_episodes: 40 }), '7').episode, 7)

// --- off-by-one is THE mistake this box produces, so the refusal names the range ----------------
const over = episodeChoice(row({ total_episodes: 40 }), 40)
assert.ok(!over.ok, '40 does not exist in a 40-episode dataset')
assert.match(over.reason, /40 episodes, numbered 0–39/, 'name the range, not just "invalid"')
assert.match(episodeChoice(row({ total_episodes: 1 }), 1).reason, /1 episode,/, 'singular reads like English')

// --- nonsense is refused next to the box, not deep in a loader ----------------------------------
assert.ok(!episodeChoice(row({ total_episodes: 5 }), 'banana').ok)
assert.ok(!episodeChoice(row({ total_episodes: 5 }), 2.5).ok, 'half an episode is not an episode')
assert.ok(!episodeChoice(row({ total_episodes: 5 }), -1).ok)
assert.match(episodeChoice(row({ total_episodes: 5 }), -1).reason, /start at 0/)

// --- an ABSENT count is not zero episodes: pass it through, say the count is unknown -------------
const unknown = episodeChoice(row(), 12)
assert.ok(unknown.ok && unknown.episode === 12, 'a missing fact must not become a refusal')
assert.equal(unknown.countKnown, false)
assert.match(unknown.reason, /does not report a count/)

// --- but a count of zero IS a fact, and it refuses -----------------------------------------------
assert.ok(!episodeChoice(row({ total_episodes: 0 }), 0).ok)

console.log('episodeChoice: 16 assertions passed')
