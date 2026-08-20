import assert from 'node:assert/strict'
import { nameVerdict, freeVariant } from '/tmp/datasetName.mjs'

const known = [
  { repo_id: 'local/cubes', root: '/d/local/cubes', total_episodes: 40 },
  { repo_id: 'local/half', root: '/d/local/half', total_episodes: 0 },
  { repo_id: 'local/cubes-2', root: '/d/local/cubes-2', total_episodes: 3 },
  { repo_id: 'lerobot/pusht', local: false, total_episodes: 206 },
]

// A free name says nothing at all: this warns EARLIER than the backend, never instead of it.
assert.equal(nameVerdict('local/fresh', known), null)
assert.equal(nameVerdict('', known), null)
assert.equal(nameVerdict('   ', known), null)

// No evidence is not evidence of a problem. A failed/absent listing must stay silent rather than
// guess - the same posture as the origin badge and the arm-role slots.
assert.equal(nameVerdict('local/cubes', null), null)

// A taken name names the stake: 40 episodes is an afternoon of hand-guiding.
const v = nameVerdict('local/cubes', known)
assert.ok(v && v.message.includes('40 episode(s)'))
assert.match(v.message, /refuses to reuse/)
// ...and the suggestion skips cubes-2, which is also taken.
assert.equal(v.suggestion, 'local/cubes-3')

// Zero episodes reads as an interrupted session, and must not claim recorded work is at stake.
const half = nameVerdict('local/half', known)
assert.ok(half && /interrupted session/.test(half.message))
assert.ok(!/40|destroy/.test(half.message))

// A Hub dataset of the same name is not what a local recording collides with.
assert.equal(nameVerdict('lerobot/pusht', known), null)

// Pressing the suggestion twice walks forward instead of making "cubes-2-2".
assert.equal(freeVariant('local/cubes-3', known), 'local/cubes-4')
assert.equal(freeVariant('local/cubes-2', known), 'local/cubes-3')
assert.equal(freeVariant('', known), null)
// A whole family taken: bounded search, honest empty answer, no hang.
const many = Array.from({ length: 400 }, (_, i) => ({ repo_id: `local/x-${i + 2}` }))
assert.equal(freeVariant('local/x', many), null)

console.log('datasetName: all assertions passed')
