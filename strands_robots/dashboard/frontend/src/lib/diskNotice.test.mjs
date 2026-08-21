import assert from 'node:assert/strict'
import { build } from 'esbuild'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const out = join(mkdtempSync(join(tmpdir(), 'disknotice-')), 'm.mjs')
await build({ entryPoints: ['src/lib/diskNotice.ts'], outfile: out, format: 'esm', bundle: true })
const { diskNoticeView } = await import(out)

// nothing to say renders nothing
assert.equal(diskNoticeView(null), null)
assert.equal(diskNoticeView(undefined), null)
// a malformed payload must render NOTHING, never an empty warning box
assert.equal(diskNoticeView({ level: 'weird', headline: 'x' }), null)
assert.equal(diskNoticeView({ level: 'tight', headline: '   ' }), null)

const tight = { level: 'tight', free_mb: 6000, headline: '5.9Gi free - tight', advice: 'watch it' }
const v = diskNoticeView(tight)
assert.equal(v.tone, 'warn')
assert.equal(v.urgent, false, 'a tight disk polled at 1Hz must not shout at a screen reader')
assert.equal(v.advice, 'watch it', 'the backend keeps authorship of ordinary advice')

const crit = { level: 'critical', free_mb: 700, headline: 'only 700Mi free', advice: 'free space first' }
const idle = diskNoticeView(crit)
assert.equal(idle.tone, 'bad')
assert.equal(idle.urgent, true)
assert.equal(idle.advice, 'free space first', 'idle: the backend advice IS reachable, keep it')

// THE RULE THIS FILE EXISTS FOR: mid-session, "free space first" is unreachable advice for someone
// holding an arm over a live dataset, so the words change and the artifact is what they protect.
const live = diskNoticeView(crit, { recording: true })
assert.equal(live.urgent, true)
assert.match(live.advice, /Stop after this episode/)
assert.match(live.advice, /complete and safe/)
assert.ok(!/free space first/.test(live.advice))
assert.equal(live.headline, crit.headline, 'the measured fact is never rewritten')
assert.notEqual(live.testid, idle.testid, 'the two cases are distinguishable on the page')

console.log('diskNotice: 15 assertions ok')
