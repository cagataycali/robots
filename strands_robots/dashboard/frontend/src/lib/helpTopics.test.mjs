import assert from 'node:assert/strict'
import { HELP_TOPICS, DOC_LINKS, REPO_DOC_PATHS, DOCS_ORIGIN } from '/tmp/helpTopics.mjs'

// --- the help must be usable with no network -------------------------------
assert.ok(HELP_TOPICS.length >= 4, 'a single paragraph is not onboarding')
for (const t of HELP_TOPICS) {
  assert.ok(t.title.length > 0 && t.lines.length > 0, `${t.title}: empty topic`)
  for (const line of t.lines) {
    assert.ok(line.trim().length > 20, `${t.title}: a stub line`)
    assert.doesNotMatch(line, /https?:\/\//, `${t.title}: help text must not depend on a URL`)
  }
}

const text = HELP_TOPICS.flatMap(t => t.lines).join(' ').toLowerCase()

// THE SAFETY CONTENT IS THE POINT — JOURNEYS #7's cost was "the cheapest legal
// first action is an unlabelled ▶ on a real arm". Help that omits the brake, or
// omits the safe first move, would leave that exactly as it was.
assert.match(text, /stop all/, 'the brake must be named')
assert.match(text, /"\."|\bthe "\." key\b/, 'the . hotkey is the only control that works on every screen')
assert.match(text, /power switch/, 'name the brake that does not go through this page')
assert.match(text, /mock/, 'a safe first action must be offered by name')
assert.match(text, /sim/, 'the safe first action happens on a sim peer')
assert.match(text, /confirmation/, 'say that a real arm asks first')

// the four-step loop is present and in order
const collect = HELP_TOPICS.find(t => /collect/i.test(t.title))
assert.ok(collect, 'the collect->train->deploy path must be explained')
assert.equal(collect.lines.length, 4, 'four steps, one line each')
assert.ok(collect.lines[0].includes('devices'))
assert.ok(collect.lines[1].includes('record'))
assert.ok(collect.lines[2].includes('train'))

// --- a link that 404s is worse than no link --------------------------------
assert.ok(DOC_LINKS.length > 0, 'JOURNEYS #7: zero links was the finding')
for (const l of DOC_LINKS) {
  assert.ok(l.url.startsWith(`${DOCS_ORIGIN}/`), `${l.url}: help links stay on the docs site`)
  assert.ok(l.label && l.note, `${l.url}: an unexplained link is another guess`)
  // MEASURED 2026-08-19: every /dashboard/* page on the deployed site 404s,
  // though the markdown exists in this repo and mkdocs.yml lists it. The
  // obvious "Fleet Dashboard > Quickstart" link would have been a 404 handed to
  // a first-time operator.
  assert.doesNotMatch(l.url, /\/dashboard\//,
    'the deployed site has no /dashboard/* pages yet — keep those as repo paths')
}
// no duplicate destinations
assert.equal(new Set(DOC_LINKS.map(l => l.url)).size, DOC_LINKS.length)

// --- unpublished pages are TEXT, so they cannot rot into broken links -------
for (const p of REPO_DOC_PATHS) {
  assert.match(p, /^docs\/.+\.md — .+/, `${p}: a path plus what it is for`)
  assert.doesNotMatch(p, /https?:\/\//, 'these are paths in the repo, not URLs')
}

console.log('helpTopics: all assertions passed')
