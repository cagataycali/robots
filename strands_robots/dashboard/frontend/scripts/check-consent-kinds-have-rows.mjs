#!/usr/bin/env node
/**
 * Can every consent grant the SERVER can hold be REVOKED from the permissions screen?
 *
 * A grant the operator cannot see is a grant they cannot take back, and these grants are not
 * cosmetic: trust_remote_code executes code from a model repo, policy_host_allow hands a remote
 * host frames out and actions back. Q120 shipped two new kinds whose rows nobody added, and the
 * screen printed "nothing extra is allowed here" OVER live grants (Q121). The lesson is that the
 * UI-side list of kinds went stale TWICE — at 1 kind, then at 4 — because nothing failed when it
 * did.
 *
 * There is already an audit for this (audit-consent-rows.mjs), and it is a good one, but it drives
 * a BROWSER against a RUNNING dashboard on :8090. That means it cannot run in `npm test`, so on any
 * ordinary day it guards nothing: the seventh kind will ship exactly the way the fifth and sixth
 * did. This check is the cheap half that always runs — consent.py's KINDS tuple is the single
 * source of truth, and every kind in it must be named by ConsentSettings.tsx.
 *
 * It reports "X of Y checked" and REFUSES to exit 0 when it has been narrowed to nothing: a tool
 * that can be narrowed must say so, or a skipped parse prints a green "0 problems" (that false
 * green fooled an agent auditing the audits in this very repo).
 *
 * Run: node scripts/check-consent-kinds-have-rows.mjs   (also run at the end of `npm test`)
 */
import fs from 'node:fs'
import path from 'node:path'

const HERE = path.resolve(new URL('.', import.meta.url).pathname)
const CONSENT_PY = path.resolve(HERE, '../../consent.py')
const SCREEN = path.resolve(HERE, '../src/components/ConsentSettings.tsx')

const fail = msg => { console.error(`  FAIL  consent kinds: ${msg}`); process.exit(1) }

for (const p of [CONSENT_PY, SCREEN]) {
  if (!fs.existsSync(p)) fail(`cannot read ${path.relative(HERE, p)} — this check cannot be silently skipped`)
}

const py = fs.readFileSync(CONSENT_PY, 'utf8')
// KINDS: tuple[str, ...] = ( "a", "b", ... )
const block = py.match(/^KINDS[^=]*=\s*\(([^)]*)\)/m)
if (!block) fail('KINDS is no longer a tuple literal in consent.py — this parser must be updated, not skipped')
const kinds = [...new Set([...block[1].matchAll(/"([a-z_]+)"/g)].map(m => m[1]))]
if (kinds.length === 0) fail('parsed ZERO kinds from consent.py: narrowed to nothing, which is not a pass')

const screen = fs.readFileSync(SCREEN, 'utf8')
// A kind counts as reachable only if the screen names it as a STRING — that is what indexes the
// payload and what /api/consent/revoke is called with. A mention in a comment is not a row, so
// comments are stripped first.
const code = screen.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '')
const missing = kinds.filter(k => !code.includes(`'${k}'`) && !code.includes(`"${k}"`) && !code.includes(`.${k}`))

if (missing.length) {
  fail(
    `${missing.join(', ')} — the server can hold ${missing.length === 1 ? 'this grant' : 'these grants'} ` +
    `but the permissions screen cannot name ${missing.length === 1 ? 'it' : 'them'}, so nobody can revoke ` +
    `${missing.length === 1 ? 'it' : 'them'}. Add a row to components/ConsentSettings.tsx ` +
    `(granted_state + revoke_patch already handle every kind in KINDS).`,
  )
}
console.log(`  PASS  every consent kind can be revoked from the screen (${kinds.length} of ${kinds.length} checked: ${kinds.join(', ')})`)
