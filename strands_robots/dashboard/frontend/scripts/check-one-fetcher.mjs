#!/usr/bin/env node
/**
 * Does every HTTP request in this app still go through src/lib/endpoints.ts?
 *
 * That single funnel is not a style preference — three separate behaviours are attached to it, and each
 * is INVISIBLE when a component calls fetch() directly:
 *   · the bearer token (a direct fetch just 401s, and remotely that reads as "logged out");
 *   · the Q79 server-age explanation — a 404 from a route the running server does not have is rendered
 *     as "restart the dashboard to pick it up" instead of blaming the resource ("no such camera");
 *   · the 10 currently-dark routes are covered BY CONSTRUCTION because of it, which is exactly how
 *     /api/record/upload-preflight (surfaced only today, by a scan that had been blind to it) turned out
 *     to already degrade honestly. Nobody wrote a line of code for that route: the funnel did it.
 *
 * So the property worth pinning is not "endpoints.ts is nice", it is "a NEW screen cannot silently opt
 * out of the explanations". Measured when written: 3 fetch call sites, all in endpoints.ts.
 *
 * Node scripts under scripts/ are excluded on purpose — they are audits talking to the server from
 * outside the app, and they must not be routed through the app's own token/404 machinery.
 *
 * Run: node scripts/check-one-fetcher.mjs
 */
import fs from 'node:fs'
import path from 'node:path'

const ROOT = path.resolve(new URL('.', import.meta.url).pathname, '..')
const SRC = path.join(ROOT, 'src')
const FUNNEL = path.join(SRC, 'lib', 'endpoints.ts')

const walk = (d) => fs.readdirSync(d, { withFileTypes: true })
  .flatMap(e => e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)])

const files = walk(SRC).filter(f => /\.(ts|tsx)$/.test(f) && !/\.test\./.test(f) && f !== FUNNEL)

// `fetch(` preceded by nothing that makes it a definition/mock, and not inside a line comment.
const CALL = /(^|[^\w.$])fetch\s*\(/
const offenders = []
for (const f of files) {
  fs.readFileSync(f, 'utf8').split('\n').forEach((line, i) => {
    const code = line.replace(/\/\/.*$/, '').replace(/\/\*.*?\*\//g, '')
    if (CALL.test(code) && !/globalThis\.fetch\s*=/.test(code)) {
      offenders.push(`${path.relative(ROOT, f)}:${i + 1}  ${line.trim().slice(0, 90)}`)
    }
  })
}

if (offenders.length) {
  console.error(`  ${offenders.length} direct fetch call(s) outside src/lib/endpoints.ts:`)
  for (const o of offenders) console.error(`    ${o}`)
  console.error('  A request made here carries no bearer token and gets NO server-age explanation: its')
  console.error('  404 will blame the resource ("no such camera") when the server is merely older than')
  console.error('  this bundle. Use api()/post() from lib/endpoints, or state here why this one is exempt.')
  process.exit(1)
}
console.log(`  every request goes through lib/endpoints (${files.length} app files carry no direct fetch)`)
process.exit(0)
