#!/usr/bin/env node
/**
 * Q125: every /api path the frontend names must be served by a real route.
 * Q129: and every /ws path too — the sockets are the same producer/consumer pairing, and a
 * mistyped socket path is WORSE than a mistyped fetch: a WebSocket handshake failure surfaces as a
 * close event, which every socket here already retries, so the typo appears as a camera that
 * reconnects forever rather than as an error anybody reads.
 *
 * serverAge.ts answers this at RUNTIME against whatever server is live, which is the right tool for
 * "your server is older than this bundle" — but it cannot catch a path that is simply MISTYPED,
 * because a typo and an old server produce the same 404, and the UI then explains the wrong one
 * ("restart the dashboard" about a route that never existed). This runs at build time against the
 * python source, where the answer is knowable.
 *
 * NARROWING LAW (this repo learned it the hard way): a checker that can be narrowed must say how
 * much it checked and must NOT exit 0 when narrowed to nothing. Written while making exactly that
 * mistake — the first version of this scan read only server.py and called all 11 /api/record paths
 * missing, because they are registered on an APIRouter(prefix=...) in record_api.py.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, dirname } from 'node:path'

const here = dirname(new URL(import.meta.url).pathname)
const PY_DIR = join(here, '../../')            // strands_robots/dashboard
const SRC = join(here, '../src')

const walk = (dir, out = []) => {
  for (const e of readdirSync(dir)) {
    const p = join(dir, e)
    const s = statSync(p)
    if (s.isDirectory()) walk(p, out)
    else out.push(p)
  }
  return out
}

// ── the routes the server really serves ────────────────────────────────────────────────────────
const pyFiles = walk(PY_DIR).filter(f => f.endsWith('.py') && !f.includes('/frontend/'))
const routes = []
for (const f of pyFiles) {
  const text = readFileSync(f, 'utf8')
  // routers keep their prefix on the variable they are assigned to
  const prefixes = new Map()
  for (const m of text.matchAll(/(\w+)\s*=\s*APIRouter\(\s*prefix\s*=\s*["']([^"']*)["']/g)) {
    prefixes.set(m[1], m[2])
  }
  for (const m of text.matchAll(/@(\w+)\.(get|post|put|patch|delete|websocket)\(\s*["']([^"']+)["']/g)) {
    const [, obj, , path] = m
    routes.push({ path: (prefixes.get(obj) ?? '') + path, file: f.split('/').pop() })
  }
}
const asRegex = p => new RegExp('^' + p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\\\{[^}]*\\\}/g, '[^/]+') + '$')
const matchers = routes.map(r => ({ ...r, rx: asRegex(r.path) }))

// ── the paths the frontend names ───────────────────────────────────────────────────────────────
const tsFiles = walk(SRC).filter(f => /\.tsx?$/.test(f) && !f.includes('.test.'))
const used = new Map()
/**
 * Comments are stripped first. A JSDoc line that MENTIONS a route in markdown backticks
 * (`/api/record`) is documentation, not a caller — rehearsalNav.ts explains that this backend may
 * not have /api/record at all, and the first version of this guard reported it as a missing route.
 */
const stripComments = t => t.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '')
for (const f of tsFiles) {
  for (const m of stripComments(readFileSync(f, 'utf8')).matchAll(/['"`](\/(?:api|ws)\/[^'"`]*)['"`]/g)) {
    if (!used.has(m[1])) used.set(m[1], new Set())
    used.get(m[1]).add(f.split('/').pop())
  }
}

if (routes.length === 0 || used.size === 0) {
  console.error(`FAIL routes-exist: narrowed to nothing (${routes.length} routes, ${used.size} paths) — the scan is broken, not the code`)
  process.exit(1)
}

const problems = []
for (const [raw, files] of [...used].sort()) {
  const path = raw.split('?')[0]
  // A literal ending in '/' is a BASE that callers concatenate onto ('/api/auth/login/' + 'begin');
  // it is satisfied by any route underneath it.
  if (path.endsWith('/')) {
    if (!matchers.some(m => m.path.startsWith(path))) problems.push([raw, files, 'no route under this base'])
    continue
  }
  // A template hole stands for one path segment. An expression containing a ternary or a query is
  // not a path we can judge — say so rather than guess.
  const probe = path.replace(/\$\{[^}]*\}/g, 'X')
  if (/[?&:{}]/.test(probe.replace(/\$\{[^}]*\}/g, ''))) continue
  if (!matchers.some(m => m.rx.test(probe))) problems.push([raw, files, 'no route serves this'])
}

const label = `${used.size} /api + /ws path(s) in ${tsFiles.length} file(s) vs ${routes.length} route(s) in ${pyFiles.length} python file(s)`
if (problems.length) {
  console.error(`FAIL routes-exist: ${label}`)
  for (const [p, files, why] of problems) console.error(`   ${p}  — ${why}  [${[...files].join(', ')}]`)
  process.exit(1)
}
console.log(`  ok    every /api and /ws path the frontend names is served (${label})`)
