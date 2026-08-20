#!/usr/bin/env node
/**
 * Which routes does the SHIPPED bundle call that the RUNNING server does not have? (Q79)
 *
 * The dashboard process lives for days while this bundle is rebuilt on every landed change, so the two
 * halves are routinely different ages and the symptom is a bare 404 that blames the resource. This reads
 * the running server's own /openapi.json and diffs it against the routes the current source registers —
 * no POSTing, no side effects, safe against a live fleet (never probe /api/safety/estop to see if it is
 * there).
 *
 * Run: node scripts/audit-server-age.mjs        (env: STRANDS_DASH_URL, STRANDS_DASH_TOKEN_FILE)
 * Exit 0 always: an old running server is news for the operator, not a broken build.
 */
import fs from 'node:fs'
import path from 'node:path'

const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const TOKEN_FILE = process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`
const token = fs.existsSync(TOKEN_FILE) ? fs.readFileSync(TOKEN_FILE, 'utf8').trim() : ''

const serverPy = path.resolve('../server.py')
const src = fs.readFileSync(serverPy, 'utf8')
const declared = [...src.matchAll(/@app\.(?:get|post|put|delete)\("(\/api[^"]*)"/g)].map(m => m[1])

let live
try {
  const res = await fetch(`${BASE}/openapi.json`, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
  if (!res.ok) throw new Error(`HTTP ${res.status} — is the token current?`)
  live = Object.keys((await res.json()).paths ?? {})
} catch (e) {
  console.log(`  SKIP  no server to compare at ${BASE}: ${e.message}`)
  process.exit(0)
}

const missing = declared.filter(p => !live.includes(p)).sort()
const stale = live.filter(p => p.startsWith('/api/') && !declared.includes(p)).sort()

console.log(`  server at ${BASE} publishes ${live.length} paths; this source declares ${declared.length}`)
if (missing.length === 0) {
  console.log('  PASS  the running server has every route this source calls')
} else {
  console.log(`  OLD   ${missing.length} route(s) this bundle calls are NOT on the running server:`)
  for (const p of missing) console.log(`          ${p}`)
  console.log('        → the UI explains these as "restart the dashboard to pick it up" (lib/serverAge.ts).')
  console.log('        → an owner-run restart from a terminal makes them live (never restart from a daemon:')
  console.log('          a launchd-descended process can never be granted camera access on macOS).')
}
if (stale.length) console.log(`  note  ${stale.length} live route(s) not found in server.py (registered elsewhere, or removed since)`)
