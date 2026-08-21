#!/usr/bin/env node
/**
 * Q127: no <img> may point at one of this dashboard's own /api routes.
 *
 * A browser image request carries no Authorization header, so such an image 401s in every
 * authenticated session (i.e. every remote one) and renders as a broken glyph the operator reads as
 * missing data. AuthedImg fetches the bytes through lib/endpoints and hands over an object URL.
 *
 * WHAT THIS CAN AND CANNOT SEE, stated plainly so nobody trusts it too far: it catches an /api path
 * written INSIDE an <img> tag, which is the shape a regression takes when someone renders a route
 * directly. It cannot see `src={url}` where url arrived in a payload — the exact shape of the
 * original bug — because that is only knowable at runtime. The thumbnail case is therefore ALSO
 * pinned by test_dashboard_route_reach.py knowing that route is server-emitted, and by AuthedImg
 * being the only importer of apiBlob outside CameraGallery.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, dirname } from 'node:path'

const SRC = join(dirname(new URL(import.meta.url).pathname), '../src')
const walk = (d, out = []) => {
  for (const e of readdirSync(d)) {
    const p = join(d, e)
    statSync(p).isDirectory() ? walk(p, out) : out.push(p)
  }
  return out
}
const files = walk(SRC).filter(f => /\.tsx$/.test(f) && !f.includes('.test.'))
const bad = []
for (const f of files) {
  // Comments first, always: this guard's OWN file quotes the forbidden pattern to explain it, and
  // version one reported itself. Third time in three iterations that a comment was read as code
  // (Q125 twice), so it is now the first thing any scan in this repo does.
  const text = readFileSync(f, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '')
  for (const m of text.matchAll(/<img\b[^>]*>/gs)) {
    if (/\/api\//.test(m[0])) bad.push([f.split('/').pop(), m[0].replace(/\s+/g, ' ').slice(0, 90)])
  }
}
if (files.length === 0) {
  console.error('FAIL authed-images: no .tsx files found — the scan is broken, not the code')
  process.exit(1)
}
if (bad.length) {
  console.error(`FAIL authed-images: an <img> names an /api route (use AuthedImg — a browser image request has no bearer token)`)
  for (const [f, snip] of bad) console.error(`   ${f}: ${snip}`)
  process.exit(1)
}
console.log(`  ok    no <img> fetches an /api route unauthenticated (${files.length} components)`)
