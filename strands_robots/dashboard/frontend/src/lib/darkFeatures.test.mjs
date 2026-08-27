// build: npx esbuild src/lib/darkFeatures.ts --bundle --format=esm --outfile=/tmp/darkFeatures.mjs
// build: npx esbuild src/lib/bundleRoutes.generated.ts --bundle --format=esm --outfile=/tmp/bundleRoutes.mjs
import assert from 'node:assert/strict'
import { darkRoutes, darkFeatureMessage } from '/tmp/darkFeatures.mjs'

const NEEDED = ['/api/fleet', '/api/deploy/snippet', '/api/devices/camera/{index}/modes']

// A server that has everything: nothing to say.
assert.deepEqual(darkRoutes(['/api/fleet', '/api/deploy/snippet', '/api/devices/camera/{index}/modes'], NEEDED), [])

assert.deepEqual(darkRoutes(['/api/fleet'], NEEDED),
  ['/api/deploy/snippet', '/api/devices/camera/{index}/modes'], 'sorted, and only the missing ones')

// A TEMPLATED need against a templated server path: no second path algebra, serverAge's
// segment matcher accepts a literal {index} as one segment.
assert.deepEqual(darkRoutes(['/api/devices/camera/{index}/modes'], ['/api/devices/camera/{index}/modes']), [])
assert.deepEqual(darkRoutes(['/api/devices/camera/{i}/modes'], ['/api/devices/camera/{index}/modes']), [],
  'the server names its parameter differently — still the same route')

assert.deepEqual(darkRoutes(['/api/auth/login/begin', '/api/auth/login/finish'], ['/api/auth/login/']), [],
  'a base is satisfied by a route underneath it')
assert.deepEqual(darkRoutes(['/api/fleet'], ['/api/auth/login/']), ['/api/auth/login/'],
  'a base with nothing beneath it IS dark')

// SILENCE WHEN UNKNOWN. Each of these is a way of not knowing, and none of them is "everything is dark".
for (const unknown of [null, undefined, []]) {
  assert.deepEqual(darkRoutes(unknown, NEEDED), [], `unknown route table (${JSON.stringify(unknown)}) says nothing`)
}

// The sentence: a count, the remedy, and WHY the remedy must be a terminal.
assert.equal(darkFeatureMessage([]), null, 'nothing dark → no banner at all')
const one = darkFeatureMessage(['/api/deploy/snippet'])
assert.match(one, /^1 feature on this page is dark/, 'singular reads as English')
assert.match(one, /Restarting the dashboard from a terminal/)
assert.match(one, /never grant camera access/, 'names why a daemon restart is not the remedy')
const many = darkFeatureMessage(['/a', '/b', '/c'])
assert.match(many, /^3 features on this page are dark/)
assert.match(many, /lights them up/)

// The generated list is the default subject, and it is not empty — an empty default would make this
// whole banner silently dead while every test above still passed.
const { BUNDLE_ROUTES } = await import('/tmp/bundleRoutes.mjs')
assert.ok(BUNDLE_ROUTES.length > 30, `the generated route list carries the real surface (${BUNDLE_ROUTES.length})`)
assert.ok(BUNDLE_ROUTES.includes('/api/fleet'), 'the fleet route is in the extraction')
assert.deepEqual(darkRoutes([...BUNDLE_ROUTES]), [], 'a server with exactly what the bundle calls is not dark')
console.log('darkFeatures: all assertions passed')
