import assert from 'node:assert/strict'
const { routeKnown, normalisePath, staleRouteMessage, unroutedByDetail } = await import('/tmp/serverAge.mjs')

const live = [
  '/api/fleet',
  '/api/devices',
  '/api/devices/logs/{peer_id}',
  '/api/robots/{peer_id}/teleop/publish',
]

// A plain route, present.
assert.equal(routeKnown(live, '/api/fleet'), true)
// Query strings and trailing slashes are not part of the route.
assert.equal(routeKnown(live, '/api/fleet?since=3'), true)
assert.equal(routeKnown(live, '/api/devices/'), true)
assert.equal(normalisePath('/api/fleet?a=1#x'), '/api/fleet')
assert.equal(normalisePath('/'), '/', 'a bare root must survive the trailing-slash trim')

// Templated routes match a real id — and only ONE segment of it.
assert.equal(routeKnown(live, '/api/devices/logs/so101-arm-1'), true)
assert.equal(routeKnown(live, '/api/robots/so101-arm-2/teleop/publish'), true)
assert.equal(routeKnown(live, '/api/devices/logs/so101/extra'), false, '{param} is one segment, not a path')

// The eight real cases from the live diff: absent => false, so the caller can explain the age.
for (const p of ['/api/devices/camera/2/modes', '/api/deploy/snippet', '/api/checkpoints/features',
                 '/api/robots/so101-arm-1/policy-fit', '/api/network/hint', '/api/training/output-dir']) {
  assert.equal(routeKnown(live, p), false, p)
}

// Unknown stays unknown: never blame the server for OUR missing fetch.
for (const nothing of [null, undefined, [], 'not-an-array']) {
  assert.equal(routeKnown(nothing, '/api/fleet'), null, JSON.stringify(nothing))
}

// The message names the path and the remedy, and carries no query noise.
const msg = staleRouteMessage('/api/devices/camera/2/modes?probe=1')
assert.match(msg, /does not have \/api\/devices\/camera\/2\/modes —/)
assert.ok(!msg.includes('probe=1'))
assert.match(msg, /Restart the dashboard/)

{
  assert.equal(unroutedByDetail('no endpoint at /api/datasets/labels'), true)
  // The framework's stock wording, which this app never sends — accepted for a TestClient app or a
  // later refactor. A matcher written from the docs alone was dead code here.
  assert.equal(unroutedByDetail('Not Found'), true)
  assert.equal(unroutedByDetail(' not found '), true)
  // THE DISCRIMINATOR: a route's own resource 404 must never be read as staleness, or an operator is
  // sent to restart a server that is working correctly.
  assert.equal(unroutedByDetail('no dataset directory at /tmp/x'), false)
  assert.equal(unroutedByDetail('unknown peer so101-arm-9'), false)
  for (const junk of [null, undefined, 404, {}, []]) assert.equal(unroutedByDetail(junk), false)
}
console.log('serverAge: all assertions passed')
