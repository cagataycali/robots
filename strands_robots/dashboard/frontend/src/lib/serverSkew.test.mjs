// Assertions for version-skew detection (lib/serverSkew.ts).
// Run: npx esbuild src/lib/serverSkew.ts --bundle --format=esm --outfile=/tmp/serverSkew.mjs && node src/lib/serverSkew.test.mjs
import assert from 'node:assert/strict'

const { skewHint, failureText, probeFromError } = await import('/tmp/serverSkew.mjs')

// THE SHAPE THIS DASHBOARD ACTUALLY SENDS, measured against the live server (server.py ~2117):
// {"error":"not found","detail":"no endpoint at /api/datasets/labels"}. The first version of this
// file tested only for FastAPI's stock "Not Found" and would therefore have been dead code.
{
  const hint = skewHint({
    status: 404, error: 'not found', detail: 'no endpoint at /api/datasets/labels',
    path: '/api/datasets/labels',
  })
  assert.ok(hint, 'the real unrouted-/api shape must be recognised')
  assert.match(hint, /Restart the dashboard from a terminal/)
}

// FastAPI's stock shape is still accepted: a bare TestClient app or a later refactor sends it.
{
  const hint = skewHint({ status: 404, detail: 'Not Found', path: '/api/datasets/labels' })
  assert.ok(hint)
  assert.match(hint, /newer than the dashboard process/)
  assert.match(hint, /Restart the dashboard from a terminal/)
  assert.match(hint, /camera grant/, 'why a terminal and not an agent — the trap that caused this state')
  assert.match(hint, /\/api\/datasets\/labels/)
}

// THE DISCRIMINATOR: a route's OWN 404 is about a resource, and must not send anyone to restart a
// server that is working perfectly.
{
  assert.equal(skewHint({ status: 404, detail: 'no dataset directory at /tmp/x' }), null)
}

// Everything else is not skew: auth, validation, a method mismatch (route exists), a dead socket.
{
  for (const status of [401, 403, 405, 422, 500, 502, 0, null, undefined]) {
    assert.equal(skewHint({ status, detail: 'Not Found' }), null, `status ${status} is not skew`)
  }
  assert.equal(skewHint(null), null)
  assert.equal(skewHint({ status: 404 }), null, 'no detail at all is not evidence of skew')
  // A resource 404 from a route that DOES exist, in this app's own wording.
  assert.equal(skewHint({ status: 404, error: 'not found', detail: 'no dataset directory at /tmp/x' }), null)
}

// Case and whitespace come from a real server, not from a spec.
{
  assert.ok(skewHint({ status: 404, detail: ' not found ' }))
}

// failureText keeps the caller's own message when the failure is something else.
{
  assert.match(failureText({ status: 404, detail: 'Not Found' }, 'boom'), /newer than the dashboard/)
  assert.equal(failureText({ status: 500, detail: 'x' }, 'boom'), 'boom')
  assert.equal(failureText(null, 'boom'), 'boom')
}

// probeFromError reads an HttpError-shaped throw without depending on its class.
{
  const p = probeFromError({ status: 404, body: { error: 'not found', detail: 'no endpoint at /api/x' } }, '/api/x')
  assert.deepEqual(p, { status: 404, detail: 'no endpoint at /api/x', error: 'not found', path: '/api/x' })
  assert.ok(skewHint(p), 'end to end: a real thrown HttpError produces the hint')
  assert.deepEqual(probeFromError(new Error('network down'), '/api/x'), { status: null, detail: null, error: null, path: '/api/x' })
  assert.deepEqual(probeFromError(undefined), { status: null, detail: null, error: null, path: undefined })
}

console.log('serverSkew: all assertions passed')
