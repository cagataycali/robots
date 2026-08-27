import assert from 'node:assert/strict'
import { safeFilename, snippetRefusal } from '/tmp/deploySnippet.mjs'

// The server names the file; we only make it safe to WRITE.
assert.equal(safeFilename('so101-arm-1.py'), 'so101-arm-1.py')
assert.equal(safeFilename(null, 'so101-arm-1'), 'so101-arm-1.py')
assert.equal(safeFilename('', null), 'robot.py')
assert.equal(safeFilename(undefined, ''), 'robot.py')
// a parent__child peer id is legitimate and must survive unharmed
assert.equal(safeFilename('so101-follower-twin__so101.py'), 'so101-follower-twin__so101.py')
// a slash would become a directory the browser will not create: the download then fails silently
assert.equal(safeFilename('robots/arm 1.py'), 'robots-arm-1.py')
assert.equal(safeFilename('c:\\arm*?.py'), 'c-arm-.py')
// never a dotfile, never bare
assert.equal(safeFilename('...py'), 'py.py')
assert.equal(safeFilename('.hidden.py'), 'hidden.py')
// exactly one .py, whatever the server sent
assert.equal(safeFilename('arm'), 'arm.py')
assert.equal(safeFilename('arm.py.py'), 'arm.py.py'.replace(/\.py$/, '') + '.py')

// A refusal must name a remedy when one exists — this one is the common case, not a bug.
assert.match(snippetRefusal(422, 'no profile remembered for \'ABC123\''), /spawn it once/)
assert.match(snippetRefusal(404, ''), /spawn it once/)
assert.equal(snippetRefusal(422, 'payload has no robot_name'), 'payload has no robot_name')
assert.match(snippetRefusal(401, ''), /session expired/)
assert.match(snippetRefusal(500, ''), /could not write the snippet/)
assert.equal(snippetRefusal(500, 'disk full'), 'disk full')
console.log('deploySnippet: all assertions passed')

// --- Q122's other half: the operator can now SUPPLY the hub address (added section, not a new file)
import { hubAddressMissing, cleanHubHost } from '/tmp/deploySnippet.mjs'

const WITH_HUB = 'import os\nos.environ.setdefault("ZENOH_CONNECT", "tcp/192.168.1.20:7447")\n'
const WITHOUT = '# NOTE: robots.cagatay.my is a public address with no zenoh port.\nimport os\n'

// The PRESENCE of the real line is the proof, not the absence of a note: the note outlives a
// successful override, so keying off it would keep nagging someone who already fixed this.
assert.equal(hubAddressMissing(WITH_HUB), null, 'an address made it in — say nothing')
assert.equal(hubAddressMissing(WITH_HUB + WITHOUT), null, 'a note beside a real line is not a gap')
assert.equal(hubAddressMissing(WITHOUT), 'robots.cagatay.my is a public address with no zenoh port',
  "the server's own sentence is reused, never re-derived")
assert.match(hubAddressMissing('import os\n'), /connect to nothing/, 'no note still needs a prompt')
assert.equal(hubAddressMissing(''), null)
assert.equal(hubAddressMissing(null), null)

// Rejections carry a reason, because a disabled button with no sentence asks the operator to guess.
assert.deepEqual(cleanHubHost('  192.168.1.20  '), { host: '192.168.1.20', why: '' }, 'trimmed')
assert.deepEqual(cleanHubHost('gpu.lan:7447'), { host: 'gpu.lan:7447', why: '' })
assert.deepEqual(cleanHubHost('gpu.lan/'), { host: 'gpu.lan', why: '' }, 'a trailing slash is a typo, not a refusal')
assert.equal(cleanHubHost('http://gpu.lan:7447').host, null)
assert.match(cleanHubHost('http://gpu.lan').why, /paste a host, not a URL/)
assert.match(cleanHubHost('').why, /type the address/)
assert.match(cleanHubHost('  ').why, /type the address/)
assert.match(cleanHubHost('gpu lan').why, /no spaces/)
assert.match(cleanHubHost('gpu.lan:').why, /must be a port number/)
assert.match(cleanHubHost('gpu.lan:zenoh').why, /must be a port number/)
assert.match(cleanHubHost(':7447').why, /not a host name/)

// The judgement that is NOT here on purpose: the server refuses public/loopback when it
// GUESSES, and uses an explicit override verbatim.
assert.equal(cleanHubHost('203.0.113.9').why, '', 'a public address is the server\'s call, not ours')
assert.equal(cleanHubHost('localhost:7447').why, '', 'even loopback: the operator may be testing on one box')
