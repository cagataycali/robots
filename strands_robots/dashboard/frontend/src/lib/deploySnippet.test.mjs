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
