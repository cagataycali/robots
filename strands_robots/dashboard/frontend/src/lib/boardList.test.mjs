import assert from 'node:assert/strict'
import { boardListEmptyLine } from '/tmp/boardList.mjs'

// --- the defect: a failed request rendered as a hardware verdict ---------------
{
  const v = boardListEmptyLine({ scanned: false, error: 'HTTP 401' })
  assert.equal(v.kind, 'unscanned')
  assert.match(v.message, /scan failed \(HTTP 401\)/)
  assert.match(v.message, /not because nothing is plugged in/, 'name the ambiguity, do not resolve it')
  assert.doesNotMatch(v.message, /no servo board detected/, 'a failed scan may never claim absent hardware')
}

// --- nothing asked yet: also not a verdict ------------------------------------
{
  const v = boardListEmptyLine({ scanned: false })
  assert.equal(v.kind, 'scanning')
  assert.match(v.message, /scanning USB/)
  assert.doesNotMatch(v.message, /no servo board/)
  // null and '' are the same as absent — an error state cleared to null must not
  // read as an error string.
  assert.equal(boardListEmptyLine({ scanned: false, error: null }).kind, 'scanning')
  assert.equal(boardListEmptyLine({ scanned: false, error: '  ' }).kind, 'scanning')
}

// --- the one case that MAY talk about hardware --------------------------------
{
  const v = boardListEmptyLine({ scanned: true })
  assert.equal(v.kind, 'detected')
  assert.match(v.message, /no servo board detected/)
  assert.match(v.message, /nothing on USB enumerated as a serial bus/, 'say what was looked for')
}

// --- a stale error alongside a successful scan is kept, not hidden ------------
{
  const v = boardListEmptyLine({ scanned: true, error: 'refresh timed out' })
  assert.equal(v.kind, 'detected')
  assert.match(v.message, /no servo board detected/)
  assert.match(v.message, /last refresh also reported: refresh timed out/)
}

console.log('boardList: all assertions passed')
