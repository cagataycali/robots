import assert from 'node:assert/strict'
import { armHosts, isChildOf } from '/tmp/armHosts.mjs'

// --- the live fleet, exactly as measured -----------------------------------
{
  const hosts = armHosts([
    { peer_id: 'so101-follower', joints: 0 },
    { peer_id: 'so101-leader', joints: 0 },
    { peer_id: 'so101-follower-twin', joints: 0 },
    { peer_id: 'so101-follower-twin__so101', joints: 6 },
  ])
  assert.deepEqual(Object.keys(hosts), ['so101-follower-twin'])
  assert.match(hosts['so101-follower-twin'].why, /hosts so101-follower-twin__so101/)
  assert.match(hosts['so101-follower-twin'].why, /this is the process/)
}
// A jointless REAL arm is not a host: it is a broken arm, which is a different sentence.
{
  const hosts = armHosts([{ peer_id: 'so101-leader', joints: 0 }])
  assert.deepEqual(hosts, {})
}
// --- evidence outranks structure -------------------------------------------
{
  // A parent that publishes its own joints stays offerable, child or no child.
  const hosts = armHosts([
    { peer_id: 'arm', joints: 6 },
    { peer_id: 'arm__wrist', joints: 3 },
  ])
  assert.deepEqual(hosts, {})
}
// --- several robots in one process ------------------------------------------
{
  const hosts = armHosts([
    { peer_id: 'sim', joints: null },
    { peer_id: 'sim__a', joints: 6 },
    { peer_id: 'sim__b', joints: 6 },
  ])
  assert.match(hosts['sim'].why, /hosts 2 robots \(sim__a, sim__b\)/)
}
// --- naming edges -----------------------------------------------------------
assert.equal(isChildOf('sim__a', 'sim'), true)
assert.equal(isChildOf('sim__', 'sim'), false, 'the half-formed name prune_peers also refuses')
assert.equal(isChildOf('sim-a', 'sim'), false, 'a single separator is a different peer, not a child')
assert.equal(isChildOf('simulator__a', 'sim'), false, 'prefix-but-not-parent must not adopt a stranger')
assert.deepEqual(armHosts(null), {}, 'no peers is not an error')
assert.deepEqual(armHosts([{ peer_id: '' }, null]), {}, 'junk in the list is skipped, not thrown on')
console.log('armHosts: all assertions passed')
