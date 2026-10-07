### Docs: the mesh page shows the `[mesh]` install line before its first example

`docs/learn/mesh/index.md` opened with a two-robot example but only named the
`[mesh]` extra in passing further down. On the `[sim-mujoco]` install the
previous page shows, that example logged one warning and printed
`mesh.alive == False` with no peers, which looks the same as a mesh whose
partner went offline. The page now opens with
`pip install 'strands-robots[sim-mujoco,mesh]'`, the way the bridges page does.
