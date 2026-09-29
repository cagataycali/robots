### Fixed: `instruction_read` follows the policy that ran, not the provider class

`run_policy` said `instruction_read=True` for a lerobot_local ACT checkpoint and
for `wbc`, neither of which reads the words. `Policy.reads_instruction` and
`instruction_free_actions` are plain class defaults now, so a provider can answer
per instance: `LerobotLocalPolicy` reads it off the loaded model's language input
(ACT, diffusion and VQ-BeT say no; SmolVLA, pi0 and MolmoAct2 say yes) and
`WBCPolicy` declares that its joint targets come from its velocity command. Every
rollout envelope carries the note for them, as it already did for the mock.
Closes #4159.
